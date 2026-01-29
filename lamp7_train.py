import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import numpy as np


class PQCodebookModel(nn.Module):
    """PQ Codebook模型（用于Stage 2，只用于量化，不训练）"""
    def __init__(self, codebook_path, device="cpu"):
        super().__init__()
        # 加载训练好的PQ codebook
        checkpoint = torch.load(codebook_path, map_location=device)
        
        if "codebooks" in checkpoint:
            # 将codebooks转换为ParameterList
            codebooks_list = []
            for cb in checkpoint["codebooks"]:
                if isinstance(cb, torch.Tensor):
                    codebooks_list.append(nn.Parameter(cb.to(device), requires_grad=False))
                else:
                    codebooks_list.append(nn.Parameter(torch.tensor(cb, device=device), requires_grad=False))
            
            self.codebooks = nn.ParameterList(codebooks_list)
            self.num_subspaces = checkpoint.get("num_subspaces", len(self.codebooks))
            self.subspace_dim = checkpoint.get("subspace_dim", self.codebooks[0].shape[1] if len(self.codebooks) > 0 else None)
            self.codebook_size = self.codebooks[0].shape[0]
            self.emb_dim = self.num_subspaces * self.subspace_dim
        else:
            raise ValueError(f"Checkpoint must contain 'codebooks' key. Found keys: {checkpoint.keys()}")
        
        print(f"Loaded PQ codebook: {self.num_subspaces} subspaces, each {self.subspace_dim}D, {self.codebook_size} entries per subspace")
    
    def quantize(self, embeddings):
        """
        Product Quantization: 将embeddings分成多个子空间，每个子空间独立量化
        embeddings: (batch, seq_len, emb_dim)
        返回: (batch, seq_len, emb_dim) 量化后的embeddings, (batch, seq_len, num_subspaces) 每个子空间的索引
        """
        batch_size, seq_len, emb_dim = embeddings.shape
        flat_embs = embeddings.view(-1, emb_dim)  # (batch * seq_len, emb_dim)
        
        # 将embeddings分成子空间
        subspace_embs = flat_embs.view(-1, self.num_subspaces, self.subspace_dim)
        
        quantized_parts = []
        all_indices = []
        
        # 对每个子空间独立量化
        for i, codebook in enumerate(self.codebooks):
            subspace = subspace_embs[:, i, :]  # (batch * seq_len, subspace_dim)
            
            # 确保codebook在正确的设备上
            codebook = codebook.to(subspace.device)
            
            # 计算距离: (batch * seq_len, codebook_size)
            distances = torch.cdist(subspace, codebook, p=2)
            
            # 找到最近邻索引
            indices = torch.argmin(distances, dim=-1)  # (batch * seq_len,)
            
            # 从codebook中获取量化后的embeddings
            quantized = codebook[indices]  # (batch * seq_len, subspace_dim)
            
            quantized_parts.append(quantized)
            all_indices.append(indices)
        
        # 拼接所有子空间的量化结果
        quantized = torch.cat(quantized_parts, dim=-1)  # (batch * seq_len, emb_dim)
        quantized = quantized.view(batch_size, seq_len, emb_dim)
        
        # 所有子空间的索引: (batch * seq_len, num_subspaces)
        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = all_indices.view(batch_size, seq_len, self.num_subspaces)
        
        return quantized, all_indices
    
    def forward(self, embeddings):
        """量化embeddings（不训练，只用于推理）"""
        quantized, indices = self.quantize(embeddings)
        return quantized, indices


class MLPProjection(nn.Module):
    """MLP投影层：将量化后的embeddings投影到LLM维度"""
    def __init__(self, input_dim=1024, hidden_dim=None, output_dim=4096):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim  # 默认使用output_dim作为hidden_dim
        
        # 两层MLP: input_dim -> hidden_dim -> output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        print(f"Initialized MLP: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, seq_len, output_dim)
        """
        return self.mlp(x)


def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling for sentence embeddings"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_query_LaMP_7(inp):
    """从 input 中提取查询文本（LaMP-7格式）"""
    substr = "before or after it: "
    plc = inp.find(substr)
    if plc == -1:
        # 如果没有找到分隔符，返回整个input
        return inp.strip()
    query = inp[plc + len(substr):].strip()
    return query


def ensure_chat_template(tokenizer, model_path):
    """
    确保tokenizer有chat_template，如果没有则从文件中加载
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        return
    
    # 尝试从文件中加载chat_template
    for name in ("chat_template.jinja", "chat_template.txt", "chat_template.json"):
        template_path = os.path.join(model_path, name)
        if os.path.exists(template_path):
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()
                print(f"✅ 从文件加载 chat_template: {template_path}")
                return
            except Exception as e:
                print(f"⚠️  读取 chat_template 失败: {template_path} -> {e}")


def format_chat_template(messages, tokenizer):
    """
    格式化chat template，如果没有chat_template则手动构建Qwen格式
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        # 使用tokenizer的chat_template
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"⚠️  apply_chat_template 失败: {e}，使用fallback")
            # 如果apply_chat_template失败，使用fallback
            pass
    
    # 手动构建Qwen格式的chat template（fallback）
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(formatted_parts)


class LaMP7Dataset(Dataset):
    """LaMP-7数据集：从JSON读取，实时编码profile"""
    def __init__(self, json_path, outputs_path, encoder_model, encoder_tokenizer, llm_tokenizer, 
                 his_len=8, max_length=2048, device="cpu"):
        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.his_len = his_len
        self.max_length = max_length
        self.device = device
        
        # 加载数据
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 加载ground truth outputs
        print(f"Loading ground truth from {outputs_path}...")
        with open(outputs_path, 'r', encoding='utf-8') as f:
            outputs_data = json.load(f)
        
        # 构建id到output的映射
        self.id_to_output = {}
        if "golds" in outputs_data:
            for item in outputs_data["golds"]:
                item_id = item.get("id", "")
                output = item.get("output", "")
                if item_id:
                    self.id_to_output[item_id] = output
        else:
            # 如果没有golds字段，直接遍历
            for item in outputs_data:
                item_id = item.get("id", "")
                output = item.get("output", "")
                if item_id:
                    self.id_to_output[item_id] = output
        
        print(f"Loaded {len(self.data)} data samples")
        print(f"Loaded {len(self.id_to_output)} ground truth outputs")
        
        # 占位符token（用于定位soft token插入位置）
        self.placeholder_token = "<USR_EMB>"
        # 确保占位符在LLM的tokenizer词表中
        if self.placeholder_token not in llm_tokenizer.get_vocab():
            llm_tokenizer.add_tokens([self.placeholder_token])
            print(f"Added placeholder token to LLM tokenizer: {self.placeholder_token}")
    
    def encode_text(self, text):
        """使用encoder实时编码文本"""
        self.encoder_model.eval()
        encoder_device = next(self.encoder_model.parameters()).device
        
        with torch.no_grad():
            inputs = self.encoder_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(encoder_device)
            
            outputs = self.encoder_model(**inputs)
            token_embeddings = outputs[0]  # (1, seq_len, hidden_size)
            
            # Mean pooling
            embeddings = mean_pooling(token_embeddings, inputs['attention_mask'])
            
            # L2归一化（contriever通常需要归一化）
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.squeeze(0).cpu()  # (1024,)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item.get("id", "")  # 样本ID
        input_text = item.get("input", "")  # 需要改写的推文输入
        profile = item.get("profile", [])  # 用户历史profile
        
        # 通过id从ground truth中获取output
        output = self.id_to_output.get(item_id, "")  # ground truth改写后的推文
        
        # 构建profile文本并编码（LaMP-7格式：直接使用text字段）
        user_embs_list = []
        for i, profile_item in enumerate(profile[:self.his_len]):
            text = profile_item.get("text", "")
            
            # LaMP-7: 直接使用text，不需要组合title
            if text.strip():
                # 编码
                emb = self.encode_text(text)
                user_embs_list.append(emb)
            else:
                # 如果text为空，使用零向量
                user_embs_list.append(torch.zeros(1024))
        
        # 如果不够，用零向量填充
        while len(user_embs_list) < self.his_len:
            user_embs_list.append(torch.zeros(1024))
        
        # 堆叠成tensor: (his_len, 1024)
        user_embs = torch.stack(user_embs_list[:self.his_len])
        
        return {
            "input": input_text,  # 需要改写的推文输入
            "output": output,  # ground truth改写后的推文
            "user_embeddings": user_embs  # (his_len, 1024)
        }


class LaMP7Trainer:
    """LaMP-7训练器：训练MLP，LLM冻结"""
    def __init__(self, llm_model, mlp_model, pq_codebook_model, tokenizer, accelerator, his_len=8, max_length=2048):
        self.llm_model = llm_model
        self.mlp_model = mlp_model
        self.pq_codebook_model = pq_codebook_model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.his_len = his_len
        self.max_length = max_length
        self.placeholder_token = "<USR_EMB>"
        
        # LLM冻结
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        # PQ codebook冻结（已经是requires_grad=False）
        # MLP需要训练
        for param in self.mlp_model.parameters():
            param.requires_grad = True
    
    def compute_loss(self, batch):
        """计算loss"""
        inputs = batch["input"]  # 需要改写的推文输入列表
        outputs = batch.get("output", [""] * len(inputs))  # ground truth改写后的推文列表（如果有）
        user_embeddings = batch["user_embeddings"]  # (batch_size, his_len, 1024)
        
        device = user_embeddings.device
        batch_size = len(inputs)
        his_len = user_embeddings.size(1)
        
        # 对user_embeddings进行PQ量化（不需要梯度）
        with torch.no_grad():
            quantized_embs, _ = self.pq_codebook_model(user_embeddings)  # (batch, his_len, 1024)
        
        # 通过MLP投影到LLM维度（需要梯度）
        llm_embs = self.mlp_model(quantized_embs)  # (batch, his_len, llm_dim)
        
        # 构建输入
        input_ids_list = []
        labels_list = []
        placeholder_positions_list = []
        
        placeholder_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        
        for i in range(batch_size):
            input_text = inputs[i]
            output_text = outputs[i] if i < len(outputs) else ""
            
            # 从input中提取查询文本（LaMP-7格式）
            query_text = get_query_LaMP_7(input_text)
            
            # 构建prompt（使用soft prompt方式，不在文本中插入占位符）
            # 在embedding层面直接替换，使用单个标记token来定位插入位置
            if his_len > 0:
                # 使用单个占位符标记来定位，后续会用his_len个soft tokens替换
                user_prompt_text = (
                    f"User style embedding: {self.placeholder_token}\n"
                    "Based on the user's style embedding provided above, please paraphrase the user's input tweet without any explanation before or after it.\n"
                    f"{query_text}"
                )
            else:
                user_prompt_text = (
                    "Please paraphrase the user's input tweet without any explanation before or after it.\n"
                    "Please generate it in the following format: {'tweet': 'generated tweet'} without explanation, and use only English.\n"
                    f"{query_text}"
                )
            
            # 使用chat template，包含user message和assistant回复（如果有ground truth）
            if output_text:
                messages = [
                    {"role": "user", "content": user_prompt_text},
                    {"role": "assistant", "content": output_text}
                ]
            else:
                messages = [
                    {"role": "user", "content": user_prompt_text}
                ]
            
            # 应用chat template（如果没有chat_template则使用fallback）
            formatted = format_chat_template(messages, self.tokenizer)
            
            # Tokenize
            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
            
            # 找到占位符的位置（用于定位soft token插入位置）
            placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()
            
            # 记录插入位置和数量
            if len(placeholder_positions) > 0 and his_len > 0:
                # 占位符的位置，后续会在这个位置插入his_len个soft tokens
                insert_start_pos = placeholder_positions[0]
                placeholder_positions_list.append((insert_start_pos, his_len))
            else:
                placeholder_positions_list.append((None, 0))
                insert_start_pos = None
            
            # 创建labels
            labels = input_ids.clone()
            
            if output_text:
                # 如果有ground truth，找到assistant回复的起始位置
                # 对于Qwen格式，通常是"<|im_start|>assistant\n"之后
                assistant_start = None
                for j in range(len(input_ids) - 1):
                    # 查找assistant标记
                    if input_ids[j].item() == self.tokenizer.convert_tokens_to_ids("<|im_start|>"):
                        if j + 1 < len(input_ids):
                            # 检查下一个token是否是assistant
                            next_token = input_ids[j + 1].item()
                            # 可能是assistant的token id
                            if "assistant" in self.tokenizer.convert_ids_to_tokens([next_token])[0].lower():
                                assistant_start = j + 3  # 跳过<|im_start|>assistant\n
                                break
                
                if assistant_start is None:
                    # 如果找不到，尝试其他方法：找到占位符之后的位置
                    if insert_start_pos is not None:
                        assistant_start = insert_start_pos + his_len + 1
                    else:
                        assistant_start = len(input_ids) // 2
                
                # 只计算assistant回复部分的loss
                labels[:assistant_start] = -100
            else:
                # 如果没有ground truth，不计算loss
                labels[:] = -100
            
            # 占位符位置和soft token位置都不计算loss
            if insert_start_pos is not None and his_len > 0:
                # 占位符本身和后续的soft token位置都不计算loss
                for j in range(his_len):
                    pos = insert_start_pos + j
                    if pos < len(labels):
                        labels[pos] = -100
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Padding（left padding，因为LLM是decoder-only）
        max_seq_len = max(ids.size(0) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            seq_len = input_ids.size(0)
            padding_length = max_seq_len - seq_len
            
            # Left padding
            padded_input = torch.cat([
                torch.full((padding_length,), self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id, dtype=input_ids.dtype),
                input_ids
            ])
            padded_input_ids.append(padded_input)
            
            padded_label = torch.cat([
                torch.full((padding_length,), -100, dtype=labels.dtype),
                labels
            ])
            padded_labels.append(padded_label)
            
            # Attention mask
            attention_mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long),
                torch.ones(seq_len, dtype=torch.long)
            ])
            attention_masks.append(attention_mask)
        
        input_ids_batch = torch.stack(padded_input_ids).to(device)  # (batch_size, max_seq_len)
        labels_batch = torch.stack(padded_labels).to(device)  # (batch_size, max_seq_len)
        attention_mask_batch = torch.stack(attention_masks).to(device)  # (batch_size, max_seq_len)
        
        # 获取LLM的embedding层
        llm_embeddings = self.llm_model.get_input_embeddings()
        input_embs = llm_embeddings(input_ids_batch)  # (batch_size, max_seq_len, llm_dim)
        
        # 替换占位符位置的embeddings（使用soft prompt方式）
        for i in range(batch_size):
            insert_start_pos, num_tokens = placeholder_positions_list[i]
            if insert_start_pos is not None and num_tokens > 0:
                # 调整位置（因为left padding）
                padding_length = max_seq_len - input_ids_list[i].size(0)
                adjusted_start_pos = insert_start_pos + padding_length
                
                # 在占位符位置插入his_len个soft tokens
                # 如果序列长度足够，直接替换；否则需要扩展序列
                if adjusted_start_pos + num_tokens <= max_seq_len:
                    # 直接替换占位符及其后续位置的embeddings
                    for j in range(num_tokens):
                        if adjusted_start_pos + j < max_seq_len:
                            input_embs[i, adjusted_start_pos + j] = llm_embs[i, j]
                else:
                    # 如果位置超出，只替换能放下的部分
                    available_slots = max_seq_len - adjusted_start_pos
                    for j in range(min(available_slots, num_tokens)):
                        input_embs[i, adjusted_start_pos + j] = llm_embs[i, j]
        
        # 前向传播
        outputs = self.llm_model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask_batch,
            labels=labels_batch
        )
        
        loss = outputs.loss
        
        return loss


def train(args):
    # 初始化accelerate
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        print("=" * 80)
        print("LaMP-7 Training: MLP Projection with PQ Codebook")
        print("=" * 80)
    
    # 加载encoder（Contriever）
    if accelerator.is_local_main_process:
        print(f"\nLoading encoder (Contriever) from {args.encoder_path}...")
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, trust_remote_code=True)
    encoder_model = AutoModel.from_pretrained(args.encoder_path, trust_remote_code=True)
    encoder_model.eval()
    # encoder不需要训练，所以不需要prepare
    
    # 加载LLM
    if accelerator.is_local_main_process:
        print(f"\nLoading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    # 确保chat_template被正确加载
    ensure_chat_template(tokenizer, args.llm_path)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    llm_model.eval()
    
    # 加载PQ codebook
    if accelerator.is_local_main_process:
        print(f"\nLoading PQ codebook from {args.codebook_path}...")
    pq_codebook_model = PQCodebookModel(codebook_path=args.codebook_path, device=accelerator.device)
    pq_codebook_model.eval()
    
    # 创建MLP
    if accelerator.is_local_main_process:
        print(f"\nCreating MLP projection layer...")
    llm_dim = llm_model.config.hidden_size
    hidden_dim = args.hidden_dim if args.hidden_dim else llm_dim
    
    mlp_model = MLPProjection(
        input_dim=pq_codebook_model.emb_dim,  # 1024
        hidden_dim=hidden_dim,
        output_dim=llm_dim
    )
    
    if accelerator.is_local_main_process:
        print(f"MLP: {pq_codebook_model.emb_dim} -> {hidden_dim} -> {llm_dim}")
    
    # 创建数据集
    dataset = LaMP7Dataset(
        json_path=args.data_path,
        outputs_path=args.outputs_path,
        encoder_model=encoder_model,
        encoder_tokenizer=encoder_tokenizer,
        llm_tokenizer=tokenizer,
        his_len=args.his_len,
        max_length=args.max_length,
        device="cpu"  # 数据在CPU上，编码时再移到GPU
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # 因为需要实时编码，所以不用多进程
    )
    
    # 创建训练器
    trainer = LaMP7Trainer(
        llm_model=llm_model,
        mlp_model=mlp_model,
        pq_codebook_model=pq_codebook_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        his_len=args.his_len,
        max_length=args.max_length
    )
    
    # 创建优化器（只优化MLP）
    optimizer = torch.optim.AdamW(
        mlp_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 使用accelerate准备
    mlp_model, optimizer, dataloader = accelerator.prepare(
        mlp_model, optimizer, dataloader
    )
    
    # 将encoder移到accelerator管理的设备上
    encoder_model = encoder_model.to(accelerator.device)
    
    # 将PQ codebook移到accelerator管理的设备上
    pq_codebook_model = pq_codebook_model.to(accelerator.device)
    
    # 将LLM移到accelerator管理的设备上（如果需要）
    if not hasattr(llm_model, 'hf_device_map'):
        llm_model = llm_model.to(accelerator.device)
    
    # 训练循环
    if accelerator.is_local_main_process:
        print(f"\nStarting training...")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  History length: {args.his_len}")
        print(f"  Max length: {args.max_length}")
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        mlp_model.train()
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            loss = trainer.compute_loss(batch)
            
            accelerator.backward(loss)
            
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(mlp_model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            global_step += 1
            
            # 更新进度条
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 保存checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if accelerator.is_local_main_process:
                    # 获取unwrapped模型
                    unwrapped_mlp = accelerator.unwrap_model(mlp_model)
                    
                    # 获取实际维度
                    actual_input_dim = unwrapped_mlp.mlp[0].in_features
                    actual_hidden_dim = unwrapped_mlp.mlp[0].out_features
                    actual_output_dim = unwrapped_mlp.mlp[-1].out_features
                    
                    checkpoint_dict = {
                        "mlp": unwrapped_mlp.state_dict(),
                        "step": global_step,
                        "epoch": epoch + 1,
                        "input_dim": actual_input_dim,
                        "hidden_dim": actual_hidden_dim,
                        "output_dim": actual_output_dim,
                        "his_len": args.his_len
                    }
                    
                    checkpoint_path = os.path.join(args.output_dir, f"lamp7_checkpoint-{global_step}.pt")
                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(checkpoint_dict, checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")
        
        # 每个epoch结束后保存
        if accelerator.is_local_main_process:
            unwrapped_mlp = accelerator.unwrap_model(mlp_model)
            
            actual_input_dim = unwrapped_mlp.mlp[0].in_features
            actual_hidden_dim = unwrapped_mlp.mlp[0].out_features
            actual_output_dim = unwrapped_mlp.mlp[-1].out_features
            
            checkpoint_dict = {
                "mlp": unwrapped_mlp.state_dict(),
                "step": global_step,
                "epoch": epoch + 1,
                "input_dim": actual_input_dim,
                "hidden_dim": actual_hidden_dim,
                "output_dim": actual_output_dim,
                "his_len": args.his_len
            }
            
            checkpoint_path = os.path.join(args.output_dir, f"lamp7_checkpoint-epoch{epoch + 1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"\nSaved epoch checkpoint to {checkpoint_path}")
    
    if accelerator.is_local_main_process:
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="LaMP-7 Training: MLP with PQ Codebook")
    
    # 数据参数
    parser.add_argument("--data_path", type=str,
                       default="/home/notebook/code/group/wangliang/lamp_data/LaMP_7/train/train_questions.json",
                       help="LaMP-7 JSON数据文件路径")
    parser.add_argument("--outputs_path", type=str,
                       default="/home/notebook/code/group/wangliang/lamp_data/LaMP_7/train/train_outputs.json",
                       help="LaMP-7 ground truth outputs JSON文件路径")
    parser.add_argument("--encoder_path", type=str, required=True,
                       help="Encoder模型路径（Contriever）")
    
    # 模型参数
    parser.add_argument("--codebook_path", type=str, required=True,
                       help="训练好的PQ codebook模型路径")
    parser.add_argument("--llm_path", type=str, required=True,
                       help="LLM模型路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--his_len", type=int, default=8,
                       help="Number of historical profile items to use")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=300,
                       help="Maximum sequence length")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension for MLP (default: same as LLM dim)")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps (0 to disable)")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()

