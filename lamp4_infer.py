import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import numpy as np


class PQCodebookModel(nn.Module):
    """PQ Codebook模型（用于推理，只用于量化，不训练）"""
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


class InferenceDataset(Dataset):
    """LaMP-4推理数据集类"""
    def __init__(self, json_path, encoder_model, encoder_tokenizer, llm_tokenizer, 
                 his_len=8, max_length=400):
        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.his_len = his_len
        self.max_length = max_length
        
        # 加载数据
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} data samples")
        
        # 占位符token
        self.placeholder_token = "<USR_EMB>"
        if self.placeholder_token not in llm_tokenizer.get_vocab():
            llm_tokenizer.add_tokens([self.placeholder_token])
            print(f"Added placeholder token: {self.placeholder_token}")
    
    def encode_text(self, text, device):
        """使用encoder实时编码文本"""
        self.encoder_model.eval()
        
        with torch.no_grad():
            inputs = self.encoder_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            outputs = self.encoder_model(**inputs)
            token_embeddings = outputs[0]  # (1, seq_len, hidden_size)
            
            # Mean pooling
            embeddings = mean_pooling(token_embeddings, inputs['attention_mask'])
            
            # L2归一化（contriever通常需要归一化）
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.squeeze(0)  # (1024,)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item.get("id", "")
        input_text = item.get("input", "")  # 需要生成标题的文章文本
        profile = item.get("profile", [])  # 用户历史profile
        
        return {
            "id": item_id,
            "input": input_text,
            "profile": profile
        }


def collate_fn(batch, encoder_model, encoder_tokenizer, device, his_len=8):
    """Collate function for DataLoader，实时编码profile"""
    ids = [item["id"] for item in batch]
    inputs = [item["input"] for item in batch]
    profiles = [item["profile"] for item in batch]
    
    # 实时编码profile
    user_embeddings_list = []
    for profile in profiles:
        user_embs_list = []
        for i, profile_item in enumerate(profile[:his_len]):
            text = profile_item.get("text", "")
            title = profile_item.get("title", "")
            
            # 组织成："The user wrote a news title '{title}' based on the following text: {text}"
            profile_text = f"The user wrote a news title '{title}' based on the following text: {text}"
            
            # 编码
            emb = encode_text_with_encoder(profile_text, encoder_model, encoder_tokenizer, device)
            user_embs_list.append(emb)
        
        # 如果不够，用零向量填充
        while len(user_embs_list) < his_len:
            user_embs_list.append(torch.zeros(1024, device=device))
        
        # 堆叠成tensor: (his_len, 1024)
        user_embs = torch.stack(user_embs_list[:his_len])
        user_embeddings_list.append(user_embs)
    
    user_embeddings = torch.stack(user_embeddings_list)  # (batch_size, his_len, 1024)
    
    return {
        "id": ids,
        "input": inputs,
        "user_embeddings": user_embeddings
    }


def encode_text_with_encoder(text, encoder_model, encoder_tokenizer, device):
    """使用encoder编码文本，返回在指定device上的tensor"""
    encoder_model.eval()
    
    with torch.no_grad():
        inputs = encoder_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        outputs = encoder_model(**inputs)
        token_embeddings = outputs[0]  # (1, seq_len, hidden_size)
        
        # Mean pooling
        embeddings = mean_pooling(token_embeddings, inputs['attention_mask'])
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.squeeze(0)  # (1024,)，在device上


def load_models(codebook_path, mlp_path, llm_path, encoder_path, device="cuda:0"):
    """加载所有模型"""
    print(f"Loading models on {device}...")
    
    # 加载encoder
    print(f"  Loading encoder from {encoder_path}...")
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
    encoder_model = AutoModel.from_pretrained(encoder_path, trust_remote_code=True)
    encoder_model.to(device)
    encoder_model.eval()
    
    # 加载PQ codebook
    print(f"  Loading PQ codebook from {codebook_path}...")
    pq_codebook_model = PQCodebookModel(codebook_path, device=device)
    pq_codebook_model.eval()
    
    # 加载MLP
    print(f"  Loading MLP from {mlp_path}...")
    mlp_checkpoint = torch.load(mlp_path, map_location=device)
    mlp_model = MLPProjection(
        input_dim=mlp_checkpoint["input_dim"],
        hidden_dim=mlp_checkpoint["hidden_dim"],
        output_dim=mlp_checkpoint["output_dim"]
    )
    mlp_model.load_state_dict(mlp_checkpoint["mlp"])
    mlp_model.to(device)
    mlp_model.eval()
    
    # 加载LLM
    print(f"  Loading LLM from {llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device}  # 只使用指定的device
    )
    llm_model.eval()
    
    # 确保占位符在tokenizer中
    placeholder_token = "<USR_EMB>"
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])
        print(f"  Added placeholder token: {placeholder_token}")
    
    print("All models loaded!")
    return encoder_model, encoder_tokenizer, pq_codebook_model, mlp_model, llm_model, tokenizer


def inference_batch(batch, encoder_model, encoder_tokenizer, pq_codebook_model, mlp_model, 
                   llm_model, tokenizer, his_len=8, max_new_tokens=512, device="cuda:0"):
    """
    对batch进行推理
    
    Args:
        batch: 包含id, input, user_embeddings的字典
        encoder_model: Encoder模型（用于实时编码，虽然这里已经编码好了）
        encoder_tokenizer: Encoder的tokenizer
        pq_codebook_model: PQ codebook模型
        mlp_model: MLP投影模型
        llm_model: LLM模型
        tokenizer: LLM的tokenizer
        his_len: 历史长度
        max_new_tokens: 最大生成token数
        device: 设备
    
    Returns:
        generated_texts: 生成的文本列表
    """
    placeholder_token = "<USR_EMB>"
    inputs = batch["input"]
    user_embeddings = batch["user_embeddings"].to(device)  # (batch_size, his_len, 1024)
    batch_size = len(inputs)
    
    # PQ量化（不需要梯度）
    with torch.no_grad():
        quantized_embs, _ = pq_codebook_model(user_embeddings)  # (batch_size, his_len, 1024)
    
    # MLP投影到LLM维度（不需要梯度）
    with torch.no_grad():
        llm_embs = mlp_model(quantized_embs)  # (batch_size, his_len, llm_dim)
    
    # 获取LLM的embedding层和数据类型
    llm_embeddings = llm_model.get_input_embeddings()
    model_dtype = next(llm_model.parameters()).dtype
    
    # 确保llm_embs使用正确的数据类型
    llm_embs = llm_embs.to(dtype=model_dtype)
    
    # 构建每个样本的prompt并tokenize
    input_ids_list = []
    placeholder_positions_list = []
    
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    for i in range(batch_size):
        input_text = inputs[i]
        
        # 构建prompt
        placeholder_str = " ".join([placeholder_token] * his_len)
        user_prompt_text = (
            "You are asked to write a news title based on a text and user prototype, "
            f"trying to imitate the user's writing style. The user prototype are {placeholder_str}.\n"
            f"The text is: {input_text}\n"
            "Please write a news title for this text."
        )
        
        # 使用chat template
        messages = [
            {"role": "user", "content": user_prompt_text}
        ]
        
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs_tokenized = tokenizer(
            formatted,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=400
        )
        
        input_ids = inputs_tokenized["input_ids"].squeeze(0)  # (seq_len,)
        input_ids_list.append(input_ids)
        
        # 找到占位符位置
        placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()
        placeholder_positions_list.append(placeholder_positions)
    
    # Padding到相同长度
    max_seq_len = max(ids.size(0) for ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []
    
    for input_ids in input_ids_list:
        seq_len = input_ids.size(0)
        padding_length = max_seq_len - seq_len
        
        # Left padding
        padded = torch.cat([
            torch.full((padding_length,), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, dtype=input_ids.dtype),
            input_ids
        ])
        padded_input_ids.append(padded)
        
        # Attention mask
        attention_mask = torch.cat([
            torch.zeros(padding_length, dtype=torch.long),
            torch.ones(seq_len, dtype=torch.long)
        ])
        attention_masks.append(attention_mask)
    
    input_ids_batch = torch.stack(padded_input_ids).to(device)  # (batch_size, max_seq_len)
    attention_mask_batch = torch.stack(attention_masks).to(device)  # (batch_size, max_seq_len)
    
    # 获取base embeddings
    input_embs = llm_embeddings(input_ids_batch)  # (batch_size, max_seq_len, llm_dim)
    input_embs = input_embs.to(dtype=model_dtype)
    
    # 替换占位符位置的embeddings
    for i in range(batch_size):
        placeholder_positions = placeholder_positions_list[i]
        if len(placeholder_positions) >= his_len:
            # 调整位置（因为left padding）
            padding_length = max_seq_len - input_ids_list[i].size(0)
            adjusted_positions = [pos + padding_length for pos in placeholder_positions[:his_len]]
            
            for j, pos in enumerate(adjusted_positions):
                if pos < max_seq_len:
                    input_embs[i, pos] = llm_embs[i, j]
    
    # 批量生成
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs_embeds=input_embs,
            attention_mask=attention_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码每个样本（只取新生成的部分）
    generated_texts = []
    for i in range(batch_size):
        input_seq_len = input_ids_list[i].size(0)
        generated_ids = outputs[i]  # 只取新生成的部分
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="LaMP-4推理：使用训练好的PQ codebook和MLP")
    
    # 路径参数
    parser.add_argument("--data_path", type=str,
                       default="/home/notebook/code/group/wangliang/lamp_data/LaMP_4/dev/dev_questions.json",
                       help="输入的JSON数据文件路径")
    parser.add_argument("--encoder_path", type=str, required=True,
                       help="Encoder模型路径（Contriever）")
    parser.add_argument("--codebook_path", type=str, required=True,
                       help="训练好的PQ codebook模型路径")
    parser.add_argument("--mlp_path", type=str, required=True,
                       help="训练好的MLP模型路径")
    parser.add_argument("--llm_path", type=str, required=True,
                       help="LLM模型路径")
    parser.add_argument("--output_path", type=str,
                       default="/mnt/workspace/wangliang/alignx/lamp4_inference_results_qwen_bert.json",
                       help="输出结果文件路径")
    
    # 推理参数
    parser.add_argument("--his_len", type=int, default=8,
                       help="Number of historical profile items to use")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=40,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备 (cuda:0, cpu, etc.)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="处理的样本数量（None表示处理全部）")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers for DataLoader")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    encoder_model, encoder_tokenizer, pq_codebook_model, mlp_model, llm_model, tokenizer = load_models(
        codebook_path=args.codebook_path,
        mlp_path=args.mlp_path,
        llm_path=args.llm_path,
        encoder_path=args.encoder_path,
        device=device
    )
    
    # 创建数据集
    print(f"\nCreating dataset...")
    dataset = InferenceDataset(
        json_path=args.data_path,
        encoder_model=encoder_model,
        encoder_tokenizer=encoder_tokenizer,
        llm_tokenizer=tokenizer,
        his_len=args.his_len,
        max_length=400
    )
    
    # 如果指定了num_samples，只取前N个
    if args.num_samples:
        dataset.data = dataset.data[:args.num_samples]
        print(f"Processing first {len(dataset)} samples")
    else:
        print(f"Processing all {len(dataset)} samples")
    
    # 创建自定义collate_fn（需要传入encoder）
    def collate_fn_wrapper(batch):
        return collate_fn(batch, encoder_model, encoder_tokenizer, device, args.his_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        num_workers=args.num_workers,
        pin_memory=False  # 关闭pin_memory，因为embeddings已经在CPU上，会在inference_batch中移到GPU
    )
    
    # 推理
    print(f"\nStarting batch inference (batch_size={args.batch_size})...")
    results = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferencing")):
        try:
            generated_texts = inference_batch(
                batch=batch,
                encoder_model=encoder_model,
                encoder_tokenizer=encoder_tokenizer,
                pq_codebook_model=pq_codebook_model,
                mlp_model=mlp_model,
                llm_model=llm_model,
                tokenizer=tokenizer,
                his_len=args.his_len,
                max_new_tokens=args.max_new_tokens,
                device=device
            )
            
            # 保存结果
            for i, generated_text in enumerate(generated_texts):
                global_idx = batch_idx * args.batch_size + i
                if global_idx < len(dataset):
                    results.append({
                        "id": batch["id"][i],
                        "input": batch["input"][i],
                        "generated": generated_text
                    })
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # 为这个batch的所有样本添加错误标记
            for i in range(len(batch["input"])):
                global_idx = batch_idx * args.batch_size + i
                if global_idx < len(dataset):
                    results.append({
                        "id": batch["id"][i],
                        "input": batch["input"][i],
                        "generated": f"ERROR: {str(e)}"
                    })
    
    # 保存结果
    print(f"\nSaving results to {args.output_path}...")
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(results)} results to {args.output_path}")
    
    # 打印一些统计信息
    print(f"\nInference completed!")
    print(f"  Total samples processed: {len(results)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  History length: {args.his_len}")
    print(f"  Max new tokens: {args.max_new_tokens}")


if __name__ == "__main__":
    main()
