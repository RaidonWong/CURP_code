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


def encode_text_with_encoder(text, encoder_model, encoder_tokenizer, device):
    """使用encoder实时编码文本"""
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
        
        # L2归一化（contriever通常需要归一化）
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.squeeze(0)  # (1024,)


class LaMP7InferenceDataset(Dataset):
    """LaMP-7推理数据集：从JSON读取，实时编码profile"""
    def __init__(self, json_path, encoder_model, encoder_tokenizer, llm_tokenizer, 
                 his_len=8, device="cpu"):
        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.his_len = his_len
        self.device = device
        
        # 加载数据
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} data samples")
        
        # 占位符token
        self.placeholder_token = "<USR_EMB>"
        # 确保占位符在LLM的tokenizer词表中
        if self.placeholder_token not in llm_tokenizer.get_vocab():
            llm_tokenizer.add_tokens([self.placeholder_token])
            print(f"Added placeholder token to LLM tokenizer: {self.placeholder_token}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        item_id = item.get("id", "")  # 样本ID
        input_text = item.get("input", "")  # 需要改写的推文输入
        profile = item.get("profile", [])  # 用户历史profile
        
        # 构建profile文本并编码（LaMP-7格式：直接使用text字段）
        user_embs_list = []
        for i, profile_item in enumerate(profile[:self.his_len]):
            text = profile_item.get("text", "")
            
            # LaMP-7: 直接使用text，不需要组合title
            if text.strip():
                # 编码
                emb = encode_text_with_encoder(text, self.encoder_model, self.encoder_tokenizer, self.device)
                user_embs_list.append(emb.cpu())
            else:
                # 如果text为空，使用零向量
                user_embs_list.append(torch.zeros(1024))
        
        # 如果不够，用零向量填充
        while len(user_embs_list) < self.his_len:
            user_embs_list.append(torch.zeros(1024))
        
        # 堆叠成tensor: (his_len, 1024)
        user_embs = torch.stack(user_embs_list[:self.his_len])
        
        return {
            "id": item_id,
            "input": input_text,  # 需要改写的推文输入
            "user_embeddings": user_embs  # (his_len, 1024)
        }


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
    ensure_chat_template(tokenizer, llm_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    llm_model.eval()
    
    # 确保占位符在tokenizer中
    placeholder_token = "<USR_EMB>"
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])
        print(f"  Added placeholder token: {placeholder_token}")
    
    print("✅ All models loaded!")
    return encoder_model, encoder_tokenizer, pq_codebook_model, mlp_model, llm_model, tokenizer


def inference_batch(batch, encoder_model, encoder_tokenizer, pq_codebook_model, mlp_model, 
                   llm_model, tokenizer, his_len=8, max_new_tokens=128, device="cuda:0"):
    """
    对batch进行推理
    
    Args:
        batch: 包含id, input, user_embeddings的字典
        encoder_model: Encoder模型
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
        
        # 从input中提取查询文本（LaMP-7格式）
        query_text = get_query_LaMP_7(input_text)
        
        # 构建prompt（使用soft prompt方式，不在文本中插入占位符）
        if his_len > 0:
            # 使用单个占位符标记来定位，后续会用his_len个soft tokens替换
            user_prompt_text = (
                f"User style embedding: {placeholder_token}\n"
                "Based on the user's style embedding provided above, please paraphrase the user's input tweet without any explanation before or after it.\n"
                f"{query_text}"
            )
        else:
            user_prompt_text = (
                "Please paraphrase the user's input tweet without any explanation before or after it.\n"
                "Please generate it in the following format: {'tweet': 'generated tweet'} without explanation, and use only English.\n"
                f"{query_text}"
            )
        
        # 使用chat template
        messages = [{"role": "user", "content": user_prompt_text}]
        
        # 应用chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: 手动构建Qwen格式
            formatted = f"<|im_start|>user\n{user_prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        )
        
        input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
        
        # 找到占位符的位置
        placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)[0].tolist()
        
        if len(placeholder_positions) > 0 and his_len > 0:
            insert_start_pos = placeholder_positions[0]
            placeholder_positions_list.append((insert_start_pos, his_len))
        else:
            placeholder_positions_list.append((None, 0))
        
        input_ids_list.append(input_ids)
    
    # Padding（left padding，因为LLM是decoder-only）
    max_seq_len = max(ids.size(0) for ids in input_ids_list)
    
    padded_input_ids = []
    attention_masks = []
    
    for input_ids in input_ids_list:
        seq_len = input_ids.size(0)
        padding_length = max_seq_len - seq_len
        
        # Left padding
        padded_input = torch.cat([
            torch.full((padding_length,), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, dtype=input_ids.dtype),
            input_ids
        ])
        padded_input_ids.append(padded_input)
        
        # Attention mask
        attention_mask = torch.cat([
            torch.zeros(padding_length, dtype=torch.long),
            torch.ones(seq_len, dtype=torch.long)
        ])
        attention_masks.append(attention_mask)
    
    input_ids_batch = torch.stack(padded_input_ids).to(device)  # (batch_size, max_seq_len)
    attention_mask_batch = torch.stack(attention_masks).to(device)  # (batch_size, max_seq_len)
    
    # 获取LLM的embedding层
    input_embs = llm_embeddings(input_ids_batch)  # (batch_size, max_seq_len, llm_dim)
    
    # 替换占位符位置的embeddings（使用soft prompt方式）
    for i in range(batch_size):
        insert_start_pos, num_tokens = placeholder_positions_list[i]
        if insert_start_pos is not None and num_tokens > 0:
            # 调整位置（因为left padding）
            padding_length = max_seq_len - input_ids_list[i].size(0)
            adjusted_start_pos = insert_start_pos + padding_length
            
            # 在占位符位置插入his_len个soft tokens
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
    
    # 生成
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs_embeds=input_embs,
            attention_mask=attention_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码生成结果
    generated_texts = []
    for i, output_ids in enumerate(outputs):
        # 找到输入部分的长度
        input_len = input_ids_list[i].size(0)
        # 提取生成的部分（去掉padding）
        padding_length = max_seq_len - input_len
        gen = output_ids
        pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
        generated_texts.append(pred)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="LaMP-7 Inference: MLP with PQ Codebook")
    
    # 模型路径
    parser.add_argument("--codebook_path", type=str, required=True,
                       help="训练好的PQ codebook模型路径")
    parser.add_argument("--mlp_path", type=str, required=True,
                       help="训练好的MLP模型路径")
    parser.add_argument("--llm_path", type=str, required=True,
                       help="LLM模型路径")
    parser.add_argument("--encoder_path", type=str, required=True,
                       help="Encoder模型路径（Contriever）")
    
    # 数据路径
    parser.add_argument("--questions_path", type=str, required=True,
                       help="测试问题JSON文件路径")
    parser.add_argument("--outputs_path", type=str, default=None,
                       help="Ground truth输出JSON文件路径（可选，用于评估）")
    parser.add_argument("--output_path", type=str, required=True,
                       help="预测结果保存路径")
    
    # 推理参数
    parser.add_argument("--his_len", type=int, default=8,
                       help="Number of historical profile items to use")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="最大生成token数")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备（cuda:0, cuda:1, cpu等）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LaMP-7 Inference: MLP with PQ Codebook")
    print("=" * 80)
    print(f"Codebook: {args.codebook_path}")
    print(f"MLP: {args.mlp_path}")
    print(f"LLM: {args.llm_path}")
    print(f"Encoder: {args.encoder_path}")
    print(f"Questions: {args.questions_path}")
    print(f"Output: {args.output_path}")
    print(f"History length: {args.his_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # 加载模型
    encoder_model, encoder_tokenizer, pq_codebook_model, mlp_model, llm_model, tokenizer = load_models(
        codebook_path=args.codebook_path,
        mlp_path=args.mlp_path,
        llm_path=args.llm_path,
        encoder_path=args.encoder_path,
        device=args.device
    )
    
    # 加载ground truth（如果有）
    golds = {}
    if args.outputs_path:
        print(f"\nLoading ground truth from {args.outputs_path}...")
        with open(args.outputs_path, 'r', encoding='utf-8') as f:
            outputs_data = json.load(f)
        if "golds" in outputs_data:
            golds = {item['id']: item['output'] for item in outputs_data['golds']}
        else:
            for item in outputs_data:
                item_id = item.get("id", "")
                output = item.get("output", "")
                if item_id:
                    golds[item_id] = output
        print(f"✅ 加载 {len(golds)} 条ground truth")
    
    # 创建数据集
    dataset = LaMP7InferenceDataset(
        json_path=args.questions_path,
        encoder_model=encoder_model,
        encoder_tokenizer=encoder_tokenizer,
        llm_tokenizer=tokenizer,
        his_len=args.his_len,
        device=args.device
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # 因为需要实时编码，所以不用多进程
    )
    
    # 推理
    print(f"\n开始推理...")
    predictions = []
    
    for batch in tqdm(dataloader, desc="🧠 推理"):
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
            device=args.device
        )
        
        # 保存结果
        for i, pred_text in enumerate(generated_texts):
            item_id = batch["id"][i]
            predictions.append({
                'id': item_id,
                'input': batch["input"][i],
                'prediction': pred_text,
                'gold': golds.get(item_id, '') if golds else ''
            })
    
    # 保存结果
    print(f"\n正在保存结果到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 推理完成！共生成 {len(predictions)} 条预测结果")
    print(f"✅ 结果已保存至: {args.output_path}")
    
    # 如果有ground truth，显示一些统计信息
    if golds:
        print(f"\n📊 统计信息:")
        print(f"   总样本数: {len(predictions)}")
        print(f"   有ground truth的样本数: {len(golds)}")
        
        # 显示前几个预测结果示例
        print(f"\n📝 预测结果示例（前3条）:")
        for i, pred in enumerate(predictions[:3]):
            print(f"\n   [{i+1}] ID: {pred['id']}")
            print(f"       输入: {pred['input'][:100]}...")
            print(f"       预测: {pred['prediction'][:100]}...")
            if pred['gold']:
                print(f"      真实: {pred['gold'][:100]}...")


if __name__ == "__main__":
    main()

