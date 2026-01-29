import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import re


class PQCodebookModel(nn.Module):
    """PQ Codebook模型（用于推理）"""
    def __init__(self, codebook_path, device="cpu"):
        super().__init__()
        checkpoint = torch.load(codebook_path, map_location=device)
        
        if "codebooks" in checkpoint:
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
        """Product Quantization"""
        batch_size, seq_len, emb_dim = embeddings.shape
        flat_embs = embeddings.view(-1, emb_dim)
        subspace_embs = flat_embs.view(-1, self.num_subspaces, self.subspace_dim)
        
        quantized_parts = []
        for i, codebook in enumerate(self.codebooks):
            subspace = subspace_embs[:, i, :]
            codebook = codebook.to(subspace.device)
            distances = torch.cdist(subspace, codebook, p=2)
            indices = torch.argmin(distances, dim=-1)
            quantized = codebook[indices]
            quantized_parts.append(quantized)
        
        quantized = torch.cat(quantized_parts, dim=-1)
        quantized = quantized.view(batch_size, seq_len, emb_dim)
        return quantized
    
    def forward(self, embeddings):
        quantized = self.quantize(embeddings)
        return quantized


class MLPProjection(nn.Module):
    """MLP投影层"""
    def __init__(self, input_dim=768, hidden_dim=None, output_dim=4096):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling for sentence embeddings"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def extract_input_description(input_str):
    """从 review_writing 的 input 中提取 description"""
    desc_match = re.search(r'description "([^"]+)"', input_str)
    description = desc_match.group(1) if desc_match else None
    return description


def get_query_review_writing(inp):
    """从 review_writing 的 input 中提取查询信息"""
    description = extract_input_description(inp)
    if description:
        return description
    return inp.strip()


def ensure_chat_template(tokenizer, model_path):
    """确保tokenizer有chat_template"""
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        return
    
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


def format_chat_template(messages, tokenizer, add_generation_prompt=False):
    """格式化chat template
    
    Args:
        messages: 消息列表
        tokenizer: tokenizer
        add_generation_prompt: 是否添加生成提示（推理时应该为True）
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            print(f"⚠️  apply_chat_template 失败: {e}，使用fallback")
            pass
    
    # Fallback: 手动构建Qwen格式
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # 如果add_generation_prompt=True，添加assistant的起始标记
    if add_generation_prompt:
        formatted_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(formatted_parts)


def load_mlp_checkpoint(checkpoint_path, device):
    """加载训练好的MLP模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    input_dim = checkpoint.get("input_dim", 768)
    hidden_dim = checkpoint.get("hidden_dim", None)
    output_dim = checkpoint.get("output_dim", 4096)
    
    mlp = MLPProjection(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    mlp.load_state_dict(checkpoint["mlp"])
    mlp = mlp.to(device)
    mlp.eval()
    
    print(f"✅ Loaded MLP checkpoint from {checkpoint_path}")
    print(f"   Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    
    return mlp


def encode_text(encoder_model, encoder_tokenizer, text, max_length=512, device="cuda"):
    """使用encoder编码文本（单个文本）"""
    encoder_model.eval()
    
    with torch.no_grad():
        inputs = encoder_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 确保序列长度不超过max_length
        if inputs['input_ids'].size(1) > max_length:
            inputs['input_ids'] = inputs['input_ids'][:, :max_length]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]
            if 'token_type_ids' in inputs:
                inputs['token_type_ids'] = inputs['token_type_ids'][:, :max_length]
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = encoder_model(**inputs)
        token_embeddings = outputs[0]
        
        embeddings = mean_pooling(token_embeddings, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.squeeze(0).cpu()


def encode_texts_batch(encoder_model, encoder_tokenizer, texts, max_length=512, device="cuda"):
    """批量编码文本"""
    encoder_model.eval()
    
    with torch.no_grad():
        inputs = encoder_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 确保序列长度不超过max_length
        if inputs['input_ids'].size(1) > max_length:
            inputs['input_ids'] = inputs['input_ids'][:, :max_length]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]
            if 'token_type_ids' in inputs:
                inputs['token_type_ids'] = inputs['token_type_ids'][:, :max_length]
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = encoder_model(**inputs)
        token_embeddings = outputs[0]
        
        embeddings = mean_pooling(token_embeddings, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu()


class ReviewWritingInferenceDataset(Dataset):
    """Review Writing推理数据集"""
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                self.data.append(entry)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'id': item.get("id", ""),
            'input': item.get("input", ""),
            'output': item.get("output", ""),
            'reviewerId': item.get("reviewerId", ""),
            'profile': item.get("profile", [])
        }


def collate_fn(batch):
    """自定义collate函数，处理不同长度的profile列表"""
    # profile 列表长度可能不同，需要特殊处理
    return {
        'id': [item['id'] for item in batch],
        'input': [item['input'] for item in batch],
        'output': [item['output'] for item in batch],
        'reviewerId': [item['reviewerId'] for item in batch],
        'profile': [item['profile'] for item in batch]  # 保持为列表，不进行collate
    }


def main():
    parser = argparse.ArgumentParser(description="Review Writing Inference: MLP with PQ Codebook")
    
    # 数据参数
    parser.add_argument("--data_path", type=str,
                       default="/mnt/workspace/wangliang/review_writing/val-00000-of-00001.json",
                       help="Review Writing JSONL数据文件路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出JSON文件路径")
    
    # 模型参数
    parser.add_argument("--encoder_path", type=str, required=True,
                       help="Encoder模型路径（Contriever）")
    parser.add_argument("--codebook_path", type=str, required=True,
                       help="训练好的PQ codebook模型路径")
    parser.add_argument("--mlp_checkpoint", type=str, required=True,
                       help="训练好的MLP checkpoint路径")
    parser.add_argument("--llm_path", type=str, required=True,
                       help="LLM模型路径")
    
    # 推理参数
    parser.add_argument("--his_len", type=int, default=8,
                       help="Number of historical profile items to use")
    parser.add_argument("--max_length", type=int, default=600,
                       help="Maximum sequence length for LLM")
    parser.add_argument("--max_encoder_length", type=int, default=512,
                       help="Maximum sequence length for encoder")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Review Writing Inference: MLP with PQ Codebook")
    print("=" * 80)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载encoder
    print(f"\nLoading encoder from {args.encoder_path}...")
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, trust_remote_code=True)
    encoder_model = AutoModel.from_pretrained(args.encoder_path, trust_remote_code=True)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    print("✅ Encoder loaded")
    
    # 加载LLM
    print(f"\nLoading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    ensure_chat_template(tokenizer, args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    llm_model.eval()
    print("✅ LLM loaded")
    
    # 加载PQ codebook
    print(f"\nLoading PQ codebook from {args.codebook_path}...")
    pq_codebook_model = PQCodebookModel(codebook_path=args.codebook_path, device=device)
    pq_codebook_model.eval()
    print("✅ PQ codebook loaded")
    
    # 加载MLP
    print(f"\nLoading MLP from {args.mlp_checkpoint}...")
    mlp_model = load_mlp_checkpoint(args.mlp_checkpoint, device)
    print("✅ MLP loaded")
    
    # 添加占位符token
    placeholder_token = "<USR_EMB>"
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])
        llm_model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Added placeholder token: {placeholder_token}")
    
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    # 加载数据
    print(f"\nLoading data from {args.data_path}...")
    dataset = ReviewWritingInferenceDataset(args.data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    # 推理
    print(f"\nStarting batch inference (batch_size={args.batch_size})...")
    predictions = []
    
    llm_embeddings = llm_model.get_input_embeddings()
    
    for batch in tqdm(dataloader, desc="Inferencing"):
        batch_size = len(batch['id'])
        input_texts = batch['input']
        output_texts = batch['output']
        reviewer_ids = batch['reviewerId']
        profiles = batch['profile']
        item_ids = batch['id']
        
        # 批量编码profile
        batch_user_embs_list = []
        for i in range(batch_size):
            profile = profiles[i]
            user_embs_list = []
            
            # 收集所有需要编码的review texts
            review_texts = []
            for profile_item in profile[:args.his_len]:
                review_text = profile_item.get("reviewText", "")
                if review_text.strip():
                    review_texts.append(review_text)
                else:
                    review_texts.append("")  # 空文本，后续用零向量
            
            # 批量编码非空的review texts
            non_empty_texts = [t for t in review_texts if t.strip()]
            if non_empty_texts:
                non_empty_embs = encode_texts_batch(
                    encoder_model, encoder_tokenizer, non_empty_texts,
                    max_length=args.max_encoder_length, device=device
                )
                non_empty_idx = 0
                for text in review_texts:
                    if text.strip():
                        user_embs_list.append(non_empty_embs[non_empty_idx])
                        non_empty_idx += 1
                    else:
                        user_embs_list.append(torch.zeros(768))
            else:
                # 全部为空
                user_embs_list = [torch.zeros(768) for _ in review_texts]
            
            # 填充到his_len
            while len(user_embs_list) < args.his_len:
                user_embs_list.append(torch.zeros(768))
            
            batch_user_embs_list.append(torch.stack(user_embs_list[:args.his_len]))
        
        # 堆叠成batch: (batch_size, his_len, 768)
        user_embs = torch.stack(batch_user_embs_list).to(device)
        
        # PQ量化和MLP投影
        with torch.no_grad():
            quantized_embs = pq_codebook_model(user_embs)  # (batch_size, his_len, 768)
            llm_embs = mlp_model(quantized_embs)  # (batch_size, his_len, llm_dim)
        
        # 构建prompts
        prompts = []
        for i in range(batch_size):
            input_text = input_texts[i]
            if args.his_len > 0:
                user_prompt_text = (
                    f"User style embedding: {placeholder_token}\n"
                    "Based on the user's style embedding provided above, please generate the review text written by a reviewer who has a given an overall rating for a product.\n"
                    f"{input_text} Write at least 250 words."
                )
            else:
                user_prompt_text = (
                    "Please generate the review text written by a reviewer who has a given an overall rating for a product.\n"
                    f"{input_text}"
                )
            messages = [{"role": "user", "content": user_prompt_text}]
            # 推理时使用 add_generation_prompt=True 来添加assistant的起始标记
            formatted = format_chat_template(messages, tokenizer, add_generation_prompt=True)
            prompts.append(formatted)
        
        # 批量tokenize（left padding）
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        ).to(device)
        
        input_ids = inputs["input_ids"]  # (batch_size, seq_len)
        attention_mask = inputs["attention_mask"]  # (batch_size, seq_len)
        
    
        input_embs = llm_embeddings(input_ids)  # (batch_size, seq_len, llm_dim)
        
        # 替换占位符位置的embeddings
        for i in range(batch_size):
            placeholder_positions = (input_ids[i] == placeholder_id).nonzero(as_tuple=True)[0].tolist()
            if len(placeholder_positions) > 0 and args.his_len > 0:
                insert_start_pos = placeholder_positions[0]
                for j in range(min(args.his_len, input_embs.size(1) - insert_start_pos)):
                    if insert_start_pos + j < input_embs.size(1):
                        input_embs[i, insert_start_pos + j] = llm_embs[i, j]
        
        # 批量生成
        with torch.no_grad():
            outputs = llm_model.generate(
                inputs_embeds=input_embs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 批量解码
        input_lens = attention_mask.sum(dim=1).cpu().tolist()
        for i in range(batch_size):
            input_len = input_lens[i]
            generated_ids = outputs[i]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            predictions.append({
                'id': item_ids[i],
                'input': input_texts[i],
                'prediction': pred,
                'gold': output_texts[i],
                'reviewerId': reviewer_ids[i]
            })
    
    # 保存结果
    print(f"\nSaving results to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Inference completed! Results saved to {args.output_path}")
    print(f"   Total samples: {len(predictions)}")


if __name__ == "__main__":
    main()

