# CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs

## Introduction

This repository contains the implementation of CURP (Codebook-based Continuous User Representation for Personalized Generation with LLMs), a novel framework for learning interpretable and continuous user representations to enhance personalized text generation with large language models (LLMs). The training process can be mainly devided into 2 stages. In the first stage we construct a universal codebook via product quantization with balanced K-Means initialization to build user prototype. In the second stage we project the prototype representation into LLM's space and guide for personalization.


## Installation

```bash
git clone https://github.com/RaidonWong/CURP.git curp
cd curp
pip install -r requirements.txt
```



##  Data Preprocessing



### 1. **AlignX Dataset**
- **Source**: [AlignX](https://huggingface.co/datasets/JinaLeejnl/AlignX)
- **Fields Used**: `"prompt"`, `"chosen"`, `"rejected"`, `"Demographic Information"`, `"User-Generated Content"`
- **Note**: We use a randomly filtered deduplicated subset. Can be replaced with any user history dataset with broad knowledge distribution.

### 2. **Val Datasets**
- **Sources**: 
  - [LaMP Benchmark](https://lamp-benchmark.github.io/)
  - [LongLaMP Benchmark](https://longlamp-benchmark.github.io/)
- **Tasks**: News Headline, Tweet Paraphrase, Review Writing
- **Preprocessing**:
  - **News Headline**: Filter short texts; use LLaMA-3-8B to judge headline-paragraph association.
  - **Tweet Paraphrase**: Remove noisy entries (e.g., `@<ref>`); use LLM for filter.
  - **Review Writing**: Keep consistent-rating reviews, length filter


## Usage

- First of all, you need to download the Qwen-2.5-7B-Instruct model and Contriever model
  - [LLaMA-3-8B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  - [Contriever](https://huggingface.co/facebook/contriever)
Then you need to add two speicial token, `<PAD>` and `<USR_EMB>` and resize the LLM tokenizer embedding input. The `<PAD>` is for padding and the `<USR_EMB>` is for indicating the place to insert our user model.

```python
special_tokens = {
    "additional_special_tokens": ["<PAD>", "<USR_EMB>"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```


- Secondly, you need to pretrain a Product Quantized codebook by 
```bash
bash step1.sh
```

- Thirdly, you need to align the representation with LLMs by
  
```bash
bash step2.sh
```


- After training, you can inference by
  
```bash
python inference.py
```


The argument is by default. If you want to change, you can change as you need.
































