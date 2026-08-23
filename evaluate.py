import os
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import json
import argparse
import math
import re
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _f1(overlap, predicted, reference):
    if predicted == 0 or reference == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2 * precision * recall / (precision + recall)


def _ngram_f1(pred_tokens, ref_tokens, n):
    pred = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
    ref = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum((pred & ref).values())
    return _f1(overlap, sum(pred.values()), sum(ref.values()))


def _lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def compute_rouge(predictions, references):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        rouge_scorer = None
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        values = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
        return {
            'rouge1': float(np.mean([value['rouge1'].fmeasure for value in values])),
            'rouge2': float(np.mean([value['rouge2'].fmeasure for value in values])),
            'rougeL': float(np.mean([value['rougeL'].fmeasure for value in values])),
        }
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        pred_tokens, ref_tokens = _tokens(pred), _tokens(ref)
        r1.append(_ngram_f1(pred_tokens, ref_tokens, 1))
        r2.append(_ngram_f1(pred_tokens, ref_tokens, 2))
        lcs = _lcs_length(pred_tokens, ref_tokens)
        rl.append(_f1(lcs, len(pred_tokens), len(ref_tokens)))
    return {
        'rouge1': float(np.mean(r1)),
        'rouge2': float(np.mean(r2)),
        'rougeL': float(np.mean(rl)),
    }


def compute_bleu(predictions, references):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        sentence_bleu = None
    if sentence_bleu is not None:
        smoother = SmoothingFunction().method1
        values = []
        for pred, ref in zip(predictions, references):
            pred_tokens, ref_tokens = _tokens(pred), _tokens(ref)
            values.append(
                sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother)
                if pred_tokens else 0.0
            )
        return {'bleu': float(np.mean(values))}
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens, ref_tokens = _tokens(pred), _tokens(ref)
        if not pred_tokens:
            scores.append(0.0)
            continue
        precisions = []
        for n in range(1, 5):
            pred_ngrams = Counter(
                tuple(pred_tokens[i:i + n])
                for i in range(max(0, len(pred_tokens) - n + 1))
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i + n])
                for i in range(max(0, len(ref_tokens) - n + 1))
            )
            overlap = sum((pred_ngrams & ref_ngrams).values())
            # Add-one smoothing keeps short generated strings well-defined.
            precisions.append((overlap + 1.0) / (sum(pred_ngrams.values()) + 1.0))
        brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(
            1.0 - len(ref_tokens) / max(1, len(pred_tokens))
        )
        scores.append(brevity * math.exp(sum(math.log(p) for p in precisions) / 4.0))
    return {'bleu': float(np.mean(scores))}


def compute_bert_sim(predictions, references, device='cuda', model_path=None,
                     required=False):
    # Truncate using the actual RoBERTa tokenizer so we never exceed the 512-token limit
    # regardless of language (CJK tokenizes at ~1.5-2 tokens/char).
    try:
        from bert_score import score as bert_score_fn
        from transformers import AutoTokenizer
    except ImportError as exc:
        if required:
            raise RuntimeError("BERTScore requested but bert-score is not installed") from exc
        print("  Warning: bert-score is unavailable; skipping bert_cos_sim")
        return {}
    if not model_path:
        if required:
            raise ValueError(
                "BERTScore requires --bert_model_path or BERT_MODEL_PATH"
            )
        print("  Warning: no BERTScore model configured; skipping bert_cos_sim")
        return {}
    if not os.path.isdir(model_path):
        if required:
            raise FileNotFoundError(f"BERTScore model path does not exist: {model_path}")
        print(f"  Warning: BERTScore model path not found ({model_path}); skipping")
        return {}
    _tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    def _trunc(texts, max_tokens=500):
        result = []
        for t in texts:
            ids = _tok.encode(t, add_special_tokens=False)[:max_tokens]
            result.append(_tok.decode(ids, skip_special_tokens=True))
        return result
    P, R, F = bert_score_fn(
        _trunc(predictions), _trunc(references),
        model_type=model_path,
        num_layers=17,
        device=device,
        verbose=False
    )
    return {'bert_cos_sim': F.mean().item()}


LAMP2_LABELS = [
    'sci-fi', 'based on a book', 'comedy', 'action', 'twist ending',
    'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy',
    'romance', 'thought-provoking', 'social commentary', 'violence', 'true story',
]


def _normalize_label(text: str) -> str:
    return text.strip().lower()


def _parse_classification(text, task):
    normalized = _normalize_label(text)
    if task == 'lamp1':
        match = re.fullmatch(r"\[?\s*([12])\s*\]?\.?", normalized)
        return match.group(1) if match else normalized
    if task == 'lamp2':
        exact = normalized.strip(" .\"'")
        if exact in LAMP2_LABELS:
            return exact
        matches = []
        for label in LAMP2_LABELS:
            for match in re.finditer(rf"(?<!\w){re.escape(label)}(?!\w)", normalized):
                matches.append((label, match.start(), match.end()))
        maximal = []
        for candidate in matches:
            _, start, end = candidate
            if any(
                other_start <= start
                and end <= other_end
                and (other_end - other_start) > (end - start)
                for _, other_start, other_end in matches
            ):
                continue
            maximal.append(candidate[0])
        unique = list(dict.fromkeys(maximal))
        return unique[0] if len(unique) == 1 else normalized
    return normalized


def compute_classification(predictions, references, task):
    preds = [_parse_classification(p, task) for p in predictions]
    refs  = [_parse_classification(r, task) for r in references]
    acc   = accuracy_score(refs, preds)
    f1    = f1_score(refs, preds, average='macro', zero_division=0)
    return {'accuracy': acc, 'macro_f1': f1}


def _parse_rating(text: str) -> float | None:
    try:
        v = float(text.strip())
        return v if 1.0 <= v <= 5.0 else None
    except ValueError:
        matches = re.findall(r'(?<!\d)([1-5](?:\.0+)?)(?!\d)', text)
        return float(matches[0]) if len(matches) == 1 else None


def compute_mae_rmse(predictions, references):
    parsed = [(_parse_rating(p), float(r)) for p, r in zip(predictions, references)]
    invalid = sum(1 for p, _ in parsed if p is None)
    if invalid:
        print(f'  Warning: {invalid} predictions could not be parsed as ratings (defaulted to 3.0)')
    preds = np.array([p if p is not None else 3.0 for p, _ in parsed])
    refs  = np.array([r for _, r in parsed])
    mae  = float(mean_absolute_error(refs, preds))
    rmse = float(np.sqrt(np.mean((preds - refs) ** 2)))
    return {'mae': mae, 'rmse': rmse}


def main(args):
    data = []
    with open(args.predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        raise ValueError(f"Prediction file is empty: {args.predictions_path}")
    missing = [index for index, item in enumerate(data) if 'prediction' not in item or 'reference' not in item]
    if missing:
        raise ValueError(f"Missing prediction/reference fields in rows: {missing[:10]}")

    predictions = [d['prediction'] for d in data]
    references  = [d['reference']  for d in data]

    print(f'Evaluating {len(data)} samples...')

    if args.task == 'lamp3':
        results = compute_mae_rmse(predictions, references)
    elif args.task in ('lamp1', 'lamp2'):
        results = compute_classification(predictions, references, args.task)
    else:
        rouge  = compute_rouge(predictions, references)
        bleu   = compute_bleu(predictions, references)
        bert   = compute_bert_sim(
            predictions,
            references,
            device=args.device,
            model_path=args.bert_model_path,
            required=args.require_bert_score,
        )
        results = {**rouge, **bleu, **bert}

    print('\n===== Evaluation Results =====')
    for k, v in results.items():
        print(f'  {k:<20} {v:.4f}')

    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved metrics to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('--predictions_path', required=True,
                        help='JSONL file with prediction and reference fields')
    parser.add_argument('--output_path', default=None,
                        help='Optional path to save metrics JSON')
    parser.add_argument('--device', default='cuda',
                        help='Device for BERTScore (cuda or cpu)')
    parser.add_argument('--task', default=None,
                        help='Task name (e.g. lamp2) to select appropriate metrics')
    parser.add_argument(
        '--bert_model_path',
        default=os.environ.get('BERT_MODEL_PATH'),
        help='Optional local BERTScore model; defaults to BERT_MODEL_PATH',
    )
    parser.add_argument('--require_bert_score', action='store_true',
                        help='Fail instead of skipping BERTScore when its dependency/model is unavailable')
    args = parser.parse_args()
    main(args)
