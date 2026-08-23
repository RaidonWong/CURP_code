"""
Merge LaMP input JSON and answer JSON into a single pipeline JSONL.

All original keys (id, input, profile, title, text, abstract, etc.) are
preserved exactly. Two fields are added:
  - "output": the ground-truth answer (from the answer JSON)
  - "order":  a globally unique integer added to each profile item,
              giving its row index in the embeddings.pt tensor

Output format per line (lamp4 example):
  {
    "id": "300",
    "input": "Generate a headline for ...",
    "output": "The Pre-Crisis HELOC Chickens Are Coming Home to Roost",
    "profile": [
      {"text": "Homeowners sell ...", "title": "Selling a House ...", "id": "3000", "order": 0},
      {"text": "This article summarizes ...", "title": "Introducing ...",  "id": "3001", "order": 1},
      ...
    ]
  }

encode.py, stage1.py, and stage2.py all read this format directly.
"""

import json
import os
import argparse


def main(args):
    input_stem = os.path.splitext(os.path.basename(args.input_json))[0]
    output_path = os.path.join(args.output_dir, f"{input_stem}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input_json}...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} records")

    print(f"Loading {args.answer_json}...")
    with open(args.answer_json, "r", encoding="utf-8") as f:
        answers = json.load(f)
    answer_lookup = {g["id"]: g["output"] for g in answers["golds"]}
    print(f"  {len(answer_lookup)} answers")

    global_order = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in data:
            output = answer_lookup.get(item["id"], "")
            if not output:
                skipped += 1
                continue

            # Deep-copy the item so we don't mutate the original list
            record = {k: v for k, v in item.items()}
            record["output"] = output

            # Assign a global order index to each profile entry (in-place on the copy)
            for profile_item in record.get("profile", []):
                profile_item["order"] = global_order
                global_order += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Total profile embeddings: {global_order}")
    print(f"  Examples written: {len(data) - skipped}  |  Skipped: {skipped}")
    print(f"  Output: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. python encode.py \\")
    print(f"       --input_jsonl {output_path} \\")
    print(f"       --model_dir <contriever_model_dir> \\")
    print(f"       --output_pt {args.output_dir}/{input_stem}_embeddings.pt")
    print(f"  2. python stage1.py \\")
    print(f"       --embeddings_path {args.output_dir}/{input_stem}_embeddings.pt \\")
    print(f"       --output_dir {args.output_dir}/stage1_output \\")
    print(f"       --use_balanced_kmeans_init")
    print(f"  3. python stage2.py \\")
    print(f"       --data_path {output_path} \\")
    print(f"       --embeddings_path {args.output_dir}/{input_stem}_embeddings.pt \\")
    print(f"       --codebook_path {args.output_dir}/stage1_output/stage1_final/codebook_model.pt \\")
    print(f"       --llm_path <llm_model_dir> \\")
    print(f"       --output_dir {args.output_dir}/stage2_output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LaMP JSON + answer JSON into a pipeline JSONL (preserves original keys)"
    )
    parser.add_argument("--input_json",  required=True, help="e.g. lamp4_train.json")
    parser.add_argument("--answer_json", required=True, help="e.g. lamp4_train_answer.json")
    parser.add_argument("--output_dir",  required=True, help="Directory for output JSONL")
    args = parser.parse_args()
    main(args)
