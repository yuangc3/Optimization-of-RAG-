"""Download official NQ test data and select new questions reproducibly."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "RUC-NLPIR/FlashRAG_datasets"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question_key(record):
    return " ".join(record["question"].casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-num", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--revision", help="Pin the dataset commit when reproducing a previous manifest.")
    args = parser.parse_args()
    if args.sample_num < 1:
        parser.error("sample-num must be positive")
    revision = args.revision or HfApi().dataset_info(REPO_ID).sha
    source_path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", revision=revision,
        filename="nq/test.jsonl", local_dir=ROOT / "datasets/flashrag_official",
    ))
    source = read_jsonl(source_path)
    previous = read_jsonl(ROOT / "examples/quick_start/dataset/nq/test.jsonl")
    excluded = {question_key(item) for item in previous}
    eligible = []
    seen = set()
    overlap_count = 0
    duplicate_count = 0
    for item in source:
        if not item.get("id") or not item.get("question") or not isinstance(item.get("golden_answers"), list):
            raise ValueError("Unexpected official NQ record format")
        key = question_key(item)
        if key in excluded:
            overlap_count += 1
        elif key in seen:
            duplicate_count += 1
        else:
            eligible.append(item)
            seen.add(key)
    if len(eligible) < args.sample_num:
        raise ValueError("Not enough new, unique questions for the requested sample")
    sample = random.Random(args.seed).sample(eligible, args.sample_num)
    destination = ROOT / f"datasets/nq_{args.sample_num}_heldout/nq/test.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in sample), encoding="utf-8")
    manifest = {
        "source_repo": REPO_ID, "source_revision": revision, "source_file": "nq/test.jsonl",
        "source_count": len(source), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "previous_question_count": len(previous), "excluded_overlap_count": overlap_count,
        "excluded_duplicates": duplicate_count, "eligible_count": len(eligible),
        "sample_count": len(sample), "seed": args.seed,
        "sample_file": str(destination.relative_to(ROOT)),
        "sample_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "selected_ids": [item["id"] for item in sample],
        "note": "Excludes the previous local questions. Policies must be fixed before evaluating this sample.",
    }
    manifest_path = ROOT / f"docs/nq_{args.sample_num}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_ids"}, indent=2))


if __name__ == "__main__":
    main()
