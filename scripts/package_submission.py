"""Package allow-listed project code/evidence and separate large resources.

Weights, virtual environments, credentials, caches and Git internals are never
included. Both archives preserve repository-relative paths. Extract resources
over the code artifact to reproduce with the exact corpus/index and traces.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_archive(destination, paths):
    entries = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            path = path.resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f"Invalid artifact member: {path}")
            relative = path.relative_to(ROOT).as_posix()
            if any(part in (".git", ".venv", "__pycache__", ".aws", ".codex", ".agents") for part in path.relative_to(ROOT).parts):
                raise ValueError("Excluded private/cache directory reached the allow list")
            archive.write(path, relative)
            entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    return {"file": destination.relative_to(ROOT).as_posix(), "bytes": destination.stat().st_size,
            "sha256": digest(destination), "entries": entries}


def main():
    verification = json.loads((ROOT / "docs/experiments/final_topk/final_verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "passed":
        raise ValueError("Final validation must pass first")
    if not (ROOT / "docs/performance_profilling_report.pdf").is_file():
        raise ValueError("Generate the PDF before packaging")
    code = []
    for directory in ("flashrag", "scripts", "tests"):
        code.extend(p for p in (ROOT / directory).rglob("*")
                    if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json", ".ps1") and "__pycache__" not in p.parts)
    for name in ("README.md", "LICENSE", "requirements-baseline.txt", "setup.py", "pyproject.toml", ".gitignore"):
        if (ROOT / name).is_file():
            code.append(ROOT / name)
    code.extend(p for p in (ROOT / "docs").glob("*") if p.is_file() and p.suffix in (".json", ".txt", ".pdf"))
    code.append(ROOT / "docs/performance_profilling_report.md")
    code.append(ROOT / "scripts/capture_windows_cpu.ps1")
    code.extend(p for p in (ROOT / "docs/experiments").rglob("*")
                if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".txt", ".log", ".prof"))
    for name in ("datasets/flashrag_official/nq/test.jsonl", "datasets/nq_500_heldout/nq/test.jsonl",
                 "datasets/nq_final_study/nq/dev.jsonl", "datasets/nq_final_study/nq/test.jsonl",
                 "examples/quick_start/dataset/nq/test.jsonl", "examples/quick_start/perf_baseline.py"):
        code.append(ROOT / name)
    resources = [ROOT / "examples/quick_start/indexes/general_knowledge.jsonl", ROOT / "examples/quick_start/indexes/e5_Flat.index"]
    resources.extend((ROOT / "docs/experiments").rglob("*.gz"))
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    inventory = {"code": build_archive(destination / "cs598ape_final_artifact.zip", code),
                 "resources": build_archive(destination / "cs598ape_resources.zip", resources),
                 "excluded": ["model weights", "virtual environment", "model/dataset caches", "Git internals", "credentials"],
                 "instructions": "Extract both archives into the same directory. Install dependencies and acquire the pinned Qwen/E5 models using README.md. Read the final report before interpreting performance claims."}
    (destination / "submission_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (destination / "submission_email.txt").write_text(
        "Subject: [CS598APE Final Report] Lightweight Adaptive Top-K Retrieval in FlashRAG\n\n"
        "Dear course staff,\n\n"
        "Please find our final project report and code/test artifact attached. The artifact includes the README, test suites, frozen development/evaluation protocol, and experiment evidence.\n\n"
        "The report separates Adaptive Top-K results from supporting system optimizations and states the limitations and incomplete native profiling work. No project-authored C/C++ code requires Valgrind.\n\n"
        "Repository or large-resource download link: [add shared repository/resource link]\n\n"
        "Best regards,\nYidong Zhang and Yuang Cai\n", encoding="utf-8")
    for name in ("code", "resources"):
        info = inventory[name]
        print(f"[PACKAGE] {info['file']}: {info['bytes']/2**20:.2f} MiB; {len(info['entries'])} verified files")


if __name__ == "__main__":
    main()
