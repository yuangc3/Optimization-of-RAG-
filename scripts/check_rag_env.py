"""Validate the local benchmark environment without downloading model weights."""
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import faiss
import numpy as np
from flashrag.config import Config
from flashrag.pipeline import SequentialPipeline
from flashrag.generator import HFCausalLMGenerator


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the project .venv interpreter to run this check.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter.")

    # Execute and synchronize a real CUDA operation, rather than only enumerating GPUs.
    matrix = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    result = matrix @ torch.eye(4, device="cuda")
    torch.cuda.synchronize()
    torch.testing.assert_close(result, matrix)
    device = torch.cuda.get_device_properties(0)

    base = ROOT / "examples/quick_start"
    index = faiss.read_index(str(base / "indexes/e5_Flat.index"))
    with (base / "indexes/general_knowledge.jsonl").open(encoding="utf-8") as handle:
        corpus_count = sum(1 for line in handle if line.strip())
    with (base / "dataset/nq/test.jsonl").open(encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle if line.strip()]
    if index.ntotal != corpus_count:
        raise RuntimeError("Index and corpus counts differ.")
    if len(questions) < 5:
        raise RuntimeError("At least five smoke-test questions are required.")
    # Verify that FAISS can execute a CPU search in addition to reading the file.
    _, ids = index.search(np.zeros((1, index.d), dtype=np.float32), 5)
    if ids.shape != (1, 5) or (ids < 0).any():
        raise RuntimeError("FAISS smoke search failed.")

    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "isolated_environment": sys.prefix != sys.base_prefix,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": device.name,
        "gpu_memory_bytes": device.total_memory,
        "cuda_matrix_check": "passed",
        "transformers": importlib.metadata.version("transformers"),
        "faiss": importlib.metadata.version("faiss-cpu"),
        "flashrag_imports": "passed",
        "corpus_documents": corpus_count,
        "index_vectors": index.ntotal,
        "index_dimensions": index.d,
        "faiss_search": "passed",
        "sample_questions": len(questions),
        "note": "Environment check only; no Qwen generation or RAG quality measurement.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
