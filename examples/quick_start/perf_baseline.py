import argparse
from pathlib import Path
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
import time
import torch
import statistics
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--retriever_path", type=str)
args = parser.parse_args()

config_dict = {
    "data_dir": "dataset/",
    "index_path": "indexes/e5_Flat.index",
    "corpus_path": "indexes/general_knowledge.jsonl",
    "model2path": {"e5": args.retriever_path, "qwen2.5-0.5b-instruct": args.model_path},
    "generator_model": "qwen2.5-0.5b-instruct",
    "retrieval_method": "e5",
    "metrics": ["em", "f1", "acc"],
    "retrieval_topk": 5,
    "save_intermediate_data": True,
    "framework": "hf",
    "test_sample_num": 5,
    "random_sample": False,
    "retrieval_use_fp16": False,
    "generator_batch_size": 1,
    "generation_params": {"do_sample": False, "max_tokens": 32},
}

config = Config(config_dict=config_dict)
print(f"[config] script={Path(__file__).resolve()}")
print(f"[config] retrieval_topk={config['retrieval_topk']}")

all_split = get_dataset(config)
test_data = all_split["test"]
prompt_templete = PromptTemplate(
    config,
    system_prompt="Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt="Question: {question}\nAnswer:",
)


pipeline = SequentialPipeline(config, prompt_template=prompt_templete)


for warmup in range(1, 4):
    print(f"[warmup {warmup}] excluded from results")
    pipeline.run(test_data, do_eval=False)
    torch.mps.synchronize()

batch_times = []
for repeat in range(1, 11):
    print(f"[repeat {repeat}]")
    torch.mps.synchronize()
    total_start = time.perf_counter()
    output_dataset = pipeline.run(test_data, do_eval=False)
    torch.mps.synchronize()
    total_s = time.perf_counter() - total_start
    batch_times.append(total_s)
    print(f"[profile] total_batch_s={total_s:.4f}")

print(f"[summary] median_batch_s={statistics.median(batch_times):.4f}")
qps = len(test_data) * len(batch_times) / sum(batch_times)
print(f"[summary] throughput_qps={qps:.3f}")

output_dataset = pipeline.evaluate(output_dataset, do_eval=True)
print("---generation output---")
print(output_dataset.pred)
