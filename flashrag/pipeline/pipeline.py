from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
import time 
import torch


def _synchronize_device(device):
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None, rewriter=None,
                 retrieval_policy=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator

        rewriter: Optional module whose batch_run(dataset) returns one rewritten
            query per question for retrieval. Answer prompts use the original questions.
        retrieval_policy: Optional score-based policy with max_k and select(scores).
            Retrieve max_k candidates once, then select a prefix before prompting.
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        self.rewriter = rewriter
        self.retrieval_policy = retrieval_policy
        self.last_profile = {}

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        if self.rewriter is not None:
            input_query = self.rewriter.batch_run(dataset)
            dataset.update_output("rewrite_result", input_query)
        _synchronize_device(self.device)
        retrieval_start = time.perf_counter()
        if self.retrieval_policy is None:
            retrieval_results = self.retriever.batch_search(input_query)
        else:
            retrieval_results, retrieval_scores = self.retriever.batch_search(
                input_query, num=self.retrieval_policy.max_k, return_score=True
            )
        _synchronize_device(self.device)
        retrieval_s = time.perf_counter() - retrieval_start

        selection_s = 0.0
        if self.retrieval_policy is not None:
            selection_start = time.perf_counter()
            if len(retrieval_results) != len(input_query) or len(retrieval_scores) != len(input_query):
                raise ValueError("retriever must return one document and score list per query")
            decisions = []
            selected_results = []
            for documents, scores in zip(retrieval_results, retrieval_scores):
                if len(documents) != len(scores):
                    raise ValueError("retrieval documents and scores must have matching lengths")
                decision = self.retrieval_policy.select(scores)
                decisions.append(decision)
                selected_results.append(documents[: decision["selected_k"]])
            dataset.update_output("retrieval_scores", retrieval_scores)
            dataset.update_output("selected_k", [decision["selected_k"] for decision in decisions])
            dataset.update_output("adaptive_decision", decisions)
            retrieval_results = selected_results
            selection_s = time.perf_counter() - selection_start

        print(f"[profile] retrieval_batch_s={retrieval_s:.4f}, queries={len(input_query)}")
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                prompt_start = time.perf_counter()
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                prompt_s = time.perf_counter() - prompt_start
                print(f"[profile] prompt_batch_s={prompt_s:.4f}")
                
        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        print(f"[profile] generator_device={self.generator.model.device}")
        generator_device = getattr(self.generator.model, "device", self.device)
        _synchronize_device(generator_device)
        generation_start = time.perf_counter()
        pred_answer_list = self.generator.generate(input_prompts)
        _synchronize_device(generator_device)
        generation_s = time.perf_counter() - generation_start
        self.last_profile = {
            "retrieval_s": retrieval_s,
            "selection_s": selection_s,
            "prompt_s": locals().get("prompt_s"),
            "generation_s": generation_s,
        }
        print(f"[profile] generation_batch_s={generation_s:.4f}")
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)

        self.judger = get_judger(config)
        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
        retriever = None,
        generator = None
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config, prompt_template=multi_hop_prompt_template, retriever=retriever, generator=generator, max_iter=5
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
