from src.ingest_vllm import Chunker
import torch
from torch import tensor, Tensor
import bm25s
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from src.helpers import streamline_query, clean_text_chunks
import json
from src.pydantic_models import (
    MinimalSource,
    MinimalSearchResults,
    UnansweredQuestion,
    StudentSearchResults,
    RagDataset,
    AnsweredQuestion,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from pathlib import Path
import pickle
from tqdm import tqdm


class RagAgainstTheMachine:
    """main orchestrator, all commands are defined here"""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Constructor"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.llm: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(
                    torch.float16 if self.device == "cuda" else torch.float32
                ),
                device_map="auto" if self.device == "cuda" else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except RuntimeError:
            raise RuntimeError(
                "Could not fetch model, are you connected to the internet?"
            )

    def index_docs(self, maximum_chunk_size: int = 2000) -> None:
        """has corpus chunked, indexes it and saves everything to files"""
        if maximum_chunk_size < 1000:
            raise ValueError(
                "minimum chunk size is 1000 characters"
                + ", recommended size is 1500-2000"
            )
        elif maximum_chunk_size < 1500:
            print(
                "Warning: recommended chunk size" + " is 1500-2000 characters"
            )
        elif maximum_chunk_size > 2000:
            raise ValueError("Maximum chunk size is 2000 characters")
        doc_corpus = Chunker.chunk_vllm_docs(
            maximum_chunk_size=maximum_chunk_size
        )
        code_corpus = Chunker.chunk_vllm_code(
            maximum_chunk_size=maximum_chunk_size
        )
        docs: list[str] = []
        metadata: list[MinimalSource] = []
        docs.extend(d.get("content", "") for d in doc_corpus)
        docs.extend(d.get("content", "") for d in code_corpus)
        metadata.extend(
            d.get(
                "src",
                MinimalSource(
                    file_path="",
                    first_character_index=-1,
                    last_character_index=-1,
                ),
            )
            for d in doc_corpus
        )
        metadata.extend(
            d.get(
                "src",
                MinimalSource(
                    file_path="",
                    first_character_index=-1,
                    last_character_index=-1,
                ),
            )
            for d in code_corpus
        )
        if (
            MinimalSource(
                file_path="", first_character_index=-1, last_character_index=-1
            )
            in metadata
        ):
            raise Exception("Data encapsulation violation!")
        if docs == []:
            raise Exception(
                "Corpus is empty, please make sure to pass "
                + "the correct corpus folder"
            )

        Path("./data/chunks").mkdir(parents=True, exist_ok=True)
        with open("./data/chunks/chunks.json", "w", encoding="utf-8") as f:
            json.dump(
                [m.model_dump() for m in metadata],
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("\nTokenizing chunked documents...\n")
        tokens = [clean_text_chunks(d) for d in docs]
        corpus_tokens = bm25s.tokenize(
            tokens,
            stopwords="en",
            lower=False,
        )
        print("\nIndexing chunked documents...\n")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        Path("./data/processed").mkdir(parents=True, exist_ok=True)
        with open("./data/processed/bm25_index.pkl", "wb") as f:
            pickle.dump(retriever, f)
        print("\nIndexed Corpus and saved retriever!\n")

    def load_index(self) -> tuple[bm25s.BM25, list[MinimalSource]]:
        """loads all necessary information and index"""
        if (
            not Path("./data/processed/bm25_index.pkl").exists()
            or not Path("./data/chunks/chunks.json").exists()
        ):
            raise Exception(
                "Index doesn't exist, " + "please index the corpus first!"
            )
        try:
            with open("./data/processed/bm25_index.pkl", "rb") as f:
                retriever = pickle.load(f)
            with open("./data/chunks/chunks.json", "r") as f:
                data = json.load(f)
            metadata = [MinimalSource(**item) for item in data]
            return retriever, metadata
        except Exception as e:
            print("Error retrieving BM25 index:", e)
            exit(1)

    def search_index(
        self,
        query: str,
        metadata: list[MinimalSource],
        retriever: bm25s.BM25,
        k: int = 5,
        id: str = "",
    ) -> MinimalSearchResults:
        """searches the index lexically"""
        if id:
            question = UnansweredQuestion(question_id=id, question=query)
        else:
            question = UnansweredQuestion(question=query)
        if query.strip() == "":
            return MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[],
            )
        query = streamline_query(query)
        results, scores = retriever.retrieve(
            bm25s.tokenize(query, stopwords="en"), k=k
        )
        retrieved = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            retrieved.append((doc, score))
        context_docs = [metadata[doc] for doc, _ in retrieved]
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=context_docs,
        )

    def search_set(
        self,
        set_file: str,
        metadata: list[MinimalSource],
        retriever: bm25s.BM25,
        k: int = 5,
        save: str | None = None,
    ) -> None:
        """calls search for queries from datasets"""
        file_name = set_file.split("/")[-1] if "/" in set_file else set_file
        if not save:
            save = "./data/output/search_results"
            save_l = f"{save}/{file_name}"
        elif save[-1] == "/":
            save_l = f"{save}{file_name}"
        else:
            save_l = f"{save}/{file_name}"
        with open(set_file) as f:
            d = json.load(f)
        results = []
        questions = RagDataset(rag_questions=d.get("rag_questions", []))
        for question in tqdm(
            questions.rag_questions,
            desc=f"Processing dataset {file_name} in search mode...",
        ):
            results.append(
                self.search_index(
                    question.question,
                    metadata,
                    retriever,
                    k,
                    question.question_id,
                )
            )
        file = StudentSearchResults(search_results=results, k=k).model_dump(
            mode="json"
        )
        Path(save).mkdir(parents=True, exist_ok=True)
        with open(save_l, "w") as f:
            json.dump(file, f)
        print(f"Saved search results to {save_l}")

    def answer_question(
        self,
        query: str,
        metadata: list[MinimalSource],
        retriever: bm25s.BM25,
        k: int = 5,
        id: str = "",
    ) -> MinimalAnswer:
        """retrieves context and answers questions"""
        if id:
            question = UnansweredQuestion(question_id=id, question=query)
        else:
            question = UnansweredQuestion(question=query)
        if query.strip() == "":
            return MinimalAnswer(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[],
                answer="Please provide a valid query.",
            )
        sources = self.search_index(
            query, metadata, retriever, k, id
        ).retrieved_sources
        source_text: list[str] = []
        for source in sources:
            sf = source.first_character_index
            sl = source.last_character_index
            with open(source.file_path) as f:
                source_text.append(f.read()[sf:sl])
        context = "\n-------\n".join(
            [f"[Doc {i+1}]: {doc}" for i, doc in enumerate(source_text)]
        )
        prompt = f"""You are a reliable and helpful assistant.
Provide concise, single sentence answers based on the given context.
Context:
{context}

Question: {query}

Answer: """
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False)
        with torch.no_grad():
            outputs: Tensor = self.llm.generate(  # type: ignore
                tensor([inputs], device=self.device, dtype=torch.long),
                max_length=int(len(inputs) + 256),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw_response = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        if isinstance(raw_response, list):
            raw_response = raw_response[0]
        response = (
            raw_response.split("Answer:")[1].strip().split("\n")[0].strip()
        )
        return MinimalAnswer(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
            answer=response,
        )

    def answer_set(
        self,
        set_file: str,
        metadata: list[MinimalSource],
        retriever: bm25s.BM25,
        k: int = 5,
        save: str | None = None,
    ) -> None:
        """calls answer for queries from a dataset"""
        file_name = set_file.split("/")[-1] if "/" in set_file else set_file
        if not save:
            save = "./data/output/search_results_and_answer"
            save_l = f"{save}/{file_name}"
        elif save[-1] == "/":
            save_l = f"{save}{file_name}"
        else:
            save_l = f"{save}/{file_name}"
        with open(set_file) as f:
            d = json.load(f)
        results = []
        questions = RagDataset(rag_questions=d.get("rag_questions", []))
        for question in tqdm(
            questions.rag_questions,
            desc=f"Processing dataset {file_name} in answer mode...",
        ):
            results.append(
                self.answer_question(
                    question.question,
                    metadata,
                    retriever,
                    k,
                    question.question_id,
                )
            )
        file = StudentSearchResultsAndAnswer(
            search_results=results, k=k
        ).model_dump(mode="json")
        Path(save).mkdir(parents=True, exist_ok=True)
        with open(save_l, "w") as f:
            json.dump(file, f)
        print(f"Saved answers to {save_l}")

    def evaluate_recall(
        self,
        student_result: list[MinimalSearchResults],
        ground_truth: list[AnsweredQuestion],
        k: int = 10,
        iou_threshhold: float = 0.05,
    ) -> float:
        """
        compares retrieved sources from a dataset to a ground-truth dataset"""
        if not student_result or not ground_truth:
            return 0.0
        gt_by_question_id: dict[str, list[MinimalSource]] = {}
        for entry in ground_truth:
            entry_pyd = AnsweredQuestion(**dict(entry))
            qid = entry_pyd.question_id
            gt_by_question_id[qid] = entry_pyd.sources
        scores: list[float] = []
        for result in student_result:
            qid = result.question_id
            if qid not in gt_by_question_id:
                continue
            sources = gt_by_question_id[qid]
            if not sources:
                continue
            retrieved = result.retrieved_sources[:k]
            found = 0
            for source in sources:
                source_path = source.file_path
                source_start = source.first_character_index
                source_end = source.last_character_index
                for chunk in retrieved:
                    if chunk.file_path != source_path:
                        continue
                    intersection = max(
                        0,
                        min(source_end, chunk.last_character_index)
                        - max(source_start, chunk.first_character_index),
                    )
                    union = max(source_end, chunk.last_character_index) - min(
                        source_start, chunk.first_character_index
                    )
                    if union <= 0:
                        continue
                    if (intersection / union) >= iou_threshhold:
                        found += 1
                        break
            scores.append(found / len(sources))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
