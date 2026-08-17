from ingest_vllm import Chunker

# import torch
import bm25s
import Stemmer

# from transformers import AutoModelForCausalLM
from helpers import streamline_query, clean_text_chunks
import json
from pydantic_models import (
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
        self.stemmer = Stemmer.Stemmer("english")
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # try:
        #     self.llm = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         dtype=(
        #             torch.float16 if self.device == "cuda" else torch.float32
        #         ),
        #         device_map="auto" if self.device == "cuda" else None,
        #     ).to(self.device)
        # except RuntimeError:
        #     raise RuntimeError(
        #         "Could not fetch model, are you connected to the internet?"
        #     )

    def index_docs(self, maximum_chunk_size: int = 2000) -> None:
        if maximum_chunk_size < 1000:
            raise ValueError(
                "minimum chunk size is 1000 characters"
                + ", recommended size is 1500-2000"
            )
        elif maximum_chunk_size < 1500:
            print(
                "Warning: recommended chunk size for"
                + " is 1500-2000 characters"
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
            stemmer=self.stemmer,
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
        if (
            not Path("./data/processed/bm25_index.pkl").exists()
            or not Path("./data/chunks/chunks.json").exists()
        ):
            print("Index doesn't exist, please index the corpus first!")
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
            bm25s.tokenize(query, stemmer=self.stemmer), k=k
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
        file_name = set_file.split("/")[-1] if "/" in set_file else set_file
        if not save:
            save = f"./data/output/search/{file_name}"
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
        Path("./data/output/search").mkdir(parents=True, exist_ok=True)
        with open(save, "w") as f:
            json.dump(file, f)

    def answer_question(
        self,
        query: str,
        metadata: list[MinimalSource],
        retriever: bm25s.BM25,
        k: int = 5,
        id: str = "",
    ) -> MinimalAnswer:
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
        # insert llm prompting here:
        response = "llm integration not yet implemented."
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
        file_name = set_file.split("/")[-1] if "/" in set_file else set_file
        save = f"./data/output/answer/{file_name}"
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
        Path("./data/output/answer").mkdir(parents=True, exist_ok=True)
        with open(save, "w") as f:
            json.dump(file, f)

    def evaluate_recall(
        self,
        student_result: list[MinimalSearchResults],
        ground_truth: list[AnsweredQuestion],
        k: int = 10,
        iou_threshhold: float = 0.05,
    ) -> float:
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
                    # insert iou threshhold check here
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


if __name__ == "__main__":
    rag = RagAgainstTheMachine()
    rag.index_docs()
    retriever, metadata = rag.load_index()
    rag.search_set(
        "./data/datasets/UnansweredQuestions/dataset_docs_public.json",
        metadata,
        retriever,
        k=10,
    )
    rag.search_set(
        "./data/datasets/UnansweredQuestions/dataset_code_public.json",
        metadata,
        retriever,
        k=10,
    )
    rag.answer_set(
        "./data/datasets/UnansweredQuestions/dataset_docs_public.json",
        metadata,
        retriever,
        k=10,
    )
    rag.answer_set(
        "./data/datasets/UnansweredQuestions/dataset_code_public.json",
        metadata,
        retriever,
        k=10,
    )
    with open("./data/output/search/dataset_docs_public.json") as f:
        student_results_json = json.load(f)
    with open(
        "./data/datasets/AnsweredQuestions/dataset_docs_public.json"
    ) as f:
        ground_truth_json = json.load(f)
    student_results = StudentSearchResults(
        **student_results_json
    ).search_results
    ground_truth = ground_truth_json.get("rag_questions", [])
    print(rag.evaluate_recall(student_results, ground_truth))
