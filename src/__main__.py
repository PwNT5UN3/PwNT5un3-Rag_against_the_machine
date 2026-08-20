import fire

import json
from src.orchestrator import RagAgainstTheMachine
from src.pydantic_models import StudentSearchResults, AnsweredQuestion


class RagCLI:
    """A simple class building all user available functions"""
    def __init__(self) -> None:
        """constructor"""
        self.rag = RagAgainstTheMachine()

    def index(self, max_chunk_size: int = 2000) -> None:
        """calls the method to index the corpus"""
        self.rag.index_docs(maximum_chunk_size=max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        """searching lexically for a single query"""
        retriever, metadata = self.rag.load_index()
        result = self.rag.search_index(query, metadata, retriever, k)
        for i, s in enumerate(result.retrieved_sources):
            print(
                f"[{i + 1}] {s.file_path}",
                f"({s.first_character_index}:{s.last_character_index})",
            )

    def search_dataset(
        self, dataset_path: str, k: int = 5, save_directory: str | None = None
    ) -> None:
        """searching lexically for a list of queries pulled from a dataset"""
        retriever, metadata = self.rag.load_index()
        self.rag.search_set(
            dataset_path, metadata, retriever, k, save_directory
        )

    def answer(self, query: str, k: int = 5) -> None:
        """answers a question"""
        retriever, metadata = self.rag.load_index()
        result = self.rag.answer_question(query, metadata, retriever, k)
        print(result.answer)

    def answer_dataset(
        self, dataset_path: str, k: int = 5, save_directory: str | None = None
    ) -> None:
        """Answers all questions in a dataset"""
        retriever, metadata = self.rag.load_index()
        self.rag.answer_set(
            dataset_path, metadata, retriever, k, save_directory
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str, k: int = 5
    ) -> None:
        """cli caller for the evaluate functionality"""
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            student_data = json.load(f)
        with open(dataset_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        student = StudentSearchResults(**student_data)
        gt = [AnsweredQuestion(**d) for d in gt_data.get("rag_questions", [])]
        score = self.rag.evaluate_recall(student.search_results, gt, k=k)
        print("\nResults:")
        print(f" Questions evaluated : {len(student.search_results)}")
        print(f" k                   : {k}")
        print(f" Recall@{k:<13}: {score:.4f} ({score * 100:.1f}%)")


def main() -> None:
    """The entry point function calling fire to create the CLI"""
    try:
        fire.Fire(RagCLI)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("interrupted by user")


if __name__ == "__main__":
    main()
