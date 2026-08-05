from ingest_vllm import Chunker
import torch
import bm25s
import Stemmer
from transformers import AutoModelForCausalLM
from helpers import streamline_query, clean_text_chunks
import json
from pydantic_models import MinimalSource, MinimalSearchResults, UnansweredQuestion


class RagAgainstTheMachine:
    """main orchestrator, all commands are defined here"""

    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.docs = []
        self.metadata = []
        self.retriever = bm25s.BM25()
        self.indexed_corpus = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(
                    torch.float16 if self.device == "cuda" else torch.float32
                ),
                device_map="auto" if self.device == "cuda" else None,
            ).to(self.device)
        except RuntimeError:
            raise RuntimeError(
                "Could not fetch model, are you connected to the internet?"
            )

    def index_docs(self, maximum_chunk_size: int = 2000):
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
        code_corpus = Chunker.chunk_vllm_code(maximum_chunk_size=maximum_chunk_size
        )
        self.docs.extend(
            clean_text_chunks(d.get("content")) for d in doc_corpus
        )
        self.docs.extend(d.get("content") for d in code_corpus)
        self.metadata.extend(d.get("src") for d in doc_corpus)
        self.metadata.extend(d.get("src") for d in code_corpus)
        if self.docs == []:
            raise Exception(
                "Corpus is empty, please make sure to pass "
                + "the correct corpus folder"
            )
        stemmer = Stemmer.Stemmer("english")
        print("\nTokenizing chunked documents...\n")
        tokens = bm25s.tokenize(
            self.docs, stopwords="en", stemmer=stemmer
        )
        print("\nIndexing chunked documents...\n")
        self.retriever.index(tokens)
        self.indexed_corpus = True

    def search(self, query: str, k: int = 5, id: str = ""):
        if id:
            question = UnansweredQuestion(question_id=id, question=query)
        else:
            question = UnansweredQuestion(question=query)
        if query.strip() == "":
            return MinimalSearchResults(question_id=question.question_id, question=question.question, retrieved_sources=[])
        query = streamline_query(query)
        results, scores = self.retriever.retrieve(
            bm25s.tokenize(query), k=k
        )
        retrieved = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            retrieved.append((doc, score))
        context_docs = [
            self.metadata[doc]
            for doc, _ in retrieved
        ]
        for d in context_docs:
            print(isinstance(d), MinimalSource)
        



    def answer_question_test(self):
        while True:
            query = input("Query: ")
            if query.strip() == "":
                break
            query = streamline_query(query)
            print(query)
            results, scores = self.retriever.retrieve(
                bm25s.tokenize(query), k=5
            )
            retrieved = []
            for i in range(results.shape[1]):
                doc, score = results[0, i], scores[0, i]
                print(f"Rank {i+1} (score: {score:.2f}): {doc}")
                retrieved.append((doc, score))
            context_docs = [
                ([self.docs[doc], self.metadata[doc]])
                for doc, _ in retrieved
            ]
            print("Retrieved:")
            for c, m in context_docs:
                print("\n", m, "\n--------------------------------\n")


if __name__ == "__main__":
    rag = RagAgainstTheMachine()
    rag.index_docs()
    rag.search("abbb")
