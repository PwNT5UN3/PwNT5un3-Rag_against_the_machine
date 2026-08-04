from ingest_vllm import Chunker
import torch
import bm25s
import Stemmer
from transformers import AutoModelForCausalLM
from helpers import streamline_query, clean_text_chunks
import json


class RagAgainstTheMachine:
    """main orchestrator, all commands are defined here"""

    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.docs_doc = []
        self.metadata_doc = []
        self.docs_code = []
        self.metadata_code = []
        self.retriever_docs = bm25s.BM25()
        self.retriever_code = bm25s.BM25()
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
        self.docs_doc.extend(
            clean_text_chunks(d.get("content")) for d in doc_corpus
        )
        self.docs_code.extend(d.get("content") for d in code_corpus)
        self.metadata_doc.extend(d.get("src") for d in doc_corpus)
        self.metadata_code.extend(d.get("src") for d in code_corpus)
        if self.docs_doc == [] or self.docs_code == []:
            raise Exception(
                "Corpus is empty, please make sure to pass "
                + "the correct corpus folder"
            )
        stemmer = Stemmer.Stemmer("english")
        print("\nTokenizing chunked documents...\n")
        docs_tokens = bm25s.tokenize(
            self.docs_doc, stopwords="en", stemmer=stemmer
        )
        code_tokens = bm25s.tokenize(
            self.docs_code, stopwords="en", stemmer=stemmer
        )
        print("\nIndexing chunked documents...\n")
        self.retriever_docs.index(docs_tokens)
        self.retriever_code.index(code_tokens)
        self.indexed_corpus = True

    def answer_question_test(self):
        query = input("Query: ")
        query = streamline_query(query)
        print(query)
        results, scores = self.retriever_docs.retrieve(
            bm25s.tokenize(query), k=5
        )
        results2, scores2 = self.retriever_code.retrieve(
            bm25s.tokenize(query), k=5
        )
        retrieved_doc = []
        retrieved_code = []
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"Rank {i+1} (score: {score:.2f}): {doc}")
            retrieved_doc.append((doc, score))
        for i in range(results2.shape[1]):
            doc, score = results2[0, i], scores2[0, i]
            print(f"Rank {i+1} (score: {score:.2f}): {doc}")
            retrieved_code.append((doc, score))
        context_docs = [
            ([self.docs_doc[doc], self.metadata_doc[doc]])
            for doc, _ in retrieved_doc
        ]
        context_code = [
            ([self.docs_code[doc], self.metadata_code[doc]])
            for doc, _ in retrieved_code
        ]
        print("Retrieved:")
        for c, m in context_docs:
            print("\n", m, "\n--------------------------------\n")
        for c, m in context_code:
            print("\n", m, "\n--------------------------------\n")


if __name__ == "__main__":
    rag = RagAgainstTheMachine()
    rag.index_docs()
    rag.answer_question_test()
