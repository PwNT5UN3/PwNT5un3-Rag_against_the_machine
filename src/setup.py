from src.ingest_vllm import Chunker
import torch
import bm25s
import Stemmer
from transformers import AutoModelForCausalLM

class RagAgainstTheMachine:
    """main orchestrator, all commands are defined here"""

    def __init__(self, model_name = "Qwen/Qwen3-0.6B"):
        self.docs = []
        self.chunker = Chunker
        self.retriever = bm25s.BM25()
        self.indexed_corpus = False
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None
            ).to(self.device)
        except RuntimeError:
            raise RuntimeError("Could not fetch model, are you connected to the internet?")
    
    def index_docs(self):
        self.docs.extend(self.chunker.chunk_vllm_docs(self=self.chunker))
        self.docs.extend(self.chunker.chunk_vllm_code(self=self.chunker))
        if self.docs == []:
            raise Exception("Corpus is empty, please make sure to pass " +
                            "the correct corpus folder")

        self.docs = list(map(str, self.docs))
        print(self.docs)
        stemmer = Stemmer.Stemmer('english')
        print("\nTokenizing chunked documents...\n")
        corpus_tokens = bm25s.tokenize(self.docs, stopwords='en', stemmer=stemmer)
        print("\nIndexing chunked documents...\n")
        self.retriever.index(corpus_tokens)
        self.indexed_corpus = True
    
    def answer_question_test(self):
        pass
        


if __name__ == "__main__":
    rag = RagAgainstTheMachine()
    rag.index_docs()
