from ingest_vllm import chunk_vllm_docs
import bm25s
import Stemmer

class RagAgainstTheMachine:
    def __init__(self):
        self.docs = []
        self.retriever = bm25s.BM25()
    
    def index_docs(self):
        self.docs.extend(chunk_vllm_docs())
        if self.docs == []:
            raise Exception("Corpus is empty, please make sure to pass " +
                            "the correct document folder")

        # Insert Code chunking here  

        self.docs = list(map(str, self.docs))
        stemmer = Stemmer.Stemmer('english')
        print("\nTokenizing chunked documents...\n")
        corpus_tokens = bm25s.tokenize(self.docs, stopwords='en', stemmer=stemmer)
        print("\nIndexing chunked documents...\n")
        self.retriever.index(corpus_tokens)
        


if __name__ == "__main__":
    rag = RagAgainstTheMachine()
    rag.index_docs()
