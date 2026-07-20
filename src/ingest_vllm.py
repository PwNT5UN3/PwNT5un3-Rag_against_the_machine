import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from tqdm import trange
from pydantic import BaseModel

class Corpus(BaseModel):
    '''The corpus must be a list of strings representing the chunked corpus documents'''
    corpus: list[dict]

class Chunker:
    @staticmethod
    def calc_and_add_end_index(doc: Document) -> dict:
        doc.metadata['end_index'] = doc.metadata.get('start_index', 0) + len(doc.page_content) - 1
        doc.page_content = doc.page_content.strip('\n \t')
        return {'content': doc.page_content, 'metadata': doc.metadata}

    def chunk_vllm_docs(self, directory: str = './data/raw/',
                        file_type: list[str] | str = ['md', 'txt'],
                        maximum_chunk_size: int = 2000) -> Corpus:
        if isinstance(file_type, str):
            file_type = [file_type]
        if maximum_chunk_size < 1000:
            raise ValueError("minimum chunk size for documents is 1000 characters" +
                            ", recommended size is 1500-2000")
        elif maximum_chunk_size < 1500:
            print("Warning: recommended chunk size for documents" +
                " is 1500-2000 characters")
        elif maximum_chunk_size > 2000:
            raise ValueError("Maximum chunk size for documents is 2000 characters")
        doc_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if (filepath.split('.')[-1] in file_type):
                    doc_files.append(filepath)
        doc_chunker =CharacterTextSplitter(chunk_size=maximum_chunk_size, chunk_overlap=400, separator=" ", add_start_index=True)
        docs = []
        for i in trange(len(doc_files), desc="Chunking doc files..."):
            with open(doc_files[i], 'r', encoding='utf-8') as f:
                text = f.read()
            doc = doc_chunker.create_documents([text], metadatas=[{"src": doc_files[i][2:]}])
            doc = list(map(self.calc_and_add_end_index, doc))
            docs.extend(doc)
        corpus = Corpus(corpus=docs)
        return corpus

    def chunk_vllm_code(self, directory: str = './data/raw/', file_type: list[str] | str = 'py', maximum_chunk_size: int = 2000) -> Corpus:
        if isinstance(file_type, str):
            file_type = [file_type]
        if maximum_chunk_size != 2000:
            raise ValueError("Code chunks must have a maximum size of 2000")
        code_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.split('.')[-1] in file_type:
                    code_files.append(filepath)
        code_chunker = RecursiveCharacterTextSplitter(chunk_size=maximum_chunk_size,
                                                      chunk_overlap=400,
                                                      length_function=len,
                                                      is_separator_regex=False,
                                                      strip_whitespace=False,
                                                      add_start_index=True,
                                                      separators=["\n\n", '\n', ';', ' ', ''])
        docs = []
        for i in trange(len(code_files), desc="chunking code files..."):
            with open(code_files[i], "r", encoding='utf-8') as f:
                code = f.read()
            doc = code_chunker.create_documents([code], metadatas=[{'src': code_files[i][2:]}])
            doc = list(map(self.calc_and_add_end_index, doc))
            docs.extend(doc)
        corpus = Corpus(corpus=docs)
        return corpus
