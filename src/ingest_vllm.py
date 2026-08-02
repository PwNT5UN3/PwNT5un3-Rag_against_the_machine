import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from tqdm import trange
from pydantic_models import MinimalSource

class Chunker:
    @staticmethod
    def calc_and_add_end_index(doc: Document) -> dict:
        doc.metadata['end_index'] = doc.metadata.get('start_index', 0) + len(doc.page_content) - 1
        doc.page_content = doc.page_content.strip('\n \t')
        return {'content': doc.page_content, 'metadata': doc.metadata}

    def chunk_text(self, file_name: str, max_chunk: int, text: str) -> list:
        if not text or text.strip():
            return []
        if text <= max_chunk:
            return [Document(text, metadata={'start_index': 0, 'src': file_name})]
        doc_chunker = RecursiveCharacterTextSplitter(chunk_size=max_chunk,
                                                              chunk_overlap=400,
                                                              length_function=len,
                                                              is_separator_regex=False,
                                                              strip_whitespace=False,
                                                              add_start_index=True,
                                                              separators=["#", "\n\n", '\n', ';', ' ', ''])
        docs = doc_chunker.create_documents([text], metadatas=[{"src": file_name}])
        return docs

    def chunk_vllm_docs(self, directory: str = './data/raw/',
                        file_type: list[str] | str = ['md', 'txt'],
                        maximum_chunk_size: int = 2000) -> list:
        if isinstance(file_type, str):
            file_type = [file_type]
        
        doc_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if (filepath.split('.')[-1] in file_type):
                    doc_files.append(filepath)
        docs = []
        for i in trange(len(doc_files), desc="Chunking doc files..."):
            with open(doc_files[i], 'r', encoding='utf-8') as f:
                text = f.read()
            doc = self.chunk_text(doc_files[i][2:], maximum_chunk_size, text) 
            doc = list(map(self.calc_and_add_end_index, doc))
            docs.extend(doc)
        corpus = []
        for doc in docs:
            corpus.append({"src": MinimalSource(file_path=doc["metadata"]["src"], first_character_index=doc["metadata"]["start_index"], last_character_index=doc["metadata"]["end_index"]), "content": doc["content"]})
        return corpus

    def chunk_python(self, file_name: str, max_chunk: int, text: str) -> list:
        if not text or text.strip():
            return []
        if text <= max_chunk:
            return [Document(text, metadata={'start_index': 0, 'src': file_name})]
        docs = []
        return docs

    def chunk_vllm_code(self, directory: str = './data/raw/', file_type: list[str] | str = 'py', maximum_chunk_size: int = 2000) -> list:
        if isinstance(file_type, str):
            file_type = [file_type]
        code_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.split('.')[-1] in file_type:
                    code_files.append(filepath)
        docs = []
        for i in trange(len(code_files), desc="chunking code files..."):
            with open(code_files[i], "r", encoding='utf-8') as f:
                code = f.read()
            doc = self.chunk_python(code_files[i][2:], maximum_chunk_size, code)
            doc = list(map(self.calc_and_add_end_index, doc))
            print(doc)
            docs.extend(doc)
        print(docs)
        corpus = []
        for doc in docs:
            corpus.append({"src": MinimalSource(file_path=doc["metadata"]["src"], first_character_index=doc["metadata"]["start_index"], last_character_index=doc["metadata"]["end_index"]), "content": doc["content"]})
        return corpus

if __name__ == "__main__":
    chunker = Chunker()
    chunker.chunk_vllm_docs()
