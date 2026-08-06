import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import trange
from pydantic_models import MinimalSource
import ast


class Chunker:
    
    @staticmethod
    def calc_and_add_end_index(doc: Document) -> dict:
        if not doc.metadata.get("end_index"):
            doc.metadata["end_index"] = (
                doc.metadata.get("start_index", 0) + len(doc.page_content)
            )
        doc.page_content = doc.page_content.strip("\n \t")
        return {"content": doc.page_content, "metadata": doc.metadata}

    @staticmethod
    def chunk_text(
        file_name: str, max_chunk: int, text: str, is_text: bool
    ) -> list:
        if not text or not text.strip():
            return []
        if len(text) <= max_chunk:
            return [
                Document(text.strip(), metadata={"start_index": text.find(text.strip(), 0), "src": file_name})
            ]
        doc_chunker = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk,
            chunk_overlap=min(200, max_chunk // 10),
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=True,
            add_start_index=True,
            separators=(
                ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
                if file_name.split(".")[-1] == "md"
                else ["\n\n", "\n", " ", ""]
            ),
        )
        docs = doc_chunker.create_documents(
            [text], metadatas=[{"src": file_name}]
        )
        return docs

    @staticmethod
    def chunk_vllm_docs(
        directory: str = "./data/raw/",
        file_type: list[str] | str = ["md", "txt"],
        maximum_chunk_size: int = 2000,
    ) -> list:
        if isinstance(file_type, str):
            file_type = [file_type]

        doc_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.split(".")[-1] in file_type:
                    doc_files.append(filepath)
        docs: list[dict[str, str | dict]] = []
        for i in trange(len(doc_files), desc="Chunking doc files..."):
            with open(doc_files[i], "r", encoding="utf-8") as f:
                text = f.read()
            doc = Chunker.chunk_text(
                doc_files[i][2:], maximum_chunk_size, text, True
            )
            doc = list(map(Chunker.calc_and_add_end_index, doc))
            docs.extend(doc)
        corpus = []
        for j in trange(len(docs), desc="Annotating doc chunks..."):
            doc_meta = docs[j].get("metadata")
            if not isinstance(doc_meta, dict):
                raise RuntimeError("data encapsulation violation!")
            doc_source = doc_meta.get("src", "placeholder.txt")
            doc_index_s = doc_meta.get("start_index", -1)
            doc_index_e = doc_meta.get("end_index", -1)
            corpus.append(
                {
                    "src": MinimalSource(
                        file_path=doc_source,
                        first_character_index=doc_index_s,
                        last_character_index=doc_index_e,
                    ),
                    "content": docs[j].get("content", ""),
                }
            )
        return corpus

    @staticmethod
    def chunk_python(file_name: str, max_chunk: int, text: str) -> list:
        if not text or not text.strip():
            return []
        if len(text) <= max_chunk:
            return [
                Document(text.strip(), metadata={"start_index": text.find(text.strip(), 0), "src": file_name})
            ]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return Chunker.chunk_text(file_name, max_chunk, text, False)
        lines = text.splitlines(True)
        offsets: list[int] = []
        offset = 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)
        top_level = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "lineno")]
        true_top_level = [node for node in top_level if not any(other is not node and other.lineno <= node.lineno and getattr(other, "end_lineno", 0) > node.lineno for other in top_level)]
        true_top_level.sort(key=lambda x: x.lineno)
        if not true_top_level:
            return Chunker.chunk_text(file_name, max_chunk, text, False)
        docs: list[Document] = []
        first_node_start = offsets[true_top_level[0].lineno - 1]
        if first_node_start > 0:
            prefix = text[first_node_start].strip()
            if prefix:
                docs.extend(Chunker.chunk_text(file_name, max_chunk, prefix, False))
        for node in true_top_level:
            node_start = offsets[node.lineno - 1]
            end_lineno = getattr(node, "end_lineno", node.lineno)
            if end_lineno < len(offsets):
                node_end = offsets[end_lineno]
            else:
                node_end = len(text)
            node_text = text[node_start:node_end]
            if len(node_text) <= max_chunk:
                docs.append(Document(node_text, metadata={"start_index": node_start, "end_index": node_end, "src": file_name}))
            else:
                sub_chunks = Chunker.chunk_text(file_name, max_chunk, node_text, False)
                for chunk in sub_chunks:
                    docs.append(Document(chunk.page_content, metadata={"src": file_name, "start_index": chunk.metadata.get("start_index", -1) + node_start}))
        return docs

    @staticmethod
    def chunk_vllm_code(
        directory: str = "./data/raw/",
        file_type: list[str] | str = "py",
        maximum_chunk_size: int = 2000,
    ) -> list:
        if isinstance(file_type, str):
            file_type = [file_type]
        code_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.split(".")[-1] in file_type:
                    code_files.append(filepath)
        docs: list = []
        for i in trange(len(code_files), desc="chunking code files..."):
            with open(code_files[i], "r", encoding="utf-8") as f:
                code = f.read()
            doc = Chunker.chunk_python(
                code_files[i][2:], maximum_chunk_size, code
            )
            doc = list(map(Chunker.calc_and_add_end_index, doc))
            docs.extend(doc)
        corpus = []
        for j in trange(len(docs), desc="Annotating code chunks..."):
            doc_meta = docs[j].get("metadata")
            if not isinstance(doc_meta, dict):
                raise RuntimeError("data encapsulation violation!")
            doc_source = doc_meta.get("src", "placeholder.txt")
            doc_index_s = doc_meta.get("start_index", -1)
            doc_index_e = doc_meta.get("end_index", -1)
            corpus.append(
                {
                    "src": MinimalSource(
                        file_path=doc_source,
                        first_character_index=doc_index_s,
                        last_character_index=doc_index_e,
                    ),
                    "content": docs[j].get("content", ""),
                }
            )
        return corpus


if __name__ == "__main__":
    chunker = Chunker()
    chunker.chunk_vllm_docs()
