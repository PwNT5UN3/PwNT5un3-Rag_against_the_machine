import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm, trange


def calc_and_add_end_index(doc: Document):
    doc.metadata['end_index'] = doc.metadata.get('start_index', 0) + len(doc.page_content) - 1
    doc.page_content = doc.page_content.strip('\n \t')
    return doc

def chunk_vllm_docs(directory: str = './vllm-0.10.1/docs',
                    file_type: list[str] | str = ['md', 'txt'],
                    maximum_chunk_size: int = 2000):
    if isinstance(file_type, str):
        file_type = [file_type]
    if maximum_chunk_size < 500:
        raise ValueError("minimum chunk size for documents is 500 characters" +
                         ", recommended size is 1500-2000")
    elif maximum_chunk_size < 1500:
        print("Warning: recommended chunk size for documents" +
              " is 1500-2000 characters")
    elif maximum_chunk_size > 2000:
        raise ValueError("Maximum chunk size for documents is 2000 characters")
    doc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if (filepath.split('.')[-1] in file_type):
                doc_files.append(filepath)
    chunker = RecursiveCharacterTextSplitter(chunk_size=maximum_chunk_size,
                                             chunk_overlap=200,
                                             length_function=len,
                                             is_separator_regex=False,
                                             strip_whitespace=False,
                                             add_start_index=True,
                                             separators=["\n\n",
                                                         '\n', '.', ' ', ''])
    docs = []
    for i in trange(len(doc_files)):
        with open(doc_files[i], 'r', encoding='utf-8') as f:
            text = f.read()
        doc = chunker.create_documents([text], metadatas=[{"src": doc_files[i]}])
        doc = list(map(calc_and_add_end_index, doc))
        print()
        docs.extend(doc)
    return docs


if __name__ == '__main__':
    chunk_vllm_docs()
