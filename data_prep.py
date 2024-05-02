import os
import pandas as pd 
import json
import argparse
import shutil
from langchain.schema import Document
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.chroma import Chroma

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-nyc-ce93903e254d.json'

DATA_PATH = 'data'
CHROMA_PATH = 'chroma'

embedding_function = VertexAIEmbeddings("textembedding-gecko")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks 


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    db = Chroma.from_documents(
        chunks, VertexAIEmbeddings("textembedding-gecko"), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    return
    
    
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    return

def main():
    generate_data_store()


if __name__ == "__main__":
    main()