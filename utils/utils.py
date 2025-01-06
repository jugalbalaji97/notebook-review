# main.py

import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama


def ingest_text_file(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        data = open(doc_path, 'r').read()
        print("Text file loaded.")
        return data
    else:
        raise FileNotFoundError(f"Text file not found at path: {doc_path}")


def split_documents(text:str, chunk_size:int, chunk_overlap:int):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_db(chunks, embedding_model:str, vector_store_name:str = "vec_store"):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(embedding_model)

    vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            collection_name=vector_store_name,
        )
    print("Vector database created.")
    
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    print("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Chain created successfully.")
    return chain