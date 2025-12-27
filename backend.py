from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser


import streamlit as st

# Setting up MongoDB connection
MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "vector_store_database"
COLLECTION_NAME = "embeddings_rag"
ATLAS_VECTOR_SEARCH = "vector_index"

#vectorestore setup
def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vectore_store = MongoDBAtlasVectorSearch(
        collection=collection, 
        embedding=embeddings, 
        index_name=ATLAS_VECTOR_SEARCH
    )
    return vectore_store

#convert text to embeddings function
def ingest_text(text_content):
    vector_store = get_vector_store()
    docs = Document(page_content=text_content)
    vector_store.add_documents([docs])
    

#RAG response method
def get_rag_response(query):
    vector_store = get_vector_store()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context_text = "\n\n".join([doc.page_content for doc in docs])


    #prompt_template = """Use the following context from the user in order to provide an accurate answer."""

    #batching actual prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "use the following context to answer: \n\n{context}"),
        ("human", "{question}")
    ])
 
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke ({"context": context_text, "question": query})
    return {
        "answer": answer,
        "sources": docs
    }

#vector visualization method
def get_vectors_for_visualization(query):
    vector_store = get_vector_store()
    embeddings = vector_store.embeddings
    query_vector = embeddings.embed_query(query)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    doc_data = []
    for doc in docs:
        vec = embeddings.embed_query(doc.page_content)
        doc_data.append({
            "content": doc.page_content,
            "vector": vec,
            "type": "Document"
        })
    return {
        "query_vector": query_vector,
        "docs": doc_data
    }
