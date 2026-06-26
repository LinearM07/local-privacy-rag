import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def initialize_rag_system(pdf_path="./document.pdf"):
    # 1. Load the PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split text into small chunks with context overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create vector embeddings completely locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Save chunks and embeddings into a local directory vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. Point LangChain to your running LM Studio local server
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1", 
        api_key="lm-studio",                  
        model="ignored",                      
        temperature=0.2                       
    )
    
    # 6. Set the System Prompt constraints
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the user's question accurately. If you don't know the answer "
        "or if it's not in the context, say that you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 7. Complete the RAG chain execution pipeline
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain