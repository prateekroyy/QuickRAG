!pip install langchain langchain-community langchain-groq faiss-cpu sentence-transformers

import os
import getpass
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GROQ_API_KEY"]= getpass.getpass("Enter your key: ")

def process_documents(docs):
  #split docs
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
  splits = text_splitter.split_documents(docs)

  #Embeddings
  embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
  vectorstore = FAISS.from_documents(documents=splits, embedding = embedding)
  retriever = vectorstore.as_retriever()

  #LLM and prompt template
  prompt = hub.pull("rlm/rag-prompt")
  llm = ChatGroq(model = "llama-3.1-8b-instant")
  
  def format_docs(docs):
    return "/n".join(doc.page_content for doc in docs)

  #RAGchanin
  rag_chain = (
      {"context": retriever | format_docs, "question":RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  return rag_chain

#Input choice
mode = input("Choose input type(PDF/WEB): ").strip().lower()

if mode =="web":
  link = input("Enter webpage URL: ").strip()
  loader = WebBaseLoader(web_paths=[link])
  docs = loader.load()
  rag_chain = process_documents(docs)

#Question loop
while True:
  question = input("Enter your question: ")
  if question.lower() == "exit":
    break
  answer = rag_chain.invoke(question)
  print("\nAnswer:", answer)
