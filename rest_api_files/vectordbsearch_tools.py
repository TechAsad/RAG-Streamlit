from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.tools import tool
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)

class VectorSearchTools_chroma():

  @tool("Search the vector database")
  def dbsearch(query:str):
        """
        useful to search vector database and returns most relevant chunks
        """
        # Processing PDF and DOCX files
        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device': 'cpu'})
        db = Chroma(embedding_function= embeddings, persist_directory="./chroma_db")

        
        retrieved_docs = db.similarity_search(query, 5)


        plain_texts = [f"\n{doc.page_content} {doc.metadata}" for doc in retrieved_docs]
        sources = "-"
        # Concatenate plain texts and sources into formatted strings
    
        merged_texts = '\n'.join(plain_texts) + '\n' + '\n'.join(sources)


        return retrieved_docs
      