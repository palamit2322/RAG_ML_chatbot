from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def file_loader(path):
    loader=DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )    
    documents=loader.load()
    return documents

def chunking_data(split_documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunk_data=text_splitter.split_documents(split_documents)
    return chunk_data

def get_embedding():
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    return embedding