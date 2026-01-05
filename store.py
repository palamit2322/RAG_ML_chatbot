from src.helper import file_loader, chunking_data, get_embedding
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
load_dotenv()

file=file_loader(r"data/")
chunk_data=chunking_data(file)
embedding=get_embedding()

docs=FAISS.from_documents(documents=chunk_data,embedding=embedding)

