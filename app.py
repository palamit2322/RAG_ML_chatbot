from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.helper import get_embedding
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from  src.prompts import prompt,system_prompt
from fastapi.staticfiles import StaticFiles
from store import docs
from dotenv import load_dotenv
import os
load_dotenv()

app=FastAPI(title="RAG_ML_Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
embedding=get_embedding()

retriever=docs.as_retriever(search_type="similarity",search_kwargs={"k":3})

model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5
)

rag_chain=(
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }|prompt|model
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,   # REQUIRED
            "title": "Home Page",
            "name": "Amit"
        }
    )
@app.post("/chat")
def chat(question:str):
    response=rag_chain.invoke("What is supervised learning?")
    return{
        "answer":response.content
    }

if __name__ =='__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
