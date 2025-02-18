from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Annotated, TypedDict
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sqlite3
import uuid
import json
import PyPDF2
from io import BytesIO
from pydantic import BaseModel
import asyncio
import os

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Single doc chatbot experiment - public version"

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this line
)

# Initialize database
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (conversation_id TEXT PRIMARY KEY,
                  document_text TEXT,
                  messages TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Chat message model
class ChatMessage(BaseModel):
    conversation_id: str
    user_message: str

# System message for the chatbot
SYSTEM_MESSAGE = """You are a helpful assistant that answers questions based on a provided document. 
Use the conversation history to maintain context. Summarize, extract key details, or answer specific questions as needed. 
Always reference the document content in your responses. You are not allowed to answer questions unrelated to topic of the document"""

class ChatbotAssistant:
    def __init__(self, document_text: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.document = document_text
    
    def get_response(self, messages: list) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_MESSAGE),
                *[HumanMessage(content=msg["content"]) if msg["role"] == "user" 
                  else AIMessage(content=msg["content"]) for msg in messages],
                HumanMessage(content=f"Document context: {self.document}")
            ]
        )
        return response.content

# Extract text from PDF
def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Read and extract text from PDF
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    
    # Generate conversation ID and store in database
    conversation_id = str(uuid.uuid4())
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations VALUES (?, ?, ?)",
              (conversation_id, text, json.dumps([])))
    conn.commit()
    conn.close()
    
    return {"conversation_id": conversation_id}

# Chat streaming endpoint
@app.post("/chat_stream")
async def chat_stream(message: ChatMessage):
    try:
        # Get conversation history and document from database
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute("SELECT document_text, messages FROM conversations WHERE conversation_id = ?",
                (message.conversation_id,))
        result = c.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        document_text, messages_json = result
        messages = json.loads(messages_json)
        
        # Add user message to history
        messages.append({"role": "user", "content": message.user_message})
        
        # Initialize chatbot with document context
        chatbot = ChatbotAssistant(document_text)

        async def generate_response():
            try:
                # Get complete response first
                assistant_message = chatbot.get_response(messages)
                
                # Update database with new messages
                messages.append({"role": "assistant", "content": assistant_message})
                c.execute("UPDATE conversations SET messages = ? WHERE conversation_id = ?",
                        (json.dumps(messages), message.conversation_id))
                conn.commit()
                
                # Stream response word by word
                for word in assistant_message.split():
                    yield f"{word} "
                    await asyncio.sleep(0.05)
            finally:
                conn.close()

        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "http://localhost:5173",
                "Access-Control-Allow-Credentials": "true"
            }
        )
    except Exception as e:
        print(f"Error in chat_stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat history retrieval endpoint
@app.get("/chat/{conversation_id}")
async def get_chat_history(conversation_id: str):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE conversation_id = ?",
              (conversation_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"messages": json.loads(result[0])}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
