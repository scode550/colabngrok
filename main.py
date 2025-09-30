import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from document_processor import process_document
from rag_pipeline import RAGPipeline
from vector_store import VectorStore
from models import UploadResponse, ChatRequest, ChatResponse, ChatMessage, SessionHistory, SessionSummary

app = FastAPI()

# --- CORS Middleware ---
# Allows the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Session Storage ---
# In a production environment, this would be a database (e.g., Redis, PostgreSQL)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- Helper Function to Get or Create Session ---
def get_or_create_session(session_id: str = None, role: str = None) -> Dict[str, Any]:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    
    if not role:
        raise HTTPException(status_code=400, detail="Role is required to create a new session.")

    new_session_id = str(uuid.uuid4())
    print(f"Creating new session {new_session_id} for role {role}")
    
    vector_store = VectorStore(f"faiss_index_{new_session_id}")
    rag_pipeline = RAGPipeline(vector_store)
    
    SESSIONS[new_session_id] = {
        "session_id": new_session_id,
        "rag_pipeline": rag_pipeline,
        "role": role,
        "history": [],
        "filenames": []
    }
    return SESSIONS[new_session_id]

# --- API Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: str = Body(None),
    role: str = Body(...)
):
    """
    Handles file uploads, creates a new session if one doesn't exist,
    processes documents, and adds them to the session's vector store.
    """
    session = get_or_create_session(session_id, role)
    current_session_id = session["session_id"]
    
    processed_filenames = []
    for file in files:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        chunks = process_document(file_location, file.filename)
        session["rag_pipeline"].vector_store.add_documents(chunks)
        
        session["filenames"].append(file.filename)
        processed_filenames.append(file.filename)
        os.remove(file_location)
        
    session["filenames"] = list(set(session["filenames"])) # Keep unique names

    return UploadResponse(
        session_id=current_session_id,
        message=f"Successfully processed {len(processed_filenames)} files.",
        filenames=session["filenames"]
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Handles a chat query for a specific session.
    """
    if request.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session = SESSIONS[request.session_id]
    
    # Add user message to history
    session["history"].append(ChatMessage(sender="user", content=request.query, sources=[]))

    # Get AI response
    answer, sources, confidence = session["rag_pipeline"].answer_query(request.query, session["role"])
    
    ai_message = ChatMessage(sender="ai", content=answer, sources=sources)
    session["history"].append(ai_message)

    return ChatResponse(session_id=request.session_id, response=ai_message)

@app.get("/sessions/list", response_model=List[SessionSummary])
async def list_sessions():
    """
    Returns a list of all active sessions with a title.
    """
    summaries = []
    for session_id, session_data in SESSIONS.items():
        if session_data["history"]:
            # Use the first user message as the title
            title = session_data["history"][0].content
        else:
            title = "New Chat" # Or based on uploaded files
        
        summaries.append(SessionSummary(session_id=session_id, title=title[:50])) # Truncate title
    return summaries

@app.get("/sessions/{session_id}", response_model=SessionHistory)
async def get_session_history(session_id: str):
    """
    Retrieves the full history for a given session.
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session = SESSIONS[session_id]
    return SessionHistory(
        session_id=session_id,
        role=session["role"],
        messages=session["history"]
    )

