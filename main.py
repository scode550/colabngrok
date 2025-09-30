import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from document_processor import process_document
from rag_pipeline import RAGPipeline  # We will create one instance of this
from vector_store import VectorStore
from models import UploadResponse, ChatRequest, ChatResponse, ChatMessage, SessionHistory, SessionSummary

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="Multi-Stakeholder RAG API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SINGLE, GLOBAL RAG PIPELINE INSTANCE ---
# This is the crucial efficiency improvement. Models are loaded ONCE on startup.
try:
    rag_pipeline_instance = RAGPipeline()
    logging.info("Successfully loaded the global RAG pipeline instance.")
except Exception as e:
    logging.error(f"FATAL: Could not initialize RAG Pipeline. Error: {e}")
    # In a real app, you might exit or prevent startup.
    rag_pipeline_instance = None

# --- In-Memory Session Storage ---
SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- Helper Function to Get or Create Session ---
def get_or_create_session(session_id: str = None, role: str = None) -> Dict[str, Any]:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    
    if not role:
        raise HTTPException(status_code=400, detail="Role is required to create a new session.")

    new_session_id = str(uuid.uuid4())
    logging.info(f"Creating new session {new_session_id} for role {role}")
    
    # Each session gets its own vector store, but uses the GLOBAL pipeline
    vector_store = VectorStore(f"faiss_index_{new_session_id}")
    
    SESSIONS[new_session_id] = {
        "session_id": new_session_id,
        "vector_store": vector_store,
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
    session = get_or_create_session(session_id, role)
    current_session_id = session["session_id"]
    
    processed_filenames = []
    for file in files:
        file_location = f"temp_{file.filename}"
        try:
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())

            chunks = process_document(file_location, file.filename)
            session["vector_store"].add_documents(chunks)
            
            if file.filename not in session["filenames"]:
                session["filenames"].append(file.filename)
                
            processed_filenames.append(file.filename)
        finally:
            if os.path.exists(file_location):
                os.remove(file_location)
        
    return UploadResponse(
        session_id=current_session_id,
        message=f"Successfully processed {len(processed_filenames)} files.",
        filenames=session["filenames"]
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not rag_pipeline_instance:
         raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    if request.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session = SESSIONS[request.session_id]
    session["history"].append(ChatMessage(sender="user", content=request.query, sources=[]))

    answer, sources, confidence = rag_pipeline_instance.answer_query(
        query=request.query, 
        role=session["role"],
        vector_store=session["vector_store"] # Pass the session's specific vector store
    )
    
    ai_message = ChatMessage(sender="ai", content=answer, sources=sources)
    session["history"].append(ai_message)

    return ChatResponse(session_id=request.session_id, response=ai_message)

@app.get("/sessions/list", response_model=List[SessionSummary])
async def list_sessions():
    summaries = []
    for session_id, session_data in SESSIONS.items():
        title = "New Chat"
        if session_data["history"]:
            title = session_data["history"][0].content
        elif session_data["filenames"]:
            title = ", ".join(session_data["filenames"])
            
        summaries.append(SessionSummary(session_id=session_id, title=title[:75]))
    return sorted(summaries, key=lambda s: s.title)


@app.get("/sessions/{session_id}", response_model=SessionHistory)
async def get_session_history(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session = SESSIONS[session_id]
    return SessionHistory(
        session_id=session_id,
        role=session["role"],
        messages=session["history"]
    )

