from pydantic import BaseModel
from typing import List, Optional

# --- Basic Chat and Upload Models ---

class UploadResponse(BaseModel):
    session_id: str
    message: str
    filenames: List[str]

class ChatRequest(BaseModel):
    session_id: str
    query: str
    role: str

class ChatMessage(BaseModel):
    sender: str  # "user" or "ai"
    content: str
    sources: Optional[List[str]] = []

class ChatResponse(BaseModel):
    session_id: str
    response: ChatMessage

# --- Session Management Models ---

class SessionSummary(BaseModel):
    session_id: str
    title: str

class SessionHistory(BaseModel):
    session_id: str
    role: str
    messages: List[ChatMessage]

