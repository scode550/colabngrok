import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from models import ChatRequest, UploadResponse, ChatResponse
from document_processor import process_document
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Stakeholder RAG Chatbot API",
    description="API for uploading documents and chatting with them based on stakeholder roles.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Globals ---
# In a real-world scenario, you'd use a more persistent storage solution.
# For this project, we'll store the vector store in memory/local file.
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

vector_store = VectorStore(data_dir=DATA_DIR)
rag_pipeline = RAGPipeline(vector_store)
# --- End Globals ---

@app.get("/", tags=["Status"])
async def read_root():
    """Root endpoint to check API status."""
    return {"status": "API is running"}

@app.post("/upload", response_model=UploadResponse, tags=["Document Handling"])
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document (PDF).
    The document is processed, chunked, and stored in the vector database.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is supported.")
    
    try:
        file_path = os.path.join(DATA_DIR, file.filename)
        # Save the file temporarily
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info(f"Processing document: {file.filename}")
        chunks = process_document(file_path)
        
        # Add chunks to the vector store
        vector_store.add_documents(chunks, file.filename)
        logger.info(f"Successfully processed and indexed {len(chunks)} chunks from {file.filename}.")

        # Clean up the saved file
        os.remove(file_path)

        return {"filename": file.filename, "message": f"Successfully uploaded and processed."}
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_document(request: ChatRequest):
    """
    Endpoint to handle chat queries.
    It takes a user query and their role, then returns a context-aware answer.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.role:
        raise HTTPException(status_code=400, detail="Role must be specified.")
        
    try:
        logger.info(f"Received query: '{request.query}' for role: '{request.role}'")
        answer, sources, confidence = rag_pipeline.answer_query(request.query, request.role)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence_score=confidence
        )
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating a response: {str(e)}")

# To run this app: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
