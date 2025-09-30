from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """Request model for the /chat endpoint."""
    query: str = Field(..., description="The user's query.", example="What's the success rate of UPI transactions?")
    role: str = Field(..., description="The stakeholder role of the user.", example="Product Lead")

class ChatResponse(BaseModel):
    """Response model for the /chat endpoint."""
    answer: str = Field(..., description="The generated answer to the query.", example="The success rate of UPI transactions this month is 98.5%.")
    sources: List[str] = Field(..., description="List of source documents or chunks used to generate the answer.", example=["doc1_chunk_3", "doc1_chunk_5"])
    confidence_score: Optional[float] = Field(None, description="A score representing the confidence in the answer.", example=0.92)

class UploadResponse(BaseModel):
    """Response model for the /upload endpoint."""
    filename: str = Field(..., description="The name of the uploaded file.", example="upi_transactions_report.pdf")
    message: str = Field(..., description="A status message.", example="Successfully uploaded and processed.")
