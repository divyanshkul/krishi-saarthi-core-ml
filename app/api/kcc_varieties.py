"""
KCC API Endpoints for Krishi Saarthi
Agricultural Q&A using fine-tuned TinyLlama
"""

from fastapi import APIRouter, Form, HTTPException
import time
from app.services.kcc.kcc_varieties_service import kcc_varieties_service
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryResponse(BaseModel):
    success: bool
    query: str
    response: str
    error: Optional[str] = None

class ServiceStatus(BaseModel):
    service: str
    status: str
    details: dict

@router.post("/query", response_model=QueryResponse)
async def process_agricultural_query(
    query: str = Form(..., description="Agricultural question or query")
):
    """
    Process agricultural text query using KCC (Krishi Crop Companion)
    
    **Krishi Saarthi KCC Endpoint**
    
    - Ask agricultural questions (weather, crops, farming advice)
    - Get AI-powered responses from fine-tuned TinyLlama
    """
    request_start = time.time()
    
    logger.info(f"Processing KCC query: {query}")
    
    try:
        response = await kcc_varieties_service.generate_response(query)
        
        total_time = time.time() - request_start
        logger.info(f"KCC request completed in {total_time:.3f}s")
        
        return QueryResponse(
            success=True,
            query=query,
            response=response
        )
        
    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"KCC request failed after {total_time:.3f}s: {e}")
        
        return QueryResponse(
            success=False,
            query=query,
            response="",
            error=str(e)
        )

@router.get("/status", response_model=ServiceStatus)
async def get_kcc_status():
    """
    Get KCC service status
    """
    status = kcc_varieties_service.get_status()
    return ServiceStatus(
        service="KCC",
        status="loaded" if status["loaded"] else "not_loaded",
        details=status
    )

@router.post("/load")
async def load_kcc_model():
    """
    Manually load the KCC model
    """
    try:
        await kcc_varieties_service.load_model()
        return {"success": True, "message": "KCC model loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load KCC model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")