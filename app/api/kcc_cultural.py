"""
KCC Cultural Practices API Endpoints for Krishi Saarthi
Agricultural Cultural Practices Q&A using fine-tuned TinyLlama
"""

from fastapi import APIRouter, Form, HTTPException
import time
from app.services.kcc.kcc_cultural_service import kcc_cultural_service
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
async def process_cultural_practices_query(
    query: str = Form(..., description="Agricultural cultural practices question or query")
):
    """
    Process agricultural cultural practices query using KCC Cultural Practices
    
    **Krishi Saarthi KCC Cultural Practices Endpoint**
    
    - Ask about farming practices (planting, irrigation, fertilization, pest management)
    - Get AI-powered responses from fine-tuned TinyLlama for cultural practices
    """
    request_start = time.time()
    
    logger.info(f"Processing KCC Cultural Practices query: {query}")
    
    try:
        response = await kcc_cultural_service.generate_response(query)
        
        total_time = time.time() - request_start
        logger.info(f"KCC Cultural Practices request completed in {total_time:.3f}s")
        
        return QueryResponse(
            success=True,
            query=query,
            response=response
        )
        
    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"KCC Cultural Practices request failed after {total_time:.3f}s: {e}")
        
        return QueryResponse(
            success=False,
            query=query,
            response="",
            error=str(e)
        )

@router.get("/status", response_model=ServiceStatus)
async def get_kcc_cultural_status():
    """
    Get KCC Cultural Practices service status
    """
    status = kcc_cultural_service.get_status()
    return ServiceStatus(
        service="KCC_Cultural_Practices",
        status="loaded" if status["loaded"] else "not_loaded",
        details=status
    )

@router.post("/load")
async def load_kcc_cultural_model():
    """
    Manually load the KCC Cultural Practices model
    """
    try:
        await kcc_cultural_service.load_model()
        return {"success": True, "message": "KCC Cultural Practices model loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load KCC Cultural Practices model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")