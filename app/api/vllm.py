"""
VLLM API Endpoints for Krishi Saarthi
Agricultural image response generation using fine-tuned SmolVLM
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io
from app.services.vllm.vllm_service import vllm_service
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class GenerationResponse(BaseModel):
    success: bool
    response: str
    question: str
    error: Optional[str] = None

class ServiceStatus(BaseModel):
    service: str
    status: str
    details: dict

@router.post("/generate", response_model=GenerationResponse)
async def generate_agricultural_response(
    image: UploadFile = File(..., description="Agricultural image to analyze"),
    question: str = Form(..., description="Question about the image")
):
    """
    Generate response for agricultural image using fine-tuned SmolVLM
    
    **Krishi Saarthi Core ML Endpoint**
    
    - Upload an agricultural image (crops, leaves, plants, etc.)
    - Ask a question about the image
    - Get AI-powered agricultural response
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"Processing image: {image.filename}, Question: {question}")
        
        # Generate response using VLLM service
        response = await vllm_service.generate_response(pil_image, question)
        
        return GenerationResponse(
            success=True,
            response=response,
            question=question
        )
        
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        return GenerationResponse(
            success=False,
            response="",
            question=question,
            error=str(e)
        )

@router.get("/status", response_model=ServiceStatus)
async def get_vllm_status():
    """
    Get VLLM service status
    """
    status = vllm_service.get_status()
    return ServiceStatus(
        service="VLLM",
        status="loaded" if status["loaded"] else "not_loaded",
        details=status
    )

@router.post("/load")
async def load_vllm_model():
    """
    Manually load the VLLM model
    """
    try:
        await vllm_service.load_model()
        return {"success": True, "message": "VLLM model loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load VLLM model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
