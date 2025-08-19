from fastapi import APIRouter
from datetime import datetime
from app.services.vllm.vllm_service import vllm_service
from app.services.kcc.kcc_varieties_service import kcc_varieties_service
from app.services.kcc.kcc_cultural_service import kcc_cultural_service
import torch

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Krishi Saarthi - Agri Core ML",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with more information
    """
    vllm_status = vllm_service.get_status()
    kcc_varieties_status = kcc_varieties_service.get_status()
    kcc_cultural_status = kcc_cultural_service.get_status()
    
    # Get CUDA information
    cuda_info = {
        "available": torch.cuda.is_available()
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Krishi Saarthi - Agri Core ML",
        "version": "1.0.0",
        "services": {
            "vllm": "loaded" if vllm_status["loaded"] else "not_loaded",
            "kcc_varieties": "loaded" if kcc_varieties_status["loaded"] else "not_loaded",
            "kcc_cultural": "loaded" if kcc_cultural_status["loaded"] else "not_loaded"
        },
        "uptime": "running",
        "cuda": cuda_info,
        "details": {
            "vllm": vllm_status,
            "kcc_varieties": kcc_varieties_status,
            "kcc_cultural": kcc_cultural_status
        }
    }
