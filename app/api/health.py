from fastapi import APIRouter
from datetime import datetime
from app.services.vllm.vllm_service import vllm_service

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Krishi Saarthi - Agricultural AI Services",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with more information
    """
    vllm_status = vllm_service.get_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Krishi Saarthi - Agricultural AI Services",
        "version": "1.0.0",
        "services": {
            "vllm": "loaded" if vllm_status["loaded"] else "not_loaded",
            "kcc": "not_implemented"
        },
        "uptime": "running",
        "details": {
            "vllm": vllm_status
        }
    }
