import logging
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.vllm import router as vllm_router
from app.api.kcc_varieties import router as kcc_varieties_router
from app.api.kcc_cultural import router as kcc_cultural_router
from app.services.vllm.vllm_service import vllm_service
from app.services.kcc.kcc_varieties_service import kcc_varieties_service
from app.services.kcc.kcc_cultural_service import kcc_cultural_service

def setup_logging():
    """Configure comprehensive logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("app").setLevel(logging.INFO)

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger.info("Starting Krishi Saarthi Agri Core ML...")
    logger.info("=" * 50)
    
    startup_start = time.time()
    
    try:
        logger.info("Initializing VLLM model...")
        vllm_start = time.time()
        
        await vllm_service.load_model()
        
        vllm_time = time.time() - vllm_start
        logger.info(f"VLLM model initialized successfully in {vllm_time:.2f}s")
        
        logger.info("Initializing KCC Varieties model...")
        kcc_varieties_start = time.time()
        
        # KCC Varieties will automatically load credentials from .env and service account JSON
        await kcc_varieties_service.load_model()
        
        kcc_varieties_time = time.time() - kcc_varieties_start
        logger.info(f"KCC Varieties model initialized successfully in {kcc_varieties_time:.2f}s")
        
        logger.info("Initializing KCC Cultural Practices model...")
        kcc_cultural_start = time.time()
        
        # KCC Cultural Practices will automatically load credentials from .env and service account JSON
        await kcc_cultural_service.load_model()
        
        kcc_cultural_time = time.time() - kcc_cultural_start
        logger.info(f"KCC Cultural Practices model initialized successfully in {kcc_cultural_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    startup_time = time.time() - startup_start
    logger.info("=" * 50)
    logger.info(f"Krishi Saarthi startup completed in {startup_time:.2f}s")
    logger.info("Ready to serve requests")
    logger.info("=" * 50)
    
    yield
    
    logger.info("Shutting down Krishi Saarthi Agri Core ML...")

app = FastAPI(
    title="Krishi Saarthi - Agri Core ML",
    description="Core ML endpoint for Krishi Saarthi App - VLLM, KCC Varieties and KCC Cultural Practices services",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(vllm_router, prefix="/api/vllm", tags=["vllm"])
app.include_router(kcc_varieties_router, prefix="/api/kcc/varieties", tags=["kcc_varieties"])
app.include_router(kcc_cultural_router, prefix="/api/kcc/cultural", tags=["kcc_cultural"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
