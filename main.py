import logging
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.vllm import router as vllm_router
from app.services.vllm.vllm_service import vllm_service

# Configure logging
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

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting Krishi Saarthi Agri Core ML...")
    logger.info("=" * 50)
    
    startup_start = time.time()
    
    try:
        logger.info("Initializing VLLM model...")
        model_start = time.time()
        
        await vllm_service.load_model()
        
        model_time = time.time() - model_start
        logger.info(f"VLLM model initialized successfully in {model_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to initialize VLLM model: {e}")
        raise
    
    startup_time = time.time() - startup_start
    logger.info("=" * 50)
    logger.info(f"Krishi Saarthi startup completed in {startup_time:.2f}s")
    logger.info("Ready to serve requests")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Krishi Saarthi Agri Core ML...")

# Create FastAPI instance with lifespan
app = FastAPI(
    title="Krishi Saarthi - Agri Core ML",
    description="Core ML endpoint for Krishi Saarthi App - VLLM and KCC services",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(vllm_router, prefix="/api/vllm", tags=["vllm"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
