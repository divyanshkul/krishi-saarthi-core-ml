from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.vllm import router as vllm_router

# Create FastAPI instance
app = FastAPI(
    title="Krishi Saarthi - Agricultural AI Services",
    description="Core ML endpoint for Krishi Saarthi App - VLLM and KCC services",
    version="1.0.0"
)

# Include routers
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(vllm_router, prefix="/api/vllm", tags=["vllm"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
