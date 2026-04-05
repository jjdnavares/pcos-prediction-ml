"""
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.routers import health, predict
from app.dependencies import startup_check
from app.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOW_METHODS,
    allow_headers=settings.ALLOW_HEADERS,
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

# Root redirect to docs
@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API documentation"""
    return RedirectResponse(url="/docs")

# Startup event
@app.on_event("startup")
def startup_event():
    """Load model and resources at startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    startup_check()
    logger.info("✓ Application ready to serve requests")

# Shutdown event
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
