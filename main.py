"""
AI Agent Microservice - Main Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.routes import router


app = FastAPI(
    title="AI Agent with XLSX Analysis",
    description="An AI agent using Ollama for Excel file analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "AI Agent with XLSX Analysis",
        "model": settings.model_name,
        "endpoints": {
            "ws_chat": f"WS /ws/{{session_id}} - Real-time chat (primary)",
            "upload": "POST /upload - Upload Excel/CSV file",
            "chat": "POST /chat - Chat (REST fallback)",
            "analyze": "POST /analyze - Upload and analyze in one step",
            "health": "GET /health - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
