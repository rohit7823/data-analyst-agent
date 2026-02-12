"""
FastAPI Routes for the AI Agent.
Supports both REST and WebSocket communication.
"""

import os
import json
import uuid
import logging
import aiofiles
from agent.core import safe_dumps
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config.settings import settings
from agent.core import Agent

logger = logging.getLogger(__name__)


router = APIRouter()

# Store agents per session
agents: dict[str, Agent] = {}


def get_or_create_agent(session_id: str) -> Agent:
    """Get existing agent or create new one. Auto-reloads file if session was lost due to restart."""
    if session_id not in agents:
        agent = Agent()
        agents[session_id] = agent
        
        # Try to find and reload the file for this session (handles server restarts)
        upload_dir = os.path.abspath(settings.upload_dir)
        logger.info(f"New agent for session {session_id}, checking uploads in: {upload_dir}")
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                if filename.startswith(session_id):
                    file_path = os.path.join(upload_dir, filename)
                    logger.info(f"Auto-reloading file: {file_path}")
                    result = agent.load_file(file_path)
                    logger.info(f"Reload result: {result}")
                    break
            else:
                logger.warning(f"No file found for session {session_id} in {upload_dir}")
    else:
        logger.info(f"Reusing existing agent for session {session_id}")
    
    agent = agents[session_id]
    logger.info(f"Agent state - file loaded: {agent.current_file_path}, conversation msgs: {len(agent.conversation)}")
    return agent


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    sheets: list[str]
    message: str


# ──────────────────────────────────────────────
# WebSocket Endpoint (primary chat interface)
# ──────────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with the agent.
    
    Connect: ws://localhost:3000/ws/{session_id}
    
    Send:    {"message": "your question here"}
    
    Receive (streamed events):
      {"type": "connected",    "session_id": "...", "file_loaded": true/false}
      {"type": "status",       "content": "Thinking..."}
      {"type": "tool_call",    "tool": "run_pandas", "args": {...}}
      {"type": "tool_result",  "tool": "run_pandas", "result": {...}}
      {"type": "response",     "content": "Final answer..."}
      {"type": "error",        "content": "Error message"}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: session {session_id}")
    
    agent = get_or_create_agent(session_id)
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "file_loaded": agent.current_file_path is not None,
        "file_name": os.path.basename(agent.current_file_path) if agent.current_file_path else None
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                msg = json.loads(data)
                user_message = msg.get("message", "").strip()
            except json.JSONDecodeError:
                # Treat plain text as the message
                user_message = data.strip()
            
            if not user_message:
                await websocket.send_json({
                    "type": "error",
                    "content": "Empty message received"
                })
                continue
            
            logger.info(f"WS [{session_id}] message: {user_message[:50]}...")
            
            # Stream the agent's response
            for event in agent.chat_stream(user_message):
                await websocket.send_text(safe_dumps(event))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


# ──────────────────────────────────────────────
# REST Endpoints 
# ──────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": settings.model_name,
        "ollama_url": settings.ollama_base_url
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(default=None)
):
    """Upload an Excel file for analysis."""
    if not file.filename.endswith(('.xlsx', '.xls', '.xlsm', '.csv', '.tsv')):
        raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
    
    # Create session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Save file (use absolute path)
    file_path = os.path.abspath(os.path.join(settings.upload_dir, f"{session_id}_{file.filename}"))
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Load file into agent
    agent = get_or_create_agent(session_id)
    result = agent.load_file(file_path)
    
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    
    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        sheets=result.get("sheets", []),
        message=f"File loaded successfully. Found {len(result.get('sheets', []))} sheet(s). Connect via ws://localhost:{settings.api_port}/ws/{session_id} to chat."
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the agent (REST fallback — prefer WebSocket for real-time updates)."""
    logger.info(f"REST chat - session: {request.session_id}, message: {request.message[:50]}...")
    agent = get_or_create_agent(request.session_id)
    response = agent.chat(request.message)
    
    return ChatResponse(
        session_id=request.session_id,
        response=response
    )


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(default="Summarize the key insights from this data")
):
    """Upload and analyze an Excel file in one request."""
    if not file.filename.endswith(('.xlsx', '.xls', '.xlsm', '.csv', '.tsv')):
        raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")
    
    session_id = str(uuid.uuid4())
    
    file_path = os.path.abspath(os.path.join(settings.upload_dir, f"{session_id}_{file.filename}"))
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    agent = get_or_create_agent(session_id)
    load_result = agent.load_file(file_path)
    
    if load_result.get("error"):
        raise HTTPException(status_code=400, detail=load_result["error"])
    
    response = agent.chat(question)
    
    return {
        "session_id": session_id,
        "filename": file.filename,
        "sheets": load_result.get("sheets", []),
        "question": question,
        "analysis": response
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its agent."""
    if session_id in agents:
        del agents[session_id]
        return {"message": f"Session {session_id} deleted"}
    
    raise HTTPException(status_code=404, detail="Session not found")
