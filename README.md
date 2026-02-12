# AI Agent with Ollama + XLSX Analysis

An AI agent microservice using **qwen3-coder:480b-cloud** via Ollama for Excel file analysis.

## Quick Start

### 1. Start Ollama Server
```bash
docker compose up -d
```

### 2. Pull the Model
```bash
docker exec ollama-server ollama pull qwen3-coder:480b-cloud
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Agent API
```bash
python main.py
```

### 5. Test the API
```bash
# Upload and analyze Excel file
curl -X POST "http://localhost:3000/analyze" \
  -F "file=@your_data.xlsx" \
  -F "question=Summarize the key insights"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload Excel file |
| `/chat` | POST | Chat about data |
| `/analyze` | POST | Upload + analyze |
| `/health` | GET | Health check |

## Project Structure

```
ai-agent/
├── config/settings.py     # Configuration
├── agent/
│   ├── core.py           # ReAct agent
│   ├── tools.py          # Tool registry
│   └── xlsx_tool.py      # XLSX parsing
├── api/routes.py         # FastAPI routes
├── docker-compose.yml    # Ollama container
└── main.py               # Entry point
```

## Hardware Requirements

- **GPU**: RTX 4070 12GB (or similar)
- **RAM**: 16GB+
- **Model**: qwen3-coder:480b-cloud (served via Ollama)
