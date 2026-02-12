"""
AI Agent Core - ReAct-style agent with tool calling via Ollama.
Uses SKILL.md patterns for Excel analysis best practices.
"""

import json
import math
import os
from typing import Any
from pathlib import Path
from openai import OpenAI
import pandas as pd
import numpy as np

from config.settings import settings
from agent.tools import Tool, registry
from agent.xlsx_tool import XLSXTool
from agent.memory import MemoryStore


def _clean_for_json(obj):
    """Recursively convert pandas/numpy types to JSON-safe Python types."""
    if obj is None:
        return None
    if obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if not pd.isna(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _clean_for_json(obj.tolist())
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


def safe_dumps(obj):
    """JSON-serialize with all pandas/numpy types converted to safe values."""
    return json.dumps(_clean_for_json(obj), default=str)


class Agent:
    """ReAct-style AI Agent with tool calling capabilities."""
    
    SYSTEM_PROMPT = """You are an intelligent AI assistant specialized in data analysis.
You follow Anthropic's xlsx skill best practices for Excel file handling.

When a user asks about data:
1. **Think**: Understand what the user wants
2. **Act**: Use available tools — prefer `run_pandas` for precise calculations
3. **Observe**: Review tool results
4. **Respond**: Provide a clear, insightful answer with numbers and context

## Excel Analysis Guidelines (from SKILL.md)
- Use pandas for data analysis, openpyxl for formulas/formatting
- Handle dates properly with parse_dates
- Specify data types to avoid inference issues
- For large files, focus on specific columns
- Always verify results make sense (check for NaN, zeros, edge cases)

## Tool Usage Rules
- The file is ALREADY LOADED — never ask the user for a file path
- Use `run_pandas` for any calculation, filtering, sorting, or grouping
- Use `analyze_excel` with action `summary` for quick overviews
- Use `analyze_excel` with action `column_info` for detailed column metadata
- Write precise pandas code: df.nlargest(), df.groupby(), df[df['col'] > val], etc.
- Always answer with specific numbers, not vague descriptions"""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.api_key
        )
        self.model = settings.model_name
        self.xlsx_tool = XLSXTool()
        self.memory = MemoryStore()
        self.session_id: str | None = None
        self._register_tools()
        self.conversation: list[dict] = []
        self.current_file_path: str | None = None
    
    def _register_tools(self) -> None:
        """Register available tools."""
        # XLSX Analysis Tool
        registry.register(Tool(
            name="analyze_excel",
            description="Get summary statistics or column metadata from the loaded Excel file. The file is already loaded.",
            function=self._handle_xlsx,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["summary", "column_info", "query"],
                        "description": "Action: summary (statistics), column_info (detailed column metadata), query (keyword search)"
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Optional sheet name to analyze"
                    },
                    "query": {
                        "type": "string",
                        "description": "Keyword query for 'query' action (e.g. 'sum', 'average', 'max', 'min', 'top', 'count')"
                    }
                },
                "required": ["action"]
            }
        ))
        
        # Dynamic Pandas Code Execution Tool
        registry.register(Tool(
            name="run_pandas",
            description="Execute pandas code on the loaded Excel data. The dataframe is available as 'df'. Use this for precise calculations, filtering, sorting, grouping, and any complex data operations. Examples: df.nlargest(5, 'Amount'), df.groupby('Category')['Revenue'].sum(), df[df['Cost'] > 1000], df['Price'].max(), df.describe()",
            function=self._handle_pandas,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Pandas expression to execute. The dataframe is 'df'. Use pd and np as needed."
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Optional sheet name (defaults to first sheet)"
                    }
                },
                "required": ["code"]
            }
        ))
        
        # Memory tools — let the LLM save/recall insights
        registry.register(Tool(
            name="save_memory",
            description="Save an important insight or fact to persistent memory for future reference. Use this when you discover key patterns, definitions, or findings about the data.",
            function=self._handle_save_memory,
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The insight or fact to remember"
                    }
                },
                "required": ["text"]
            }
        ))
        
        registry.register(Tool(
            name="recall_memory",
            description="Search past memories for relevant context. Use this when you need background info or past analysis results.",
            function=self._handle_recall_memory,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant memories"
                    }
                },
                "required": ["query"]
            }
        ))
    
    def _handle_xlsx(self, action: str, sheet_name: str = None, query: str = None) -> dict:
        """Handle XLSX tool calls."""
        if not self.xlsx_tool.dataframes:
            return {"error": "No file is loaded. The user must upload a file first via /upload."}
        
        if action == "summary":
            return self.xlsx_tool.get_summary(sheet_name)
        elif action == "column_info":
            return self.xlsx_tool.get_column_info(sheet_name)
        elif action == "query" and query:
            return self.xlsx_tool.query_data(query, sheet_name)
        else:
            return {"error": "Invalid action or missing parameters. Use 'summary', 'column_info', or 'query'."}
    
    def _handle_pandas(self, code: str, sheet_name: str = None) -> dict:
        """Handle pandas code execution."""
        if not self.xlsx_tool.dataframes:
            return {"error": "No file is loaded. The user must upload a file first via /upload."}
        
        return self.xlsx_tool.run_pandas_code(code, sheet_name)
    
    def _handle_save_memory(self, text: str) -> dict:
        """Save a memory via LLM tool call."""
        file_name = os.path.basename(self.current_file_path) if self.current_file_path else None
        mem_id = self.memory.add(text, metadata={"type": "insight", "file_name": file_name or ""}, session_id=self.session_id)
        return {"status": "saved", "memory_id": mem_id}
    
    def _handle_recall_memory(self, query: str) -> dict:
        """Recall memories via LLM tool call."""
        memories = self.memory.search(query, session_id=None)  # Search across all sessions
        if not memories:
            return {"result": "No relevant memories found."}
        return {"result": [{"text": m["text"], "relevance": m["score"]} for m in memories]}
    
    def _build_system_prompt(self, user_query: str = "") -> str:
        """Build the system prompt, including file context and relevant memories."""
        prompt = self.SYSTEM_PROMPT
        
        if self.xlsx_tool.dataframes:
            context = self.xlsx_tool.format_for_llm()
            prompt += f"""

=== LOADED EXCEL FILE ===
The user has already uploaded an Excel file. The data is shown below.
You MUST use this data to answer the user's questions directly.
Do NOT ask the user for a file path — the file is already loaded.
Use the `run_pandas` tool for precise calculations on this data.

{context}
=== END OF FILE DATA ==="""
        
        # Inject relevant memories
        if user_query:
            memory_context = self.memory.get_context_for_query(user_query)
            if memory_context:
                prompt += f"\n\n{memory_context}"
        
        return prompt
    
    def load_file(self, file_path: str) -> dict:
        """Load an Excel file for analysis."""
        self.current_file_path = file_path
        result = self.xlsx_tool.load_file(file_path)
        return result
    

    
    def chat(self, user_message: str) -> str:
        """Chat with the agent (REST fallback). Supports multi-round tool calling."""
        self.conversation.append({"role": "user", "content": user_message})
        
        system_prompt = self._build_system_prompt(user_query=user_message)
        messages = [{"role": "system", "content": system_prompt}] + self.conversation
        tools = registry.list_tools()
        
        try:
            for _round in range(settings.max_tool_rounds):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature
                )
                
                assistant_message = response.choices[0].message
                
                if not assistant_message.tool_calls:
                    # Final text response
                    content = assistant_message.content or ""
                    self.conversation.append({"role": "assistant", "content": content})
                    # Save to memory
                    self.memory.save_conversation_turn(
                        user_message, content,
                        session_id=self.session_id,
                        file_name=os.path.basename(self.current_file_path) if self.current_file_path else None
                    )
                    return content
                
                # Process tool calls, then loop for next round
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    result = registry.execute(func_name, **func_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": safe_dumps(result)
                    })
            
            # If we exhausted all rounds, return last content
            return assistant_message.content or "I reached the maximum number of analysis steps. Please refine your question."
            
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"
    
    def chat_stream(self, user_message: str):
        """
        Chat with the agent, yielding JSON events for WebSocket streaming.
        Supports multi-round tool calling — the LLM can chain multiple tools.
        """
        self.conversation.append({"role": "user", "content": user_message})
        
        system_prompt = self._build_system_prompt(user_query=user_message)
        messages = [{"role": "system", "content": system_prompt}] + self.conversation
        tools = registry.list_tools()
        
        yield {"type": "status", "content": "Thinking..."}
        
        try:
            for round_num in range(settings.max_tool_rounds):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature
                )
                
                assistant_message = response.choices[0].message
                
                if not assistant_message.tool_calls:
                    # Final text response
                    content = assistant_message.content or ""
                    self.conversation.append({"role": "assistant", "content": content})
                    # Save to memory
                    self.memory.save_conversation_turn(
                        user_message, content,
                        session_id=self.session_id,
                        file_name=os.path.basename(self.current_file_path) if self.current_file_path else None
                    )
                    yield {"type": "response", "content": content}
                    return
                
                # Process tool calls
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    yield {"type": "tool_call", "tool": func_name, "args": func_args}
                    
                    result = registry.execute(func_name, **func_args)
                    
                    yield {"type": "tool_result", "tool": func_name, "result": result}
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": safe_dumps(result)
                    })
                
                yield {"type": "status", "content": f"Processing results (round {round_num + 1})..."}
            
            # Max rounds reached
            yield {"type": "response", "content": "I reached the maximum number of analysis steps. Please refine your question."}
                
        except Exception as e:
            yield {"type": "error", "content": f"Error communicating with LLM: {str(e)}"}
    
    def reset(self) -> None:
        """Reset the conversation."""
        self.conversation = []
        self.xlsx_tool = XLSXTool()
        self.current_file_path = None

