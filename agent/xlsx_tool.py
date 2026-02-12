"""
XLSX Tool - Excel parsing and analysis using Anthropic's xlsx skill patterns.
Enhanced with better pandas practices, column info, and dynamic code execution.
"""

import pandas as pd
import numpy as np
from typing import Any
from pathlib import Path
import traceback


class XLSXTool:
    """Tool for parsing and analyzing Excel files following SKILL.md best practices."""
    
    SUPPORTED_EXTENSIONS = ['.xlsx', '.xls', '.xlsm', '.csv', '.tsv']
    
    def __init__(self):
        self.current_file: str | None = None
        self.dataframes: dict[str, pd.DataFrame] = {}
    
    def load_file(self, file_path: str) -> dict[str, Any]:
        """Load an Excel/CSV file with proper dtype handling (SKILL.md pattern)."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return {"error": f"Unsupported file type: {path.suffix}. Supported: {self.SUPPORTED_EXTENSIONS}"}
        
        try:
            # Handle CSV/TSV separately
            if path.suffix.lower() == '.csv':
                self.dataframes = {"Sheet1": pd.read_csv(file_path)}
            elif path.suffix.lower() == '.tsv':
                self.dataframes = {"Sheet1": pd.read_csv(file_path, sep='\t')}
            else:
                # Read all sheets (SKILL.md pattern: sheet_name=None)
                self.dataframes = pd.read_excel(file_path, sheet_name=None)
            
            self.current_file = file_path
            
            # Post-process: detect and convert date columns
            for name, df in self.dataframes.items():
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                            if converted.notna().sum() > len(df) * 0.5:
                                self.dataframes[name][col] = converted
                        except Exception:
                            pass
            
            return {
                "success": True,
                "file": path.name,
                "sheets": list(self.dataframes.keys()),
                "sheet_info": {
                    name: {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "column_types": df.dtypes.astype(str).to_dict()
                    }
                    for name, df in self.dataframes.items()
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_column_info(self, sheet_name: str | None = None) -> dict[str, Any]:
        """Get detailed column metadata — types, unique counts, sample values, nulls."""
        if not self.dataframes:
            return {"error": "No file loaded."}
        
        if sheet_name is None:
            sheet_name = list(self.dataframes.keys())[0]
        
        if sheet_name not in self.dataframes:
            return {"error": f"Sheet '{sheet_name}' not found"}
        
        df = self.dataframes[sheet_name]
        columns_info = {}
        
        for col in df.columns:
            col_data = df[col]
            info = {
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.notna().sum()),
                "null_count": int(col_data.isna().sum()),
                "unique_count": int(col_data.nunique()),
            }
            
            # Add type-specific details
            if pd.api.types.is_numeric_dtype(col_data):
                info["min"] = float(col_data.min()) if col_data.notna().any() else None
                info["max"] = float(col_data.max()) if col_data.notna().any() else None
                info["mean"] = float(col_data.mean()) if col_data.notna().any() else None
                info["sum"] = float(col_data.sum()) if col_data.notna().any() else None
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                info["min_date"] = str(col_data.min()) if col_data.notna().any() else None
                info["max_date"] = str(col_data.max()) if col_data.notna().any() else None
            else:
                # Show sample unique values for categorical/text columns
                unique_vals = col_data.dropna().unique()[:10]
                info["sample_values"] = [str(v) for v in unique_vals]
            
            columns_info[str(col)] = info
        
        return {
            "sheet": sheet_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": columns_info
        }
    
    def get_summary(self, sheet_name: str | None = None) -> dict[str, Any]:
        """Get statistical summary of the data."""
        if not self.dataframes:
            return {"error": "No file loaded."}
        
        sheets = [sheet_name] if sheet_name else list(self.dataframes.keys())
        result = {}
        
        for name in sheets:
            if name not in self.dataframes:
                result[name] = {"error": f"Sheet '{name}' not found"}
                continue
            
            df = self.dataframes[name]
            
            # Numeric statistics
            numeric_stats = {}
            numeric_cols = df.select_dtypes(include='number')
            if not numeric_cols.empty:
                numeric_stats = numeric_cols.describe().to_dict()
            
            # Column types
            col_types = df.dtypes.astype(str).to_dict()
            
            result[name] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "column_types": col_types,
                "numeric_statistics": numeric_stats,
                "sample_data": df.head(5).to_dict(orient='records'),
                "null_counts": df.isnull().sum().to_dict()
            }
        
        return result
    
    def run_pandas_code(self, code: str, sheet_name: str | None = None) -> dict[str, Any]:
        """
        Execute a pandas expression on the loaded dataframe.
        The dataframe is available as 'df'. Returns the result.
        
        Examples:
            "df.nlargest(5, 'Amount')"
            "df.groupby('Category')['Revenue'].sum()"
            "df[df['Cost'] > 1000].sort_values('Cost', ascending=False)"
            "df.describe()"
            "df['Price'].max()"
        """
        if not self.dataframes:
            return {"error": "No file loaded."}
        
        if sheet_name is None:
            sheet_name = list(self.dataframes.keys())[0]
        
        if sheet_name not in self.dataframes:
            return {"error": f"Sheet '{sheet_name}' not found"}
        
        df = self.dataframes[sheet_name]
        
        # Strip import lines (pd and np are already available)
        lines = code.split('\n')
        lines = [l for l in lines if not l.strip().startswith(('import ', 'from '))]
        code = '\n'.join(lines).strip()
        
        if not code:
            return {"error": "No executable code after stripping imports. Use pd and np directly."}
        
        # Security: block truly dangerous operations
        blocked = ['exec(', 'eval(', 'open(', 'os.', 'sys.', 
                   'subprocess', '__', 'globals', 'locals', 'compile',
                   'delattr', 'setattr', 'getattr', 'breakpoint']
        for term in blocked:
            if term in code:
                return {"error": f"Blocked operation: '{term}' is not allowed"}
        
        try:
            # Restricted namespace with df, pd, np available
            namespace = {"df": df, "pd": pd, "np": np}
            
            # Try eval first (single expressions like df.head())
            try:
                result = eval(code, {"__builtins__": {}}, namespace)
            except SyntaxError:
                # Multi-line code: use exec, capture last expression via _result_
                # Append _result_ = <last line> if last line is an expression
                code_lines = code.strip().split('\n')
                last_line = code_lines[-1].strip()
                
                # If last line doesn't have assignment, capture its value
                if '=' not in last_line or last_line.startswith(('df[', 'df.')):
                    code_lines[-1] = f"_result_ = {last_line}"
                
                exec_code = '\n'.join(code_lines)
                exec(compile(exec_code, '<agent>', 'exec'), {"__builtins__": {}}, namespace)
                result = namespace.get("_result_", "Code executed successfully (no return value)")
            
            # Sanitize for JSON serialization (NaT, NaN, Timestamps, etc.)
            def _sanitize(obj):
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                elif isinstance(obj, float) and (pd.isna(obj) or np.isnan(obj)):
                    return None
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat() if not pd.isna(obj) else None
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj) if not np.isnan(obj) else None
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return _sanitize(obj.tolist())
                elif obj is pd.NaT:
                    return None
                return obj

            # Format result based on type
            if isinstance(result, pd.DataFrame):
                result = result.where(result.notna(), None)
                if len(result) > 50:
                    return {
                        "result": _sanitize(result.head(50).to_dict(orient='records')),
                        "note": f"Showing first 50 of {len(result)} rows",
                        "total_rows": len(result)
                    }
                return {"result": _sanitize(result.to_dict(orient='records'))}
            elif isinstance(result, pd.Series):
                return {"result": _sanitize(result.to_dict())}
            elif isinstance(result, (np.integer, np.floating)):
                val = float(result)
                return {"result": None if np.isnan(val) else val}
            elif isinstance(result, (int, float, str, bool)):
                return {"result": result}
            else:
                return {"result": str(result)}
                
        except Exception as e:
            return {"error": f"Code execution failed: {str(e)}"}
    
    def query_data(self, query: str, sheet_name: str | None = None) -> dict[str, Any]:
        """Execute a simple keyword-based query on the data."""
        if not self.dataframes:
            return {"error": "No file loaded."}
        
        if sheet_name is None:
            sheet_name = list(self.dataframes.keys())[0]
        
        if sheet_name not in self.dataframes:
            return {"error": f"Sheet '{sheet_name}' not found"}
        
        df = self.dataframes[sheet_name]
        query_lower = query.lower()
        
        if "sum" in query_lower:
            return {"result": df.select_dtypes(include='number').sum().to_dict()}
        elif "mean" in query_lower or "average" in query_lower:
            return {"result": df.select_dtypes(include='number').mean().to_dict()}
        elif "max" in query_lower or "highest" in query_lower:
            return {"result": df.select_dtypes(include='number').max().to_dict()}
        elif "min" in query_lower or "lowest" in query_lower:
            return {"result": df.select_dtypes(include='number').min().to_dict()}
        elif "count" in query_lower:
            return {"result": {"row_count": len(df), "column_count": len(df.columns)}}
        elif "top" in query_lower:
            return {"result": df.head(10).to_dict(orient='records')}
        elif "bottom" in query_lower or "last" in query_lower:
            return {"result": df.tail(10).to_dict(orient='records')}
        else:
            return self.get_summary(sheet_name)
    
    def format_for_llm(self, max_rows: int = 20) -> str:
        """Format the loaded data for LLM context with rich metadata."""
        if not self.dataframes:
            return "No Excel file loaded."
        
        output = []
        output.append(f"**Excel File:** {Path(self.current_file).name}\n")
        
        for sheet_name, df in self.dataframes.items():
            output.append(f"### Sheet: {sheet_name}")
            output.append(f"- **Rows:** {len(df)}, **Columns:** {len(df.columns)}")
            
            # Column details with types
            col_details = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                nulls = df[col].isna().sum()
                null_info = f" ({nulls} nulls)" if nulls > 0 else ""
                col_details.append(f"  - `{col}` ({dtype}){null_info}")
            output.append("- **Column Details:**")
            output.extend(col_details)
            
            # Sample data
            sample = df.head(max_rows)
            output.append("\n**Sample Data:**")
            output.append(sample.to_markdown(index=False))
            
            # Statistics for numeric columns
            numeric = df.select_dtypes(include='number')
            if not numeric.empty:
                output.append("\n**Numeric Statistics:**")
                output.append(numeric.describe().to_markdown())
            
            output.append("")
        
        return "\n".join(output)
