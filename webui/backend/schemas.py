from pydantic import BaseModel, Field
from typing import Optional, List, Union

class EvaluationRequest(BaseModel):
    # API Configuration
    model_provider: str = Field(..., description="API Provider: 'openai' or 'gemini'")
    api_base_url: Optional[str] = Field(None, description="API Base URL (for OpenAI compatible)")
    api_key: str = Field(..., description="API Key")
    model_name: str = Field(..., description="Model Name (e.g., gpt-4o, gemini-1.5-pro)")
    
    # Input Configuration
    input_mode: str = Field("video", description="'video' or 'frames'")
    num_frames: int = Field(8, description="Number of frames to extract if input_mode is 'frames'")
    
    # Optional Parameters
    system_prompt: Optional[str] = Field(None, description="Custom System Prompt")
    temperature: float = Field(0.2, description="Sampling temperature")
    max_tokens: int = Field(1000, description="Max tokens for generation")
    
    # Evaluation Config
    data_root_dir: str = Field("data_new", description="Root directory of the dataset")
    test_mode: bool = Field(False, description="Whether to run in test mode (one sample per instruction)")
    num_workers: int = Field(4, ge=1, le=30, description="Number of concurrent workers (max 30)")
    
    # Evaluation Model Config (Optional - if not provided, uses inference model)
    eval_model_provider: Optional[str] = Field(None, description="Evaluation Model Provider: 'openai' or 'gemini'")
    eval_model_name: Optional[str] = Field(None, description="Evaluation Model Name")
    eval_api_key: Optional[str] = Field(None, description="Evaluation Model API Key")
    eval_api_base_url: Optional[str] = Field(None, description="Evaluation Model API Base URL (for OpenAI compatible)")
    
    # Coordinate Order Config
    coord_order: Optional[str] = Field(None, description="Coordinate order: 'xy' for [x, y] or 'yx' for [y, x]. Default: 'yx' for gemini, 'xy' for others")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str # pending, processing, completed, failed
    progress: float
    message: Optional[str] = None
    result_path: Optional[str] = None
    logs: List[str] = []

