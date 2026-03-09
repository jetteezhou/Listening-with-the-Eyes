# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # --- 1. API Basic Configuration / API 基础配置 ---
    # It is strongly recommended to configure via environment variables. Do not hardcode any keys here.
    # 强烈建议通过环境变量配置，不要在此处硬编码任何 Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")

    # --- 2. Default Model Behavior Configuration / 默认模型行为配置 ---
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")  # "openai" or "gemini" / "openai" 或 "gemini"
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-pro-preview")
    COORD_ORDER = os.getenv("COORD_ORDER", "yx")            # "xy" for [x, y], "yx" for [y, x] / "xy" 表示 [x, y], "yx" 表示 [y, x]
    
    USE_VIDEO_INPUT = os.getenv("USE_VIDEO_INPUT", "True").lower() == "true"
    USE_ASR_RESULT = os.getenv("USE_ASR_RESULT", "False").lower() == "true"
    FPS = int(os.getenv("FPS", "2"))
    GPT_IMAGE_DETAIL = os.getenv("GPT_IMAGE_DETAIL", "low")

    # --- 3. Path Configuration / 路径配置 ---
    DATA_ROOT_DIR = os.getenv("DATA_ROOT_DIR", "data/data_zn")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results/test_mode/")
    SAVE_LOG = os.getenv("SAVE_LOG", "True").lower() == "true"

    # --- 4. Parallel Execution Configuration / 并行执行配置 ---
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "20"))
    EVAL_NUM_WORKERS = int(os.getenv("EVAL_NUM_WORKERS", "30"))
    PARALLEL_MODELS = int(os.getenv("PARALLEL_MODELS", "1"))

    # --- 5. Multi-model Batch Run Configuration / 多模型批量运行配置 ---
    # If this list is not empty, main.py will ignore the single model configuration above.
    # 若此列表不为空，main.py 会忽略上面的单模型配置
    MODELS = [
        {
            "provider": "gemini",
            "name": "gemini-3-flash-preview",
            "coord_order": "yx",
            "use_video_input": True,
            "use_asr_result": False,
        }
    ]

    # --- 6. Ablation Study Configuration / 消融实验配置 ---
    TEMPORAL_ANCHOR_ABLATION_OUTPUT_DIR = "result-temporal-anchor-ablation/data_zn/"
    TEMPORAL_ANCHOR_ABLATION_MODELS = [
        {
            "provider": "gemini",
            "name": "gemini-3-flash-preview",
            "coord_order": "yx",
            "use_video_input": False,
            "use_asr_result": True,
            "ablation_mode": "no_frame_timestamps",
        },
        {
            "provider": "gemini",
            "name": "gemini-3-flash-preview",
            "coord_order": "yx",
            "use_video_input": False,
            "use_asr_result": True,
            "ablation_mode": "no_word_asr_timing",
        }
    ]
