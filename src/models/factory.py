import os
import logging
from config import Config
from src.models.base_vlm import OpenAIVLM, GeminiVLM

logger = logging.getLogger("EcoG_Factory")

class ModelFactory:
    @staticmethod
    def create_model(model_config, logger_instance=None):
        """
        Initialize model instance based on model configuration.
        根据模型配置初始化模型实例。

        Args:
            model_config: Model configuration dictionary, containing fields like provider, name, api_key, etc.
                          模型配置字典，包含 provider, name, api_key 等字段
            logger_instance: Optional logger / 可选的日志记录器

        Returns:
            Model instance / 模型实例
        """
        log = logger_instance or logger
        provider = model_config.get("provider")
        model_name = model_config.get("name")

        if not provider or not model_name:
            log.error(f"Missing provider or name in model config: {model_config}")
            return None

        # Get API key (priority: model config > global config)
        # 获取API密钥（优先使用模型配置中的，否则使用全局配置）
        if provider == "openai":
            # Check if it's a DashScope/Qwen model / 检查是否是 DashScope/Qwen 模型
            is_dashscope_model = model_name and ("qwen" in model_name.lower() or "dashscope" in model_name.lower())
            
            if is_dashscope_model:
                # DashScope models use specialized configuration / DashScope 模型使用专门的配置
                api_key = model_config.get("api_key") or Config.DASHSCOPE_API_KEY
                base_url = model_config.get("base_url") or Config.DASHSCOPE_BASE_URL
                
                log.info(f"DashScope model config check: api_key first 10 digits={api_key[:10] if api_key else 'None'}..., base_url={base_url}")
                
                if not api_key:
                    log.error(f"DashScope API Key not found (Model: {model_name}). Please configure DASHSCOPE_API_KEY in config.py or set environment variable.")
                    return None
            else:
                # Standard OpenAI model / 标准 OpenAI 模型
                api_key = model_config.get("api_key") or Config.OPENAI_API_KEY
                base_url = model_config.get("base_url") or Config.OPENAI_BASE_URL

                if not api_key:
                    log.error(f"OpenAI API Key not found (Model: {model_name}). Please configure OPENAI_API_KEY in config.py or set environment variable.")
                    return None

            log.info(f"Initializing OpenAIVLM (Model: {model_name}, Base URL: {base_url})")
            
            # Get video input configuration (priority: model config > global config)
            # 获取视频输入配置（优先使用模型配置中的，否则使用全局配置）
            use_video_input = model_config.get("use_video_input", Config.USE_VIDEO_INPUT)
            coord_order = model_config.get("coord_order", Config.COORD_ORDER)
            
            # Get image detail configuration: use Config.GPT_IMAGE_DETAIL for GPT models
            # 获取图像分辨率配置：如果是GPT模型，使用配置的GPT_IMAGE_DETAIL
            image_detail = None
            if model_name and "gpt" in model_name.lower():
                image_detail = model_config.get("image_detail", Config.GPT_IMAGE_DETAIL)
            
            return OpenAIVLM(
                api_key=api_key, 
                base_url=base_url,
                model_name=model_name, 
                accepts_video_files=use_video_input,
                coord_order=coord_order, 
                image_detail=image_detail
            )

        elif provider == "gemini":
            api_key = model_config.get("api_key") or Config.GEMINI_API_KEY

            if not api_key:
                log.error(f"Gemini API Key not found (Model: {model_name}). Please configure GEMINI_API_KEY in config.py or set environment variable.")
                return None

            log.info(f"Initializing GeminiVLM (Model: {model_name})")
            coord_order = model_config.get("coord_order", Config.COORD_ORDER)
            temperature = model_config.get("temperature", 0.0)
            
            return GeminiVLM(
                api_key=api_key, 
                model_name=model_name,
                coord_order=coord_order, 
                temperature=temperature
            )

        else:
            log.error(f"Unsupported model provider: {provider}")
            return None
