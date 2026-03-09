import json
import base64
import os
import time
import logging
from abc import ABC, abstractmethod
import PIL.Image

# 使用与 main.py 相同的 logger 名称，确保日志会被保存到文件
logger = logging.getLogger("EcoG_Logger")
# 确保 logger 有合适的级别（如果还没有设置）
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


class BaseVLM(ABC):
    """
    Abstract base class for Visual Language Models (VLM).
    视觉语言模型 (VLM) 的抽象基类。
    """

    def __init__(self, coord_order="xy"):
        """
        Initialize the model.
        初始化模型。

        Args:
            coord_order: Coordinate order, "xy" for [x, y] (default), "yx" for [y, x].
                         坐标顺序，"xy" 表示 [x, y]（默认），"yx" 表示 [y, x]。
        """
        self.coord_order = coord_order

    @staticmethod
    def _convert_coordinates(result, coord_order):
        """
        Convert coordinate format from model output based on coordinate order.
        If coord_order="yx", convert [y, x] to [x, y].
        If coord_order="xy", keep [x, y] unchanged.

        Important: This conversion is performed immediately after model output to ensure
        all subsequent logic uses [x, y] format.

        根据坐标顺序转换模型输出的坐标格式。
        如果 coord_order="yx"，将 [y, x] 转换为 [x, y]。
        如果 coord_order="xy"，保持 [x, y] 不变。

        重要：此转换在模型输出后立即执行，确保后续所有逻辑都使用 [x, y] 格式。

        Args:
            result: Result dictionary from model output / 模型输出的结果字典
            coord_order: Coordinate order, "xy" or "yx" / 坐标顺序，"xy" 或 "yx"
        """
        if coord_order == "yx" and isinstance(result, dict) and "point_list" in result:
            for pred_item in result["point_list"]:
                if "point" in pred_item and isinstance(pred_item["point"], list):
                    pt = pred_item["point"]
                    # Case 1: Single point [y, x] -> [x, y]
                    if len(pt) == 2 and isinstance(pt[0], (int, float)):
                        pred_item["point"] = [pt[1], pt[0]]
                    # Case 2: Multiple points [[y1, x1], [y2, x2]] -> [[x1, y1], [x2, y2]]
                    elif len(pt) > 0 and isinstance(pt[0], list) and len(pt[0]) == 2:
                        pred_item["point"] = [[p[1], p[0]] for p in pt]
        return result

    def _parse_json_response(self, content):
        """
        Parse JSON content returned by the model, handling potential Markdown code blocks and common syntax errors.
        解析模型返回的 JSON 内容，处理可能的 Markdown 代码块和常見語法错误。
        """
        import re
        import json
        
        if content is None:
            logger.error("_parse_json_response received None input")
            return {}
            
        content = content.strip()
        if not content:
            logger.error("_parse_json_response received empty string")
            return {}

        # 1. Prioritize extracting ```json ... ``` code blocks
        # 1. 优先提取 ```json ... ``` 代码块
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        # 2. Try direct parsing / 2. 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 3. Extract Markdown code blocks (``` ... ```)
        # 3. 提取 Markdown 代码块 (``` ... ```)
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # 4. Extract the outermost braces {...}
        # 4. 提取最外层大括号 {...}
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common errors / 尝试修复常见错误
                # Fix: double brackets "]]" -> "]"
                json_str = json_str.replace("]]", "]")
                # Fix: double braces "}}" -> "}"
                json_str = json_str.replace("}}", "}")
                # Fix: trailing commas ",]" -> "]", ",}" -> "}"
                json_str = re.sub(r",\s*]", "]", json_str)
                json_str = re.sub(r",\s*}", "}", json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                
                # Special fix: LLaVA-NeXT-Video might miss the closing bracket "]" for point_list
                # 特殊修复: LLaVA-NeXT-Video 可能漏掉 point_list 的右括号 "]"
                json_str_bracket = re.sub(r'\}\s*\}$', '}\n    ]\n}', json_str)
                try:
                    return json.loads(json_str_bracket)
                except json.JSONDecodeError:
                    pass

        logger.error(f"Failed to parse JSON returned by model: {content[:200]}..." if len(content) > 200 else f"Failed to parse JSON returned by model: {content}")
        return {}

    @abstractmethod
    def generate(self, image_paths, prompt, system_prompt=None):
        """
        Generate response based on images and prompt.
        根据图像和提示生成响应。

        Args:
            image_paths (str or list): Path or list of paths to image files.
                                       图像文件的路径 or 路径列表。
            prompt (str): User input prompt / 用户输入的提示词。
            system_prompt (str, optional): System preset prompt / 系统预设提示词。

        Returns:
            dict: Parsed JSON response (coordinates unified to [x, y] format).
                  解析后的 JSON 响应（坐标已统一为 [x, y] 格式）。
        """
        pass

    def generate_from_video(self, video_path, prompt, system_prompt=None):
        """
        Generate response based on video file and prompt.
        Default implementation raises an error; subclasses should override this to support direct video input.
        根据视频文件和提示生成响应。
        默认实现为抛出错误，子类需覆盖此方法以支持直接视频输入。

        Returns:
            dict: Parsed JSON response (coordinates unified to [x, y] format).
                  解析后的 JSON 响应（坐标已统一为 [x, y] 格式）。
        """
        raise NotImplementedError("This model does not support direct video file input. Please use the generate method with a list of frames.")


class OpenAIVLM(BaseVLM):
    """
    Model wrapper using OpenAI-compatible API (e.g., GPT-4o, vLLM, etc.).
    使用 OpenAI 兼容 API (如 GPT-4o, vLLM 等) 的模型封装。
    """

    def __init__(self, api_key, base_url=None, model_name="gpt-4o", accepts_video_files=False, coord_order="xy", image_detail=None):
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("Missing openai library, please run: pip install openai")
            raise ImportError("Please install openai library: pip install openai")

        super().__init__(coord_order=coord_order)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.accepts_video_files = accepts_video_files  # Whether to support video file path input / 是否支持视频文件路径输入
        self.image_detail = image_detail  # Image resolution setting ("low", "high", "auto" or None) / 图像输入分辨率设置（"low", "high", "auto" 或 None）
        logger.info(
            f"OpenAIVLM initialization completed. Model: {model_name}, Base URL: {base_url or 'Default'}, Video Files: {accepts_video_files}, Coord Order: {coord_order}, Image Detail: {image_detail}")

    def _is_gpt_model(self):
        """Determine if it is a GPT model / 判断是否是 GPT 模型"""
        if not self.model_name:
            return False
        model_name_lower = self.model_name.lower()
        return "gpt" in model_name_lower

    def _encode_image(self, image_path):
        """Encode image to base64 / 将图像编码为 base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, image_paths, prompt, system_prompt=None, frame_timestamps_ms=None):
        """
        Generate response based on images and prompt.
        根据图像和提示生成响应。
        
        Args:
            image_paths: Path or list of paths to image files / 图像文件路径或路径列表
            prompt: User input prompt / 用户输入的提示词
            system_prompt: Optional system preset prompt / 系统预设提示词（可选）
            frame_timestamps_ms: Optional list of timestamps (ms) for each frame / 每帧对应的时间戳（毫秒）列表（可选）
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        logger.info(f"Calling model to process {len(image_paths)} image frames")

        base64_images = []
        for img_path in image_paths:
            try:
                base64_images.append(self._encode_image(img_path))
            except Exception as e:
                logger.error(f"Failed to read image: {img_path}, error: {e}")
                return {}

        messages = []

        # 1. System Prompt / 系统提示词
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 2. User Prompt (Text + Images with Timestamps) / 用户提示词（文本 + 带时间戳的图像）
        user_content = []
        
        # Add user prompt text first / 首先添加用户提示文本
        user_content.append({
            "type": "text",
            "text": prompt
        })

        # Add images, with timestamp text before each image (if available)
        # 添加图像，每张图像前添加时间戳文本块（如果有）
        for i, b64_img in enumerate(base64_images):
            if frame_timestamps_ms and i < len(frame_timestamps_ms):
                timestamp_ms = frame_timestamps_ms[i]
                user_content.append({
                    "type": "text",
                    "text": f"<timestamp_ms>{timestamp_ms}"
                })
            
            image_url_dict = {
                "url": f"data:image/jpeg;base64,{b64_img}"
            }
            # Add detail parameter for GPT models if configured
            # 如果是GPT模型且配置了图像分辨率，添加detail参数
            if self.image_detail and self._is_gpt_model():
                image_url_dict["detail"] = self.image_detail
            user_content.append({
                "type": "image_url",
                "image_url": image_url_dict
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # Force JSON mode for supported models / 强制 JSON 模式，如果是支持该参数的模型
                response_format={"type": "json_object"},
                max_tokens=5000,
                temperature=0.2,
                timeout=360
            )
            result_content = response.choices[0].message.content
            parsed_result = self._parse_json_response(result_content)
            # Convert coordinates immediately / 立即转换坐标格式
            parsed_result = self._convert_coordinates(
                parsed_result, self.coord_order)
            return parsed_result

        except Exception as e:
            logger.error(f"API call error: {e}")
            raise e

    def generate_from_video(self, video_path, prompt, system_prompt=None):
        """
        Generate response based on video file path and prompt.
        Suitable for services supporting direct video path input (e.g., HumanOmni, DashScope/Qwen).
        根据视频文件路径和提示生成响应。
        通过 video_url 字段传递视频文件路径给服务端。
        适用于支持视频文件路径输入的服务（如 HumanOmni、DashScope/Qwen）。
        """
        # Convert to absolute path / 转换为绝对路径，确保服务端能找到文件
        abs_video_path = os.path.abspath(video_path)
        logger.info(f"Calling model to process video file: {abs_video_path}")

        # Check if it's a DashScope/Qwen model / 检查是否是 DashScope/Qwen 模型
        is_dashscope_model = self.model_name and ("qwen" in self.model_name.lower() or "dashscope" in self.model_name.lower())

        messages = []

        # 1. System Prompt / 系统提示词
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 2. User Prompt (Text + Video) / 用户提示词（文本 + 视频）
        if is_dashscope_model:
            # DashScope/Qwen model: use base64 encoded video data
            # DashScope/Qwen 模型：使用 base64 编码的视频数据
            logger.info("DashScope/Qwen model detected, using base64 encoded video data")
            try:
                with open(abs_video_path, "rb") as video_file:
                    video_data = video_file.read()
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                # Determine MIME type based on file extension / 根据文件扩展名确定 MIME 类型
                video_ext = os.path.splitext(abs_video_path)[1].lower()
                mime_type_map = {
                    '.mp4': 'video/mp4',
                    '.avi': 'video/x-msvideo',
                    '.mov': 'video/quicktime',
                    '.mkv': 'video/x-matroska',
                    '.webm': 'video/webm'
                }
                mime_type = mime_type_map.get(video_ext, 'video/mp4')
                
                user_content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{mime_type};base64,{video_base64}"
                        }
                    }
                ]
            except Exception as e:
                logger.error(f"Failed to read video file: {e}")
                raise e
        else:
            # Other models: use file:// URL / 其他模型：使用 file:// URL
            user_content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"file://{abs_video_path}"
                    }
                }
            ]

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            # DashScope/Qwen model requires stream=True / DashScope/Qwen 模型需要 stream=True
            if is_dashscope_model:
                logger.info("Using streaming mode (required by DashScope/Qwen)")
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.2,
                    stream=True,
                    stream_options={"include_usage": True},
                    timeout=360
                )
                
                # Process streaming response / 处理流式响应
                result_content = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        result_content += chunk.choices[0].delta.content
                
                logger.info(f"Streaming response received, total length: {len(result_content)}")
                if not result_content or not result_content.strip():
                    logger.error(f"Streaming response content is empty: {repr(result_content)}")
                    raise ValueError(f"Streaming response content is empty, cannot parse JSON")
            else:
                # Standard OpenAI compatible API / 标准 OpenAI 兼容 API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.2,
                    timeout=360
                )
                result_content = response.choices[0].message.content
                if result_content is None:
                    logger.error("Model returned content is None")
                    raise ValueError("Model returned content is None")
            
            if not result_content or not result_content.strip():
                logger.error(f"Model returned content is empty: {repr(result_content)}")
                raise ValueError(f"Model returned content is empty, cannot parse JSON")
            
            parsed_result = self._parse_json_response(result_content)
            # Convert coordinates immediately / 立即转换坐标格式
            parsed_result = self._convert_coordinates(
                parsed_result, self.coord_order)
            return parsed_result

        except Exception as e:
            logger.error(f"API call error: {e}")
            raise e


class GeminiVLM(BaseVLM):
    """
    Model wrapper using Google Gemini API.
    使用 Google Gemini API 的模型封装。
    """

    def __init__(self, api_key, model_name="gemini-3-flash-preview", coord_order="yx", temperature=1.0):
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error(
                "Missing google-generativeai library, please run: pip install google-generativeai")
            raise ImportError(
                "Please install google-generativeai library: pip install google-generativeai")

        super().__init__(coord_order=coord_order)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.temperature = temperature
        self.accepts_video_files = True
        logger.info(
            f"GeminiVLM initialization completed. Model: {model_name}, Coord Order: {coord_order}, Temperature: {temperature}")

    def _wait_for_file_active(self, file_obj):
        """Wait for file processing to complete / 等待文件处理完成"""
        import google.generativeai as genai
        logger.info(f"Waiting for video file processing: {file_obj.name}")
        while file_obj.state.name == "PROCESSING":
            time.sleep(2)
            file_obj = genai.get_file(file_obj.name)

        if file_obj.state.name != "ACTIVE":
            raise ValueError(f"File processing failed: {file_obj.state.name}")
        logger.info(f"Video file processing completed: {file_obj.name}")
        return file_obj

    def generate_from_video(self, video_path, prompt, system_prompt=None):
        """Generate response based on video file path and prompt / 根据视频文件和提示生成响应"""
        import google.generativeai as genai
        logger.info(f"Uploading video to Gemini: {video_path}")

        try:
            # 1. Upload video / 上传视频
            video_file = genai.upload_file(video_path)

            # 2. Wait for processing / 等待处理
            video_file = self._wait_for_file_active(video_file)

            # 3. Combine Prompt / 组合 Prompt
            full_prompt = []
            if system_prompt:
                full_prompt.append(system_prompt)

            full_prompt.append(prompt)
            full_prompt.append(video_file)

            # 4. Generate content / 生成内容
            # Gemini 3 supports json_mode through generation_config
            # Gemini 3 支持 json_mode，通过 generation_config 配置
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": self.temperature,
            }

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            result_content = response.text
            parsed_result = self._parse_json_response(result_content)
            # Convert coordinates immediately / 立即转换坐标格式
            parsed_result = self._convert_coordinates(
                parsed_result, self.coord_order)

            # 5. Cleanup (optional, to save cloud space) / 清理（可选，这里为了节省云端空间）
            try:
                genai.delete_file(video_file.name)
                logger.debug(f"Deleted cloud temporary file: {video_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete cloud file: {e}")

            return parsed_result

        except Exception as e:
            logger.error(f"Gemini video inference error: {e}")
            raise e

    def generate(self, image_paths, prompt, system_prompt=None, frame_timestamps_ms=None):
        """Generate response based on image paths and prompt / 根据图像和提示生成响应"""
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        logger.info(f"Calling Gemini model to process {len(image_paths)} image frames")

        images = []
        for img_path in image_paths:
            try:
                images.append(PIL.Image.open(img_path))
            except Exception as e:
                logger.error(f"Failed to read image: {img_path}, error: {e}")
                return {}

        # Combine Prompt / 组合 Prompt
        full_prompt = []
        if system_prompt:
            full_prompt.append(system_prompt)

        full_prompt.append(prompt)
        # Add images, inserting timestamp before each image (if available), consistent with OpenAIVLM
        # 添加图像，每张图像前插入时间戳（如果有），与 OpenAIVLM 保持一致
        for i, img in enumerate(images):
            if frame_timestamps_ms and i < len(frame_timestamps_ms):
                full_prompt.append(f"<timestamp_ms>{frame_timestamps_ms[i]}")
            full_prompt.append(img)

        try:
            # Gemini supports json_mode through generation_config / Gemini 支持 json_mode，通过 generation_config 配置
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": self.temperature,
            }

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            result_content = response.text
            parsed_result = self._parse_json_response(result_content)
            # Convert coordinates immediately / 立即转换坐标格式
            parsed_result = self._convert_coordinates(
                parsed_result, self.coord_order)
            return parsed_result

        except Exception as e:
            logger.error(f"Gemini API call error: {e}")
            raise e
