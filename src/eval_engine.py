import os
import json
import logging
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config
from src.prompts.ecog_prompts import EcoGPrompts
from src.models.factory import ModelFactory
from src.utils.video_processor import VideoProcessor
from src.eval.metrics import Evaluator
from src.data_loader import DataLoader
from src.gt_formatter import GTFormatter

# Configure logger
logger = logging.getLogger("EcoG_Engine")
logger.setLevel(logging.INFO)

class EvaluationEngine:
    def __init__(self, config, status_callback=None, logger_instance=None):
        """
        Initialize evaluation engine.
        初始化评估引擎。

        Args:
            config: Configuration dictionary containing model, paths, etc.
                    配置字典，包含模型、路径等信息
            status_callback: Status update callback function / 状态更新回调函数
            logger_instance: Optional logger / 可选的日志记录器
        """
        self.config = config
        self.status_callback = status_callback
        self.model = None
        self.lang = "zh"  # Default language / 默认语言
        
        # Use provided logger or global logger
        # 使用传入的 logger 或全局 logger
        self.logger = logger_instance or logger
        
        # Configure logging if not already configured and output_dir exists
        # 如果没有配置日志文件且有 output_dir，则配置日志
        output_dir = config.get("output_dir")
        if output_dir and not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"eval_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            self.log(f"Log file: {log_file}")

    def log(self, message, level="info"):
        """Logging helper / 日志记录辅助函数"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

        if self.status_callback:
            self.status_callback(message)

    def init_models(self):
        """
        Initialize inference model.
        初始化推理模型。
        """
        # If model_provider/name is directly in config, build model config
        # 如果 config 中直接有 provider/name，则构建模型配置
        if "model_provider" in self.config:
            model_config = {
                "provider": self.config.get("model_provider"),
                "name": self.config.get("model_name"),
                "api_key": self.config.get("api_key"),
                "base_url": self.config.get("api_base_url"),
                "coord_order": self.config.get("coord_order"),
                "use_video_input": self.config.get("input_mode") == "video",
                "image_detail": self.config.get("image_detail"),
                "temperature": self.config.get("temperature", 0.0),
                "ablation_mode": self.config.get("ablation_mode", "full_anchors")
            }
        else:
            # Otherwise use config directly (assuming it's a model config dict)
            # 否则直接使用 config（假设它是模型配置字典）
            model_config = self.config

        self.log(f"Initializing inference model: {model_config.get('provider')}/{model_config.get('name')}")
        self.model = ModelFactory.create_model(model_config, self.logger)
        if not self.model:
            raise ValueError("Inference model initialization failed")

    def process_single_sample(self, formatted_gt, options_text, output_dir, idx, total):
        """
        Process single video sample.
        处理单个视频样本。
        """
        video_id = formatted_gt["video_name"]
        video_dir = formatted_gt["_video_dir"]
        self.log(f"Processing [{idx+1}/{total}]: {video_id}")

        video_path = os.path.join(video_dir, video_id)
        
        # Initialize default failure result / 初始化默认失败结果
        failure_result = {
            "explicit_command": "FAILED_INFERENCE",
            "selected_options": [],
            "point_list": [],
            "reasoning": "Inference failed or parsing error.",
            "video_name": video_id
        }

        # Get ASR text / 获取 ASR 文本
        asr_result = formatted_gt.get("asr_result")
        transcript = asr_result["text"] if (
            asr_result and isinstance(asr_result, dict) and "text" in asr_result) else formatted_gt.get("task_template")

        if formatted_gt.get("task_template") == "指令1" or not transcript:
            transcript = "The user did not speak, only made a pointing gesture." if self.lang == "en" else "用户没有说话，只是做出了指向性动作。"

        # Get model configuration / 获取模型配置
        use_video_input = self.config.get("use_video_input", Config.USE_VIDEO_INPUT)
        if "input_mode" in self.config:
            use_video_input = (self.config["input_mode"] == "video")
            
        coord_order = self.config.get("coord_order", Config.COORD_ORDER)
        fps = self.config.get("fps", Config.FPS)
        num_frames = self.config.get("num_frames", 15)
        use_asr_result = self.config.get("use_asr_result", Config.USE_ASR_RESULT)
        ablation_mode = self.config.get("ablation_mode", "full_anchors")

        # 1. Extract frames / Prepare input
        # 1. 抽帧 / 准备输入
        frame_paths = []
        last_frame_path = None
        frame_timestamps_ms = []

        try:
            if use_video_input and getattr(self.model, 'accepts_video_files', False):
                _, last_frame_path = VideoProcessor.extract_frame(video_path, timestamp_sec=None)
            else:
                # Frame extraction mode / 抽帧模式
                if "num_frames" in self.config:
                     frame_paths, last_frame_path, frame_timestamps_ms = VideoProcessor.extract_frames(
                        video_path, num_frames=num_frames, end_timestamp_sec=formatted_gt.get("timestamp"))
                else:
                    frame_paths, last_frame_path, frame_timestamps_ms = VideoProcessor.extract_frames(
                        video_path, fps=fps, end_timestamp_sec=formatted_gt.get("timestamp"))
        except Exception as e:
            self.log(f"Failed to process video {video_id}: {e}", "error")
            failure_result["reasoning"] = f"Video processing failed: {str(e)}"
            return failure_result, formatted_gt

        # 2. Build Prompt / 2. 构建 Prompt
        per_video_obj = formatted_gt.get("_object_choices", [])
        per_video_spc = formatted_gt.get("_space_choices", [])
        effective_options_text = GTFormatter.build_options_text(per_video_obj, per_video_spc) if (per_video_obj or per_video_spc) else options_text
        
        strip_word_timestamps = (ablation_mode == "no_word_asr_timing")
        system_prompt = EcoGPrompts.get_system_prompt(
            task_template=formatted_gt.get("task_template"),
            coord_order=coord_order,
            options_text=effective_options_text,
            lang=self.lang
        )
        user_prompt = EcoGPrompts.get_user_prompt(
            transcript, asr_result=asr_result, lang=self.lang, 
            use_asr_result=use_asr_result, strip_word_timestamps=strip_word_timestamps
        )

        # 3. Model Inference / 3. 模型推理
        try:
            effective_frame_timestamps = None if ablation_mode == "no_frame_timestamps" else frame_timestamps_ms
            
            if use_video_input and getattr(self.model, 'accepts_video_files', False):
                result = self.model.generate_from_video(video_path, user_prompt, system_prompt=system_prompt)
            else:
                if not frame_paths:
                    raise ValueError("No frames extracted in frame extraction mode")
                result = self.model.generate(
                    frame_paths, user_prompt, system_prompt=system_prompt,
                    frame_timestamps_ms=effective_frame_timestamps
                )
        except Exception as e:
            self.log(f"Inference error for sample {video_id}: {e}", "error")
            failure_result["reasoning"] = f"Inference error: {str(e)}"
            return failure_result, formatted_gt

        if not result:
            failure_result["reasoning"] = "Model returned empty result"
            return failure_result, formatted_gt

        # 4. Handle result format / 4. 处理结果格式
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            result = result[0]
        
        if not isinstance(result, dict):
            failure_result["reasoning"] = f"Result format incorrect: {type(result)}"
            return failure_result, formatted_gt

        result["video_name"] = video_id
        
        # Add input/output info / 添加输入输出信息
        result["_model_input"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "use_video_input": use_video_input,
            "media_type": "video" if use_video_input else "images",
            "video_path": os.path.abspath(video_path) if use_video_input else None,
            "frame_paths": [os.path.abspath(fp) for fp in frame_paths] if not use_video_input else []
        }
        result["_model_output"] = {
            "explicit_command": result.get("explicit_command"),
            "selected_options": result.get("selected_options", []),
            "point_list": result.get("point_list", []),
            "reasoning": result.get("reasoning", "")
        }

        # 5. Visualization / 5. 可视化
        is_web_run = "web_runs" in output_dir
        if self.config.get("test_mode", False) or is_web_run or Config.SAVE_LOG:
            vis_path = os.path.join(output_dir, f"vis_{video_id}.jpg")
            try:
                processed_gt = formatted_gt.get("_processed_gt", {})
                VideoProcessor.visualize_points(
                    last_frame_path, result, vis_path, gt_json=formatted_gt, gt_items=processed_gt.get("items", []))
                
                # Add relative path for frontend / 为前端添加相对路径
                if is_web_run:
                    parts = vis_path.split("results/")
                    if len(parts) > 1:
                        result["visualization_rel_path"] = parts[-1]
            except Exception as e:
                self.log(f"Failed to visualize sample {video_id}: {e}", "warning")

        return result, formatted_gt

    def run(self):
        """
        Execute full evaluation process.
        执行完整的评估流程。
        """
        try:
            self.init_models()
            
            data_root = self.config.get("data_root_dir", Config.DATA_ROOT_DIR)
            data_root_basename = os.path.basename(data_root.rstrip(os.sep))
            self.lang = "zh" if data_root_basename.startswith("data_zn") else "en"
            self.log(f"Detected language: {self.lang}")
            
            output_dir = self.config.get("output_dir")
            if not output_dir:
                results_root = Config.OUTPUT_DIR or "results"
                model_name_safe = self.config.get("model_name", "model").replace("/", "_")
                output_dir = os.path.join(results_root, model_name_safe)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Scan directory / 扫描目录
            instruction_dirs = DataLoader.scan_data_root(data_root)
            if not instruction_dirs:
                self.log("No instruction folders found", "error")
                return None

            all_predictions = []
            all_ground_truths = []
            
            # Process by instruction / 按指令处理
            for inst_name in sorted(instruction_dirs.keys()):
                inst_paths = instruction_dirs[inst_name]
                inst_output_dir = os.path.join(output_dir, inst_name)
                os.makedirs(inst_output_dir, exist_ok=True)
                
                inst_predictions = []
                inst_ground_truths = []
                
                for dataset_name, dataset_path, instruction_path in inst_paths:
                    self.log(f"Processing dataset: {dataset_name}/{inst_name}")
                    
                    # Load and format GT / 加载和格式化 GT
                    meta_file = os.path.join(instruction_path, "annotations.json")
                    dataset, options_text, video_eval_data = DataLoader.prepare_dataset(instruction_path, meta_file)
                    
                    if self.config.get("test_mode", False):
                        dataset = dataset[:1]
                        
                    formatted_gt_list = GTFormatter.format_batch_gt_for_evaluation(dataset, instruction_path, video_eval_data)
                    
                    # Handle breakpoint resumption / 处理断点续传
                    dataset_results_file = os.path.join(inst_output_dir, f"results_{dataset_name}.json")
                    existing_preds = {}
                    if os.path.exists(dataset_results_file):
                        try:
                            with open(dataset_results_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                for p in data.get("predictions", []):
                                    model_output = p.get("_model_output", {})
                                    is_failed = (
                                        p.get("explicit_command") == "FAILED_INFERENCE"
                                        or model_output.get("explicit_command") == "FAILED_INFERENCE"
                                    )
                                    if not is_failed:
                                        existing_preds[p["video_name"]] = p
                        except: pass

                    samples_to_process = [gt for gt in formatted_gt_list if gt["video_name"] not in existing_preds]
                    
                    results = []
                    for video_name, pred in existing_preds.items():
                        # Find corresponding GT / 需要找到对应的 GT
                        gt = next((g for g in formatted_gt_list if g["video_name"] == video_name), None)
                        if gt:
                            results.append((pred, gt))

                    # Parallel inference / 并行推理
                    num_workers = self.config.get("num_workers", Config.NUM_WORKERS)
                    if samples_to_process:
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = [executor.submit(self.process_single_sample, gt, options_text, inst_output_dir, i, len(samples_to_process)) 
                                       for i, gt in enumerate(samples_to_process)]
                            for f in as_completed(futures):
                                try:
                                    res = f.result()
                                    if res: results.append(res)
                                except Exception as e:
                                    self.log(f"Inference failed: {e}", "error")

                    if results:
                        preds, gts = zip(*results)
                        inst_predictions.extend(preds)
                        inst_ground_truths.extend(gts)
                        
                        # Save dataset results / 保存数据集结果
                        with open(dataset_results_file, "w", encoding="utf-8") as f:
                            json.dump({"predictions": list(preds), "ground_truths": list(gts)}, f, indent=2, ensure_ascii=False)

                if inst_predictions:
                    all_predictions.extend(inst_predictions)
                    all_ground_truths.extend(inst_ground_truths)
            
            # Final evaluation / 最终评估
            if all_predictions:
                self.log(f"Starting calculation of evaluation metrics for {len(all_predictions)} samples...")
                metrics = Evaluator.evaluate_batch(all_predictions, all_ground_truths, num_workers=Config.EVAL_NUM_WORKERS)
                
                # Save results / 保存结果
                metrics_path = os.path.join(output_dir, "metrics.json")
                summary_path = os.path.join(output_dir, "metrics_summary.json")
                
                # Save overall results / 保存总体结果
                final_results = {
                    "model_name": self.config.get("model_name", "Unknown"),
                    "model_provider": self.config.get("model_provider", "Unknown"),
                    "overall": metrics,
                    "statistics": {"total": len(all_predictions)}
                }
                
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                
                # Save individual metrics.json for each instruction (matching main.py output structure)
                # 保存每个指令的单独 metrics.json (保持 main.py 的输出结构)
                if "instruction_breakdown" in metrics:
                    for inst_name, inst_metrics in metrics["instruction_breakdown"].items():
                        inst_metrics_path = os.path.join(output_dir, inst_name, "metrics.json")
                        # Construct format identical to main.py / 构造与 main.py 相同的格式
                        inst_final_results = {
                            "acc_cls": inst_metrics.get("acc_cls", 0.0),
                            "acc_s": inst_metrics.get("acc_s", 0.0),
                            "acc_t": inst_metrics.get("acc_t", 0.0),
                            "acc_eco": inst_metrics.get("acc_eco", 0.0),
                            "acc_seq": inst_metrics.get("acc_seq", 0.0),
                            "instruction_breakdown": {inst_name: inst_metrics},
                            "count": inst_metrics.get("count", 0)
                        }
                        try:
                            os.makedirs(os.path.dirname(inst_metrics_path), exist_ok=True)
                            with open(inst_metrics_path, "w", encoding="utf-8") as f:
                                json.dump(inst_final_results, f, indent=2, ensure_ascii=False)
                        except: pass
                
                self.log("Evaluation completed.")
                return metrics_path
            
            return None

        except Exception as e:
            self.log(f"Critical error: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            raise e
