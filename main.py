"""
Main program for EcoG task.
Execution flow:
1. Parse configuration.
2. Initialize EvaluationEngine.
3. Execute evaluation process.

EcoG 任务主程序
逻辑流程：
1. 解析配置
2. 初始化评估引擎 (EvaluationEngine)
3. 执行评估流程
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config
from src.eval_engine import EvaluationEngine
from src.utils.logger import setup_logger

def _run_model_in_parallel(model_config, results_root, model_idx, total):
    """
    Run evaluation process for a single model in a parallel thread.
    Each model uses an independent logger to avoid multi-thread log confusion.

    在并行线程中运行单个模型的评估流程。
    每个模型使用独立的 logger，避免多线程日志混乱。

    Args:
        model_config: Model configuration dictionary / 模型配置字典
        results_root: Results root directory / 结果根目录
        model_idx: Model index / 模型索引
        total: Total number of models / 模型总数

    Returns:
        tuple: (model_name, success: bool, error_msg: str or None)
    """
    model_name = model_config.get("name", f"model_{model_idx}")
    model_name_safe = model_name.replace("/", "_").replace("\\", "_")
    log_dir = os.path.join(results_root, model_name_safe, "logs")

    # Create independent logger for each model / 为每个模型创建独立的 logger
    model_logger, log_file = setup_logger(
        output_dir=log_dir,
        name=f"EcoG_Logger_{model_name_safe}",
        log_to_file=True
    )
    if log_file:
        model_logger.info(f"Log file: {log_file}")

    model_logger.info(f"\n{'#'*80}")
    model_logger.info(f"Processing model [{model_idx+1}/{total}]: {model_name}")
    model_logger.info(f"{'#'*80}")

    try:
        # Complete necessary paths in model configuration / 补全模型配置中的必要路径
        conf = model_config.copy()
        results_root = Config.OUTPUT_DIR if Config.OUTPUT_DIR else "results"
        output_dir = os.path.join(results_root, model_name_safe)
        conf["output_dir"] = output_dir
        
        # Execute evaluation using EvaluationEngine / 使用 EvaluationEngine 执行评估
        engine = EvaluationEngine(conf, logger_instance=model_logger)
        engine.run()
        
        return model_name, True, None
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        model_logger.error(f"Error processing model {model_name}: {e}")
        model_logger.error(err)
        return model_name, False, str(e)

def main():
    # 0. Initialize Main Logger / 初始化主 Logger
    results_root = Config.OUTPUT_DIR if Config.OUTPUT_DIR else "results"
    
    # Determine log directory / 确定日志目录
    is_multi_model = hasattr(Config, 'MODELS') and Config.MODELS and len(Config.MODELS) > 0
    if is_multi_model:
        log_dir = os.path.join(results_root, "logs")
    else:
        model_name_safe = Config.MODEL_NAME.replace("/", "_").replace("\\", "_")
        log_dir = os.path.join(results_root, model_name_safe, "logs")

    logger, log_file = setup_logger(output_dir=log_dir, log_to_file=True)
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    logger.info("Ego-centric Co-Speech Intent Grounding (EcoG) task started")

    # 1. Prepare list of models to evaluate / 准备待评估的模型列表
    models = []
    if is_multi_model:
        models = Config.MODELS
        logger.info(f"Multi-model configuration detected, total {len(models)} models")
    else:
        # Single model mode: build model config from global configuration
        # 单模型模式：从全局配置构建模型配置
        models = [{
            "provider": Config.MODEL_PROVIDER,
            "name": Config.MODEL_NAME,
            "coord_order": Config.COORD_ORDER,
            "use_video_input": Config.USE_VIDEO_INPUT,
            "use_asr_result": Config.USE_ASR_RESULT,
            "api_key": None,  # Use global configuration / 使用全局配置
            "base_url": None,
        }]
        logger.info("Using single model configuration mode")

    # 2. Execute evaluation in parallel or serial / 并行或串行执行评估
    total = len(models)
    parallel_models = getattr(Config, 'PARALLEL_MODELS', total) if is_multi_model else 1
    parallel_models = min(parallel_models, total)

    if parallel_models > 1:
        logger.info(f"Starting parallel processing for {total} models, parallel count: {parallel_models}")
        with ThreadPoolExecutor(max_workers=parallel_models) as executor:
            futures = {
                executor.submit(_run_model_in_parallel, config, results_root, idx, total): config 
                for idx, config in enumerate(models)
            }
            for future in as_completed(futures):
                m_config = futures[future]
                try:
                    name, success, err = future.result()
                    if success:
                        logger.info(f"✓ Model {name} processed successfully")
                    else:
                        logger.error(f"✗ Model {name} processing failed: {err}")
                except Exception as e:
                    logger.error(f"✗ Uncaught exception: {e}")
    else:
        # Serial processing / 串行处理
        for idx, config in enumerate(models):
            name, success, err = _run_model_in_parallel(config, results_root, idx, total)
            if success:
                logger.info(f"✓ Model {name} processed successfully")
            else:
                logger.error(f"✗ Model {name} processing failed: {err}")

    logger.info("All tasks completed")

if __name__ == "__main__":
    main()
