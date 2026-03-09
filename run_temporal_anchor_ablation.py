"""
Temporal Anchor Ablation 实验入口脚本
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from src.utils.logger import setup_logger
from src.eval_engine import EvaluationEngine


def _run_ablation_model(model_config, output_dir_root, model_idx, total):
    """
    在线程中运行单个 (model, ablation_mode) 消融条件。
    """
    model_name = model_config.get("name", f"model_{model_idx}")
    ablation_mode = model_config.get("ablation_mode", "full_anchors")
    model_name_safe = model_name.replace("/", "_").replace("\\", "_")
    run_key = f"{model_name_safe}__{ablation_mode}"

    # 该消融条件的专属输出目录
    run_output_dir = os.path.join(output_dir_root, run_key)
    log_dir = os.path.join(run_output_dir, "logs")
    model_logger, log_file = setup_logger(
        output_dir=log_dir,
        name=f"AblationLogger_{run_key}",
        log_to_file=True
    )
    if log_file:
        model_logger.info(f"日志文件: {log_file}")

    model_logger.info(f"\n{'#'*80}")
    model_logger.info(f"消融实验 [{model_idx+1}/{total}]: {run_key}")
    model_logger.info(f"输出目录: {os.path.abspath(run_output_dir)}")
    model_logger.info(f"{'#'*80}")

    try:
        # 配置模型专用输出路径
        conf = model_config.copy()
        conf["output_dir"] = run_output_dir
        
        # 使用 EvaluationEngine 执行评估
        engine = EvaluationEngine(conf, logger_instance=model_logger)
        engine.run()
        
        return run_key, True, None
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        model_logger.error(f"消融实验 {run_key} 出错: {e}\n{err}")
        return run_key, False, str(e)


def main():
    ablation_models = Config.TEMPORAL_ANCHOR_ABLATION_MODELS
    output_dir_root = Config.TEMPORAL_ANCHOR_ABLATION_OUTPUT_DIR

    if not ablation_models:
        print("Config.TEMPORAL_ANCHOR_ABLATION_MODELS 为空，请先在 config.py 中配置消融实验模型列表。")
        sys.exit(1)

    os.makedirs(output_dir_root, exist_ok=True)

    log_dir = os.path.join(output_dir_root, "logs")
    logger, log_file = setup_logger(output_dir=log_dir, log_to_file=True,
                                    name="TemporalAnchorAblation_Main")
    if log_file:
        logger.info(f"主日志文件: {log_file}")

    total = len(ablation_models)
    logger.info(f"Temporal Anchor Ablation 实验启动，共 {total} 个 (model, ablation_mode) 组合")
    logger.info(f"结果根目录: {os.path.abspath(output_dir_root)}")

    # 并行运行所有消融条件
    parallel = min(total, 10) # 限制并行数，避免资源耗尽
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(_run_ablation_model, config, output_dir_root, idx, total): config
            for idx, config in enumerate(ablation_models)
        }

        for future in as_completed(futures):
            mc = futures[future]
            try:
                run_key, success, error_msg = future.result()
                if success:
                    logger.info(f"✓ {run_key} 完成")
                else:
                    logger.error(f"✗ {run_key} 失败: {error_msg}")
            except Exception as e:
                logger.error(f"✗ 发生未捕获异常: {e}")

    logger.info("所有消融实验完成。")


if __name__ == "__main__":
    main()
