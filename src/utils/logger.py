# src/utils/logger.py
import logging
import os
import sys
from datetime import datetime


class TeeHandler(logging.StreamHandler):
    """
    一个同时输出到控制台和文件的Handler
    """
    def __init__(self, file_handler, console_handler):
        super().__init__()
        self.file_handler = file_handler
        self.console_handler = console_handler
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.file_handler.stream.write(msg + self.terminator)
            self.file_handler.stream.flush()
            self.console_handler.stream.write(msg + self.terminator)
            self.console_handler.stream.flush()
        except Exception:
            self.handleError(record)


def setup_logger(output_dir=None, name="EcoG_Logger", log_to_file=True):
    """
    配置并返回一个 logger 实例。

    Args:
        output_dir (str, optional): 日志文件保存目录。如果不提供，则只输出到控制台。
        name (str): logger 名称。
        log_to_file (bool): 是否保存日志到文件。如果output_dir为None，此参数无效。

    Returns:
        logging.Logger: 配置好的 logger 对象。
        str: 日志文件路径（如果保存到文件），否则为None
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        # 如果已经有handlers，尝试找到文件handler并返回其文件路径
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return logger, handler.baseFilename
        return logger, None

    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    log_file = None
    
    # 2. 文件 Handler (如果指定了目录且log_to_file为True)
    if output_dir and log_to_file:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"run_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 同时添加控制台和文件handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # 输出日志文件路径（使用logger而不是print，避免循环）
        console_handler.stream.write(f"日志将保存至: {log_file}\n")
        console_handler.stream.flush()
    else:
        # 只输出到控制台
        logger.addHandler(console_handler)

    return logger, log_file
