import os
import logging

class ConsoleLogFilter(logging.Filter):
    def filter(self, record):
        # 只有当消息中不包含 "Model: " 时，才返回 True，表示该消息应该被打印到控制台
        return "Model: " not in record.getMessage()

class Logger():
    def __init__(self, log_path):
        self.logobj = logging.getLogger()
        self.logobj.setLevel(logging.INFO)
        
        # 文件处理器 
        file_handler = logging.FileHandler(os.path.join(log_path, "train_log.txt"))

        # 流处理器 
        stream_handler = logging.StreamHandler()
        stream_handler.addFilter(ConsoleLogFilter())

        format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

        file_handler.setFormatter(format)
        stream_handler.setFormatter(format)

        # 添加处理器到日志记录器
        self.logobj.addHandler(file_handler)
        self.logobj.addHandler(stream_handler)

    def info(self, message):
        self.logobj.info(message)

    def warning(self, message):
        self.logobj.warning(message)

    def error(self, message):
        self.logobj.error(message)

    def debug(self, message):
        self.logobj.debug(message)