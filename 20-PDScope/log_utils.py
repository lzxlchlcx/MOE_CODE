import os
from datetime import datetime

_LOG_FILE = None
_LOG_LEVEL = 0  # 0=DEBUG, 1=INFO, 2=WARN

_LEVEL_MAP = {"DEBUG": 0, "INFO": 1, "WARN": 2}


def init_log(path="./log/linshi.txt", level="DEBUG"):
    global _LOG_FILE, _LOG_LEVEL
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _LOG_FILE = path
    _LOG_LEVEL = _LEVEL_MAP.get(level, 0)
    with open(_LOG_FILE, 'w') as f:
        f.write(f"=== Log started at {datetime.now()} ===\n")


def LOG(msg, level="INFO"):
    global _LOG_FILE
    if _LOG_FILE is None:
        init_log()
    lvl = _LEVEL_MAP.get(level, 1)
    if lvl < _LOG_LEVEL:
        return
    with open(_LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    # print(msg)
