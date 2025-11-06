# util_log.py
import traceback
from typing import Callable, Any

def log_exc(prefix: str, e: Exception):
    print(f"[ERROR] {prefix}: {repr(e)}")
    traceback.print_exc()

def safe_call(prefix: str, fn: Callable[[], Any], fallback: Any):
    try:
        return fn()
    except Exception as e:
        log_exc(prefix, e)
        return fallback
