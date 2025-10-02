# cuda_retry.py
import torch
import re
import time
import random
import logging
from retrying import retry

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义OOM错误模式
OOM_PATTERNS = [
    r"out of memory",
    r"memory allocation failed",
    r"cuMemoryAllocate failed",
    r"CUDA out of memory",
    r"CUDA error: out of memory",
    r"allocator.c.*failed",
    r"cudaMalloc.*failed"
]

class CudaOOMRetryHandler:
    """管理CUDA OOM重试的状态"""
    def __init__(self, max_attempts=3, base_delay=1.0, max_delay=300.0, cleanup_cache=True):
        self.max_attempts = max_attempts
        self.attempt = 0
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.cleanup_cache = cleanup_cache

    def _calculate_delay(self):
        """计算指数退避等待时间，带随机抖动"""
        delay = min(self.base_delay * (2 ** self.attempt), self.max_delay)
        jitter = random.uniform(0, 0.1 * delay)  # 添加10%的随机抖动
        return delay + jitter

    def should_retry(self, exception):
        """检查是否需要重试，并执行清理和等待"""
        # 检查是否为OOM错误
        if isinstance(exception, torch.cuda.OutOfMemoryError):
            logger.info("检测到标准CUDA OOM错误")
        elif isinstance(exception, RuntimeError):
            error_str = str(exception).lower()
            for pattern in OOM_PATTERNS:
                if re.search(pattern, error_str):
                    logger.info(f"检测到匹配模式 '{pattern}' 的OOM错误")
                    break
            else:
                logger.warning(f"非CUDA OOM错误，重试: {type(exception).__name__}: {exception}")
                return True
        elif isinstance(exception, torch.cuda.CudaError):
            logger.info(f"检测到CUDA错误: {exception}")
        else:
            logger.warning(f"非CUDA OOM错误，重试: {type(exception).__name__}: {exception}")
            return True

        # 检查重试次数
        if self.attempt >= self.max_attempts:
            logger.error(f"已达到最大重试次数 {self.max_attempts}，停止重试")
            return False

        self.attempt += 1

        # 清理显存
        if self.cleanup_cache:
            torch.cuda.empty_cache()
            logger.info("已清理CUDA缓存")

        # 记录显存状态
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        logger.info(f"当前显存 - 已分配: {allocated:.2f} MB, 已预留: {reserved:.2f} MB")

        # 计算并执行等待
        delay = self._calculate_delay()
        logger.info(f"等待 {delay:.2f} 秒后重试（第 {self.attempt}/{self.max_attempts} 次）")
        time.sleep(delay)

        return True

def cuda_oom_retry_decorator(max_attempts=3, base_delay=1.0, max_delay=300.0, cleanup_cache=True):
    """
    修正版：采用双层 wrapper 结构，确保状态在独立调用间重置，在重试循环内保持。
    """
    def retry_if_cuda_oom(exception, handler=None):
        """包装retry_if_cuda_oom，注入handler"""
        if handler is None:
            raise ValueError("Retry handler not provided")
        return handler.should_retry(exception)


    def decorator(func):
        # Handler实例创建一次，用于管理该被装饰函数的所有状态
        handler = CudaOOMRetryHandler(max_attempts, base_delay, max_delay, cleanup_cache)
        
        # 1. 内部 Wrapper：仅负责重试逻辑
        @retry(
            retry_on_exception=lambda e: retry_if_cuda_oom(e, handler=handler),
            stop_max_attempt_number=max_attempts,
            wrap_exception=True
        )
        def retry_logic_wrapper(*args, **kwargs):
            # 重试循环期间，此函数被反复调用
            # 关键：此处不进行 handler.attempt = 0 的重置操作
            return func(*args, **kwargs)

        # 2. 外部 Wrapper：仅负责在函数被调用时，进行状态初始化/重置
        def reset_wrapper(*args, **kwargs):
            # 当用户第一次调用被装饰的函数时，将尝试次数重置为0
            handler.attempt = 0 
            # 然后将控制权交给内部的重试逻辑
            return retry_logic_wrapper(*args, **kwargs)
        
        return reset_wrapper
    
    return decorator


# 重试装饰器配置
retry_decorator = cuda_oom_retry_decorator(
    max_attempts=999,           # 最多重试36次
    base_delay=120,            # 初始等待1秒
    max_delay=300.0,           # 最大等待5分钟
    cleanup_cache=True         # 清理显存
)

# 快速重试版本（用于测试）
quick_retry_decorator = cuda_oom_retry_decorator(
    max_attempts=999,            # 最多重试3次
    base_delay=120,            # 初始等待0.5秒
    max_delay=300.0,            # 最大等待10秒
    cleanup_cache=True         # 清理显存
)