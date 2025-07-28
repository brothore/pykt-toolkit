import torch
import time

def safe_cuda_execution(func, max_retries=999, wait_seconds=10):
    for attempt in range(max_retries):
        try:
            return func()  # 执行可能耗尽内存的函数
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"CUDA OOM (尝试 {attempt+1}/{max_retries}), 等待 {wait_seconds}秒...")
                time.sleep(wait_seconds)
                torch.cuda.empty_cache()  # 清空缓存
            else:
                raise e
    raise RuntimeError(f"超过最大重试次数 {max_retries} 仍内存不足")
