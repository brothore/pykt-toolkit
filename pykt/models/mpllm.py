import asyncio
from typing import List, Tuple, Optional
from openai import AsyncOpenAI
import numpy as np
from typing import *
from pykt.models.llm_config import Config
from pydantic import BaseModel
import json
import logging
import uuid
import os
import json
from typing import *
import textwrap

class MPLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:8102",
        model_path: str = "/root/qwen/Qwen3-8B",
        max_retries: int = 3,
        max_concurrent_requests: int = 16,
        timeout: int = 60,
        cache_dir: str = "mpllm_cache_qwen3-8b"
    ):
        self.emb_type = "qwen3-8b"
        self.model_name = "mpllm"
        self.base_url = base_url
        self.model_path = model_path
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.emb_type == "qwen3-8b":
            self.client = AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="no-key-required",
                timeout=timeout,
                max_retries=max_retries
            )
        elif self.emb_type == "deepseekv3":
            self.client = AsyncOpenAI(api_key=Config.api_key, base_url="https://api.deepseek.com")
            
        self.system_prompt = """你是一个知识追踪专家，需要根据学生的答题序列预测他们下一步答题的正确概率。
请严格按照以下要求执行任务：
0. [最重要]只输出数字列表，不要包含任何其他文字或解释，不需要文字解释的输出
1. 输入将提供多个学生的历史答题记录，格式为：(问题编号qid, 涉及的知识点编号kc, 回答是否正确correctness:True/False)
2. 你需要分析这些历史记录，预测每个学生回答下一个问题的正确概率
3. 输出必须是一个0到1之间的浮点数列表，表示每个学生的预测正确概率
4. 如果序列长度不足，无法进行预测，也请进行一定的猜测，给出结果来，并且不要给出文字解释
5. 输出格式示例：[0.7, 0.5, 0.8]"""

    def _get_cache_path(self, q_data: List[int]) -> str:
        batch_q_np = np.array([q_data])
        data_hash = hash(tuple(batch_q_np.tobytes()))
        return os.path.join(self.cache_dir, f"pred_{data_hash}.npy")

    async def _call_openai_api_batch(self, batch_prompts: List[str]) -> Optional[List[float]]:
        try:
            combined_prompt = "\n\n".join([
                f"学生{i+1}的历史答题记录:\n{prompt}" 
                for i, prompt in enumerate(batch_prompts)
            ])
            
            full_prompt = f"{self.system_prompt}\n\n{combined_prompt}\n\n请预测每个学生下一个问题的正确概率(输出格式如[0.1, 0.2, ...]):"
            
            response = await self.client.completions.create(
                model=self.model_path,
                prompt=full_prompt,
                max_tokens=100,
                temperature=0.1,
                stop=["\n"]
            )
            
            text = response.choices[0].text.strip()
            print(f"[DEBUG] 接收到批量返回text: {text}")
            
            try:
                # 尝试解析为列表
                if text.startswith("[") and text.endswith("]"):
                    probs = json.loads(text)
                    if isinstance(probs, list) and all(0 <= p <= 1 for p in probs):
                        return probs
                raise ValueError("返回格式不符合要求")
            except Exception as e:
                print(f"[ERROR] 解析批量预测结果失败: {str(e)}")
                return None
                
        except Exception as e:
            print(f"[ERROR] 批量请求失败: {str(e)}")
            return None

    def _construct_prompt(
        self,
        q_data: List[int],
        pid_data: List[int],
        target_data: List[int],
        predict_step: int
    ) -> str:
        history = []
        for i in range(predict_step):
            q = q_data[i]
            pid = pid_data[i]
            target = target_data[i]
            correctness = "true" if target == 1 else "false"
            history.append(f"{pid},{q},{correctness})")
        
        current_q = q_data[predict_step]
        current_pid = pid_data[predict_step]
        history.append(f"({current_pid}, {current_q}, 答题情况待预测)")

        return "\n".join(history)

    async def _process_sequence(
        self,
        q_data: List[int],
        pid_data: List[int],
        target_data: List[int]
    ) -> List[float]:
        cache_file = self._get_cache_path(q_data)
        
        if os.path.exists(cache_file):
            try:
                predictions = np.load(cache_file)
                if predictions.ndim == 2:
                    predictions = predictions[0]
                elif predictions.ndim != 1:
                    raise ValueError(f"无效的缓存维度: {predictions.shape}")
                    
                if not np.isnan(predictions).any():
                    print(f"[DEBUG] 从缓存加载预测结果: {cache_file}")
                    return predictions.tolist()
                    
                print(f"[WARNING] 缓存文件 {cache_file} 包含NaN值，将重新计算")
            except Exception as e:
                print(f"[WARNING] 加载缓存文件 {cache_file} 失败: {str(e)}，将重新计算")
        
        seq_len = len(q_data)
        if seq_len == 0:
            return []
            
        predictions = [0.0]  # 第一个时间步默认值
        
        for step in range(1, seq_len):
            prompt = self._construct_prompt(q_data, pid_data, target_data, step)
            prob = await self._call_openai_api(prompt)
            predictions.append(prob if prob is not None else 0.5)
        
        try:
            np.save(cache_file, np.array(predictions))
            print(f"[DEBUG] 预测结果已保存到: {cache_file}")
        except Exception as e:
            print(f"[ERROR] 保存缓存文件 {cache_file} 失败: {str(e)}")
        
        return predictions

    async def _process_batch_step(
        self,
        batch_q_data: List[List[int]],
        batch_pid_data: List[List[int]],
        batch_target_data: List[List[int]],
        step: int
    ) -> List[float]:
        """处理一个批次在特定时间步的预测"""
        batch_prompts = []
        valid_indices = []
        
        for i in range(len(batch_q_data)):
            if step < len(batch_q_data[i]):
                prompt = self._construct_prompt(
                    batch_q_data[i], 
                    batch_pid_data[i], 
                    batch_target_data[i], 
                    step
                )
                batch_prompts.append(prompt)
                valid_indices.append(i)
        
        if not batch_prompts:
            return []
            
        retry_count = 0
        while retry_count <= self.max_retries:
            probs = await self._call_openai_api_batch(batch_prompts)
            if probs is not None and len(probs) == len(batch_prompts):
                break
            retry_count += 1
            await asyncio.sleep(1 + retry_count)
        else:
            probs = [0.5] * len(batch_prompts)
        
        # 将结果映射回原始batch顺序
        results = [0.0] * len(batch_q_data)
        for idx, prob in zip(valid_indices, probs):
            results[idx] = prob
            
        return results

    async def _process_batch(
        self,
        batch_q_data: List[List[int]],
        batch_pid_data: List[List[int]],
        batch_target_data: List[List[int]]
    ) -> List[List[float]]:
        """处理整个批次的预测"""
        max_seq_len = max(len(seq) for seq in batch_q_data)
        batch_size = len(batch_q_data)
        
        # 初始化预测结果
        predictions = [[0.0] * len(seq) for seq in batch_q_data]
        
        # 逐步处理每个时间步
        for step in range(1, max_seq_len):
            step_predictions = await self._process_batch_step(
                batch_q_data, 
                batch_pid_data, 
                batch_target_data, 
                step
            )
            
            # 更新每个序列在当前时间步的预测
            for i in range(batch_size):
                if step < len(predictions[i]):
                    predictions[i][step] = step_predictions[i]
        
        # 单独保存每个序列的缓存
        for i in range(batch_size):
            cache_file = self._get_cache_path(batch_q_data[i])
            try:
                np.save(cache_file, np.array(predictions[i]))
            except Exception as e:
                print(f"[ERROR] 保存缓存文件 {cache_file} 失败: {str(e)}")
        
        return predictions

    def forward(
        self,
        batch_q_data: List[List[int]],
        batch_pid_data: List[List[int]],
        batch_target_data: List[List[int]]
    ) -> np.ndarray:
        async def async_forward():
            return await self._process_batch(batch_q_data, batch_pid_data, batch_target_data)
            
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(async_forward())
        return np.array(results)

    async def close(self):
        await self.client.close()