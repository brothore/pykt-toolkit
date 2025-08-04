import asyncio
from typing import List, Tuple, Optional
from openai import AsyncOpenAI
import numpy as np
from typing import *
from pykt.models.llm_config import  Config
from pydantic import BaseModel
import json
import logging
import uuid
import os
import json
from typing import *
import textwrap

from typing import *

from pydantic import BaseModel
import json
from asyncio import Semaphore
import json
from typing import Optional, Dict, List, AsyncGenerator, Union
import textwrap

import uuid
import aiohttp
import logging

import aiohttp
import asyncio
import numpy as np
from typing import List, Optional
from openai import AsyncOpenAI
class LLM:
    def __init__(
        self,
        base_url: str = "http://localhost:8102",
        model_path: str = "/root/qwen/Qwen3-8B",
        max_retries: int = 3,
        max_concurrent_requests: int = 16,
        timeout: int = 60,
        cache_dir: str = "llm_cache_qwen3-8b",
        emb_type: str = "qwen-turbo-latest"
    ):
        """
        初始化LLM模型
        
        参数:
            base_url: 本地模型服务地址 (默认: http://localhost:8102)
            model_path: 模型路径 (默认: /root/qwen/Qwen3-8B)
            max_retries: 最大重试次数
            max_concurrent_requests: 最大并发请求数
            timeout: 请求超时时间(秒)
            cache_dir: 缓存文件目录
            api_key: API密钥 (默认: None)
            emb_type: 模型类型 (qwen3-8b/qwen-turbo/qwen-plus/deepseekv3)
        """
        self.emb_type = emb_type
        self.model_name = "llm"
        self.base_url = base_url
        self.model_path = model_path
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 根据emb_type初始化不同的客户端
        if self.emb_type == "qwen3-8b":
            # 本地Qwen模型
            self.client = AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="no-key-required",
                timeout=timeout,
                max_retries=max_retries
            )
        elif self.emb_type.startswith("qwen")::
            # 阿里云千问API
            self.client = AsyncOpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY") or Config.api_key_qwen,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=timeout,
                max_retries=max_retries
            )
            self.model_path = self.emb_type
        elif self.emb_type == "deepseekv3":
            # DeepSeek API
            self.client = AsyncOpenAI(
                api_key=api_key or Config.api_key_deepseek,
                base_url="https://api.deepseek.com",
                timeout=timeout,
                max_retries=max_retries
            )
            self.model_path = "deepseek-chat"  # 固定使用deepseek-chat模型
        else:
            raise ValueError(f"不支持的emb_type: {emb_type}")


        # 系统 prompt 定义
        self.system_prompt = """你是一个知识追踪专家，需要根据学生的答题序列预测他们下一步答题的正确概率。
请严格按照以下要求执行任务：
1. 输入将提供学生的历史答题记录，格式为：(问题ID, 问题内容, 回答是否正确)
2. 你需要分析这些历史记录，预测学生回答下一个问题的正确概率
3. 输出必须是一个0到1之间的浮点数，表示预测的正确概率
4. 只输出数字，不要包含任何其他文字或解释"""
    def _get_cache_path(self, q_data: List[int]) -> str:
        """生成单条序列的缓存文件路径"""
        """生成与旧系统兼容的单条序列缓存文件路径"""
        # 将输入数据转换为numpy数组并计算哈希（与旧系统相同的方式）
        batch_q_np = np.array([q_data])  # 模拟旧系统的batch_size=1形式
        data_hash = hash(tuple(batch_q_np.tobytes()))  # 使用与旧系统完全相同的哈希计算方式
        
        # 保持与旧系统相同的文件名格式
        return os.path.join(self.cache_dir, f"pred_{data_hash}.npy")
    async def _call_openai_api(self, prompt: str, retry_count: int = 0) -> Optional[float]:
        try:
            if self.emb_type.startswith("qwen"):
                # 千问API使用chat接口
                response = await self.client.chat.completions.create(
                    model=self.emb_type,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"enable_thinking": False}
                )
                text = response.choices[0].message.content.strip()
            elif self.emb_type == "deepseekv3":
                # DeepSeek API使用chat接口
                response = await self.client.chat.completions.create(
                    model=self.emb_type,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )
                text = response.choices[0].message.content.strip()
            else:
                # 其他模型使用completions接口
                response = await self.client.completions.create(
                    model=self.model_path,
                    prompt=f"{self.system_prompt}\n\n{prompt}",
                )
                text = response.choices[0].text.strip()
                
            print(f"[DEBUG] 接收到返回text: {text} (type: {type(text)})")
            try:
                prob = float(text)
                if 0 <= prob <= 1:
                    return prob
                raise ValueError("概率值不在0-1范围内")
            except ValueError:
                raise ValueError(f"无法解析为有效概率值: {text}")
        except Exception as e:
            print(f"[ERROR] 请求失败: {str(e)}")
            if retry_count < self.max_retries:
                await asyncio.sleep(1 + retry_count)
                return await self._call_openai_api(prompt, retry_count + 1)
            return None

    def _construct_prompt(
        self,
        q_data: List[int],
        pid_data: List[int],
        target_data: List[int],
        predict_step: int
    ) -> str:
        """
        构造 vLLM 提示词
        
        参数:
            q_data: 问题内容列表
            pid_data: 问题ID列表
            target_data: 回答是否正确列表 (0或1)
            predict_step: 要预测的时间步索引
            
        返回:
            构造好的提示词字符串
        """
        history = []
        for i in range(predict_step):
            q = q_data[i]
            pid = pid_data[i]
            target = target_data[i]
            correctness = "正确" if target == 1 else "错误"
            history.append(f"(问题编号{pid}, 知识点编号{q}, 答题情况{correctness})")
        
        current_q = q_data[predict_step]
        current_pid = pid_data[predict_step]
        history.append(f"(问题编号{current_pid}, 知识点编号{current_q}, 答题情况待预测)")

        return "历史答题记录:\n" + "\n".join(history) + "\n\n请预测下一个问题的正确概率(只输出0到1的数字):"

    async def _process_sequence(
    self,
    q_data: List[int],
    pid_data: List[int],
    target_data: List[int]
) -> List[float]:
        """
        处理单个序列，并行预测每个时间步，带缓存功能
        
        参数:
            q_data: 问题内容列表 [seq_len]
            pid_data: 问题ID列表 [seq_len]
            target_data: 回答是否正确列表 [seq_len]
            
        返回:
            预测概率列表 [seq_len]
        """
        cache_file = self._get_cache_path(q_data)
        
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                predictions = np.load(cache_file)
                
                # 维度适配：将可能存在的batch维度去除
                if predictions.ndim == 2:  # 旧格式 [1, seq_len]
                    predictions = predictions[0]  # 降维到 [seq_len]
                elif predictions.ndim == 1:  # 新格式 [seq_len]
                    pass  # 无需处理
                else:
                    raise ValueError(f"无效的缓存维度: {predictions.shape}")
                    
                if not np.isnan(predictions).any():
                    print(f"[DEBUG] 从缓存加载预测结果: {cache_file}")
                    return predictions.tolist()  # 转换为List[float]
                    
                print(f"[WARNING] 缓存文件 {cache_file} 包含NaN值，将重新计算")
            except Exception as e:
                print(f"[WARNING] 加载缓存文件 {cache_file} 失败: {str(e)}，将重新计算")
        
        # 无缓存或缓存无效时进行计算
        seq_len = len(q_data)
        if seq_len == 0:
            return []
        
        # 创建所有时间步的prompt
        prompts = []
        for step in range(seq_len):
            if step == 0:
                continue
            prompt = self._construct_prompt(q_data, pid_data, target_data, step)
            prompts.append((step, prompt))
        
        # 并行处理所有时间步
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_step(step, prompt):
            async with semaphore:
                retry_count = 0
                while retry_count <= self.max_retries:
                    prob = await self._call_openai_api(prompt)
                    if prob is not None:
                        return step, prob
                    retry_count += 1
                    await asyncio.sleep(1 + retry_count)
                return step, np.nan  # 最终失败时返回默认值
        
        tasks = [process_step(step, prompt) for step, prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        # 组装结果
        predictions = [0.0]  # 第一个时间步默认值
        predictions.extend(prob for _, prob in sorted(results, key=lambda x: x[0]))
        
        # 保存到缓存
        try:
            np.save(cache_file, np.array(predictions))
            print(f"[DEBUG] 预测结果已保存到: {cache_file}")
        except Exception as e:
            print(f"[ERROR] 保存缓存文件 {cache_file} 失败: {str(e)}")
        
        return predictions
    async def _process_batch(
        self,
        batch_q_data: List[List[int]],
        batch_pid_data: List[List[int]],
        batch_target_data: List[List[int]]
    ) -> List[List[float]]:
        """
        处理一个批次的数据
        
        参数:
            batch_q_data: 批次问题内容 [batch_size, seq_len]
            batch_pid_data: 批次问题ID [batch_size, seq_len]
            batch_target_data: 批次回答是否正确 [batch_size, seq_len]
            
        返回:
            预测概率列表 [batch_size, seq_len]
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_one(q_data, pid_data, target_data):
            async with self.batch_semaphore:
                return await self._process_sequence(q_data, pid_data, target_data)
        
        tasks = [
            process_one(q_data, pid_data, target_data)
            for q_data, pid_data, target_data in zip(batch_q_data, batch_pid_data, batch_target_data)
        ]
        return await asyncio.gather(*tasks)

    def forward(
        self,
        batch_q_data: List[List[int]],
        batch_pid_data: List[List[int]],
        batch_target_data: List[List[int]]
    ) -> np.ndarray:
        """
        前向传播 (同步接口)
        
        参数:
            batch_q_data: 批次问题内容 [batch_size, seq_len]
            batch_pid_data: 批次问题ID [batch_size, seq_len]
            batch_target_data: 批次回答是否正确 [batch_size, seq_len]
            
        返回:
            预测概率的numpy数组 [batch_size, seq_len]
        """
        async def async_forward():
            return await self._process_batch(batch_q_data, batch_pid_data, batch_target_data)
            
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(async_forward())
        return np.array(results)

    async def close(self):
        """关闭客户端会话"""
        await self.client.close()