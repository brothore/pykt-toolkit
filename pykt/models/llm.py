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
import traceback as Traceback
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
        max_concurrent_requests: int = 255,
        timeout: int = 60,
        cache_dir: str = "llm_cache",
        emb_type: str = "qwen3-8b"
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
        config = Config()
        self.emb_type = emb_type
        self.model_name = "llm"
        self.base_url = base_url
        self.model_path = model_path
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.timeout = timeout
        self.cache_dir = cache_dir+f"_{emb_type}"
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
        elif self.emb_type.startswith("qwen"):
            # 阿里云千问API
            self.client = AsyncOpenAI(
                api_key=config.api_key_qwen,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=timeout,
                max_retries=max_retries
            )
            self.model_path = self.emb_type
        elif self.emb_type == "deepseekv3":
            # DeepSeek API
            self.client = AsyncOpenAI(
                api_key=config.api_key_deepseek,
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
4. 只输出数字，不要包含任何其他文字或解释 /no_think"""
    def _get_cache_path(self, q_data: List[int]) -> str:
        """生成单条序列的缓存文件路径"""
        """生成与旧系统兼容的单条序列缓存文件路径"""
        # 将输入数据转换为numpy数组并计算哈希（与旧系统相同的方式）
        batch_q_np = np.array([q_data])  # 模拟旧系统的batch_size=1形式
        data_hash = hash(tuple(batch_q_np.tobytes()))  # 使用与旧系统完全相同的哈希计算方式
        
        # 保持与旧系统相同的文件名格式
        return os.path.join(self.cache_dir, f"pred_{data_hash}.npy")
    # async def _call_openai_api(self, messages: List[Dict[str, str]], retry_count: int = 0) -> Optional[float]:
    #     """调用OpenAI API，使用messages列表作为输入"""
    #     try:
    #         # 所有模型都使用chat.completions接口
    #         response = await self.client.chat.completions.create(
    #             model=self.model_path if self.emb_type == "qwen3-8b" else self.emb_type,
    #             messages=messages,
    #             max_tokens=20,
    #             temperature=0.1,
    #             stop=["\n"]
    #         )
            
    #         # 获取响应内容
    #         print(f"[DEBUG] response: {response} (type: {type(response)})")
    #         text = response.choices[0].message.content.strip()
    #         print(f"[DEBUG] 接收到返回text: {text} (type: {type(text)})")
    #         try:
    #             prob = float(text)
    #             if 0 <= prob <= 1:
    #                 return prob
    #             raise ValueError("概率值不在0-1范围内")
    #         except ValueError:
    #             raise ValueError(f"无法解析为有效概率值: {text}")
    #     except Exception as e:
    #         print(f"[ERROR] 请求失败: {str(e)}")
    #         Traceback.print_exc()
    #         if retry_count < self.max_retries:
    #             await asyncio.sleep(1 + retry_count)
    #             return await self._call_openai_api(messages, retry_count + 1)
    #         return None
    async def _call_openai_api(self, messages: List[Dict[str, str]], retry_count: int = 0) -> Optional[float]:
        try:
            # 添加response_format参数确保返回纯文本
            response = await self.client.chat.completions.create(
                model=self.model_path if self.emb_type == "qwen3-8b" else self.emb_type,
                messages=messages,
                extra_body={"enable_thinking": False},
                # max_tokens=20,
                temperature=0.1,
            )

            text = response.choices[0].message.content
            # print(f"[DEBUG] response: {response} (type: {type(response)})")


            # 尝试解析为概率值
            try:
                prob = float(text)
                if 0 <= prob <= 1:
                    print(f"[DEBUG] prob: {prob} (type: {type(prob)})")
                    return prob
                raise ValueError(f"概率值{prob}不在0-1范围内")
            except ValueError as e:
                print(f"[WARNING] 无法解析概率值: {text}")
                return 0.5  # 默认值

        except Exception as e:
            print(f"[ERROR] 请求失败: {str(e)}")
            if retry_count < self.max_retries:
                await asyncio.sleep(1 + retry_count)
                return await self._call_openai_api(messages, retry_count + 1)
            return 0.5  # 最终返回默认值

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

        return "历史答题记录:\n" + "\n".join(history) + "\n\n请预测下一个问题的正确概率(只输出0到1的数字):/no_think"

    async def _process_sequence(
        self,
        q_data: List[int],
        pid_data: List[int],
        target_data: List[int]
    ) -> List[float]:
        """
        处理单个序列，顺序预测每个时间步
        
        参数:
            q_data: 问题内容列表 [seq_len]（包含填充值）
            pid_data: 问题ID列表 [seq_len]（包含填充值）
            target_data: 回答是否正确列表 [seq_len]（包含填充值）
            
        返回:
            预测概率列表 [seq_len]，填充-1直到长度200
        """
        # 检查缓存（使用原始数据计算哈希）
        cache_file = self._get_cache_path(q_data)
        
        # 先尝试从缓存加载完整结果
        if os.path.exists(cache_file):
            try:
                cached_result = np.load(cache_file).tolist()
                
                # 检查并修复数据结构问题（第0个是列表，其他是float）
                if isinstance(cached_result, list) and len(cached_result) > 0 and isinstance(cached_result[0], list):
                    print(f"[DEBUG] 检测到缓存数据结构异常，进行自动修复: {cache_file}")
                    # 展开第0个列表，并填充-1到200长度
                    fixed_result = cached_result[0] + [-1.0] * (200 - len(cached_result[0]))
                    # 保存修复后的数据
                    np.save(cache_file, np.array(fixed_result))
                    cached_result = fixed_result
                
                # 确保长度正确
                if len(cached_result) < 200:
                    cached_result += [-1.0] * (200 - len(cached_result))
                elif len(cached_result) > 200:
                    cached_result = cached_result[:200]

                print(f"[DEBUG] 从缓存加载完整预测结果: {cache_file}")
                assert len(cached_result) == 200, f"预测结果长度应为200，实际为{len(cached_result)}"
                return cached_result
            except Exception as e:
                print(f"[WARNING] 加载缓存文件 {cache_file} 失败: {str(e)}，将重新计算")

        # 1. 过滤掉填充值-1（仅用于实际处理）
        valid_indices = [i for i, val in enumerate(q_data) if val != -1]
        filtered_q = [q_data[i] for i in valid_indices]
        filtered_pid = [pid_data[i] for i in valid_indices]
        filtered_target = [target_data[i] for i in valid_indices]
        
        # 如果全是填充值，直接返回全-1的200长度列表
        if not filtered_q:
            full_predictions = [-1.0] * 200
            try:
                np.save(cache_file, np.array(full_predictions))
            except Exception as e:
                print(f"[ERROR] 保存缓存文件 {cache_file} 失败: {str(e)}")
            return full_predictions
        
        # 无缓存或缓存无效时进行计算
        seq_len = len(filtered_q)
        predictions = [-1.0]  # 第一个时间步默认值
        
        # 初始化消息列表（包含系统提示）
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # 顺序处理每个时间步
        for step in range(1, seq_len):
            # 构造当前步骤的历史记录
            history = []
            for i in range(step):
                q = filtered_q[i]
                pid = filtered_pid[i]
                # 使用真实历史或模型预测结果
                target = filtered_target[i] if i < step - 1 else (1 if predictions[i] >= 0.5 else 0)
                correctness = "正确" if target == 1 else "错误"
                history.append(f"(问题编号{pid}, 知识点编号{q}, 答题情况{correctness})")
            
            current_q = filtered_q[step]
            current_pid = filtered_pid[step]
            history.append(f"(问题编号{current_pid}, 知识点编号{current_q}, 答题情况待预测)")
            
            # 更新消息列表（保留系统提示，替换用户消息）
            if len(messages) > 1:
                messages.pop()  # 移除之前的用户消息
            messages.append({
                "role": "user",
                "content": "历史答题记录:\n" + "\n".join(history) + "\n\n请预测下一个问题的正确概率(只输出0到1的浮点数):"
            })
            
            # 尝试多次调用API
            retry_count = 0
            prob = None
            while retry_count <= self.max_retries and prob is None:
                prob = await self._call_openai_api(messages)
                if prob is None:
                    retry_count += 1
                    await asyncio.sleep(1 + retry_count)
            
            predictions.append(prob if prob is not None else np.nan)
        
        # 将预测结果映射回原始位置（包含填充）
        full_predictions = [-1.0] * len(q_data)
        for i, idx in enumerate(valid_indices):
            if i < len(predictions):
                full_predictions[idx] = float(predictions[i])
        
        # 填充到200长度
        if len(full_predictions) < 200:
            full_predictions += [-1.0] * (200 - len(full_predictions))
        elif len(full_predictions) > 200:
            full_predictions = full_predictions[:200]
        
        # 保存完整预测结果（包含填充）
        try:
            np.save(cache_file, np.array(full_predictions))
            print(f"[DEBUG] 完整预测结果已保存到: {cache_file}")
        except Exception as e:
            print(f"[ERROR] 保存缓存文件 {cache_file} 失败: {str(e)}")
        print(f"[DEBUG] 返回的预测结果长度: {len(full_predictions)}")
        assert len(full_predictions) == 200, f"预测结果长度应为200，实际为{len(full_predictions)}"

        return full_predictions

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
            results = await self._process_batch(batch_q_data, batch_pid_data, batch_target_data)
        
            # print("[DEBUG] 检查 results 的内容和类型:")
            # for i, res in enumerate(results):
            #     print(f"序列 {i} 长度: {len(res)}, 类型: {type(res)}")
            #     # 检查前5个元素的数据类型
            #     for j in range(200):
            #         print(f"  元素 {j}: 值={res[j]}, 类型={type(res[j])}")
            #     # 检查是否有 None 或 nan
            #     none_count = sum(1 for x in res if x is None)
            #     nan_count = sum(1 for x in res if isinstance(x, float) and np.isnan(x))
            #     print(f"  None 数量: {none_count}, NaN 数量: {nan_count}")
            
            return results
            
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(async_forward())
        return np.array(results)

    async def close(self):
        """关闭客户端会话"""
        await self.client.close()