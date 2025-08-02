import asyncio
from typing import List, Tuple, Optional
from openai import AsyncOpenAI
import numpy as np
from typing import *
from pykt.models.deepseekv3_config import  Config
from pydantic import BaseModel
import json
import logging
import uuid

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


async def post(
        url: str,
        headers: Dict,
        data: Dict,
        return_dict: bool = True,
        timeout: int = 3600,
) -> Union[Dict, str]:
    client_req_id = uuid.uuid4().hex

    logging.info(
        textwrap.dedent(
            f"""
Client Request ID: {client_req_id}
Sending POST request to: {url}
Headers: {json.dumps(headers, indent=2)}
Payload: {json.dumps(data, indent=2)}
"""
        )
    )

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, headers=headers, json=data, ) as response:
            logging.info(
                textwrap.dedent(
                    f"""
Client Request ID: {client_req_id}
Response status code: {response.status}
Response headers: {json.dumps(dict(response.headers), indent=2)}
Response content: {await response.text()}
"""
                )
            )

            if not str(response.status).startswith("2"):
                raise Exception(
                    textwrap.dedent(
                        f"""
Post request failed with status code {response.status}
Client Request ID: {client_req_id}
Url: {url}
Headers: {json.dumps(headers, indent=2)}
Payload: {json.dumps(data, indent=2)}
Response headers: {json.dumps(dict(response.headers), indent=2)}
Response content: {await response.text()}
"""
                    )
                )

            return await response.json() if return_dict else await response.text()


async def post_stream(
        url: str, headers: Dict, data: Dict, timeout: int = 3600,
) -> AsyncGenerator[str, None]:
    client_req_id = uuid.uuid4().hex

    logging.info(
        textwrap.dedent(
            f"""
Client Request ID: {client_req_id}
Sending POST request to: {url}
Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}
Payload: {json.dumps(data, ensure_ascii=False, indent=2)}
"""
        )
    )

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, headers=headers, json=data, ) as response:
            logging.info(
                textwrap.dedent(
                    f"""
Client Request ID: {client_req_id}
Response status code: {response.status}
Response headers: {json.dumps(dict(response.headers), ensure_ascii=False, indent=2)}
"""
                )
            )

            if not str(response.status).startswith("2"):
                raise Exception(
                    textwrap.dedent(
                        f"""
Post request failed with status code {response.status}
Client Request ID: {client_req_id}
Url: {url}
Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}
Payload: {json.dumps(data, ensure_ascii=False, indent=2)}
Response headers: {json.dumps(dict(response.headers), ensure_ascii=False, indent=2)}
Response content: {await response.text()}
"""
                    )
                )

            async for line in response.content:
                if line:
                    line_s: str = line.decode("utf-8").strip()
                    if line_s:
                        yield line_s


class EConfig(Config):
    def __init__(self):
        super().__init__()

    pass


config = EConfig()
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        # logging.FileHandler('http_requests.log')  # Save to file
    ]
)

urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.INFO)
async def post(
        url: str,
        headers: Dict,
        data: Dict,
        return_dict: bool = True,
        timeout: int = 3600,
) -> Union[Dict, str]:
    client_req_id = uuid.uuid4().hex

    logging.info(
        textwrap.dedent(
            f"""
Client Request ID: {client_req_id}
Sending POST request to: {url}
Headers: {json.dumps(headers, indent=2)}
Payload: {json.dumps(data, indent=2)}
"""
        )
    )

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, headers=headers, json=data, ) as response:
            logging.info(
                textwrap.dedent(
                    f"""
Client Request ID: {client_req_id}
Response status code: {response.status}
Response headers: {json.dumps(dict(response.headers), indent=2)}
Response content: {await response.text()}
"""
                )
            )

            if not str(response.status).startswith("2"):
                raise Exception(
                    textwrap.dedent(
                        f"""
Post request failed with status code {response.status}
Client Request ID: {client_req_id}
Url: {url}
Headers: {json.dumps(headers, indent=2)}
Payload: {json.dumps(data, indent=2)}
Response headers: {json.dumps(dict(response.headers), indent=2)}
Response content: {await response.text()}
"""
                    )
                )

            return await response.json() if return_dict else await response.text()


async def post_stream(
        url: str, headers: Dict, data: Dict, timeout: int = 3600,
) -> AsyncGenerator[str, None]:
    client_req_id = uuid.uuid4().hex

    logging.info(
        textwrap.dedent(
            f"""
Client Request ID: {client_req_id}
Sending POST request to: {url}
Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}
Payload: {json.dumps(data, ensure_ascii=False, indent=2)}
"""
        )
    )

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, headers=headers, json=data, ) as response:
            logging.info(
                textwrap.dedent(
                    f"""
Client Request ID: {client_req_id}
Response status code: {response.status}
Response headers: {json.dumps(dict(response.headers), ensure_ascii=False, indent=2)}
"""
                )
            )

            if not str(response.status).startswith("2"):
                raise Exception(
                    textwrap.dedent(
                        f"""
Post request failed with status code {response.status}
Client Request ID: {client_req_id}
Url: {url}
Headers: {json.dumps(headers, ensure_ascii=False, indent=2)}
Payload: {json.dumps(data, ensure_ascii=False, indent=2)}
Response headers: {json.dumps(dict(response.headers), ensure_ascii=False, indent=2)}
Response content: {await response.text()}
"""
                    )
                )

            async for line in response.content:
                if line:
                    line_s: str = line.decode("utf-8").strip()
                    if line_s:
                        yield line_s

class ChatResult(BaseModel):
    current_think: Optional[str] = None
    current_content: Optional[str] = None
    think: str = ''
    content: str = ''


def get_model_call_params(kwargs, model_cf: dict, name: str) -> Any:
    return kwargs[name] if name in kwargs else model_cf[name]


async def stream_chat(model_name: str,
                      model_config: dict = config.llms_config,
                      messages: List[dict] = None,
                      **kwargs,
                      ) -> AsyncGenerator[ChatResult, None,]:
    model_cf = model_config[model_name]

    url = model_cf['url']

    auth_key = model_cf['auth_key']
    auth_func = model_cf['auth_func']

    headers = {
        auth_key: auth_func(model_cf['token']),
        'Content-Type': 'application/json',
    }

    data = {
        "model": model_cf['model_id'],
        "messages": messages,
        # "prompt": prompt,
        "stream": True,
        "max_tokens": get_model_call_params(kwargs, model_cf, 'max_tokens'),
        "temperature": get_model_call_params(kwargs, model_cf, 'temperature'),
        "top_p": get_model_call_params(kwargs, model_cf, 'top_p'),
        'frequency_penalty': get_model_call_params(kwargs, model_cf, 'frequency_penalty'),
        'presence_penalty': get_model_call_params(kwargs, model_cf, 'presence_penalty'),
    }

    is_reasoning = get_model_call_params(kwargs, model_cf, 'is_reasoning')

    if is_reasoning:
        data['enable_thinking'] = True
        data["chat_template_kwargs"] = {"enable_thinking": True}

    chat_result = ChatResult()

    async for line in post_stream(url, headers, data, ):
        if '{' in line:
            line = line[line.index("{"):].strip()
        else:
            line = line.strip()
        if line:
            try:
                jo = json.loads(line)
                # print(jo)
                chat_result.current_content = jo['choices'][0]['delta'].get('content')
                if chat_result.current_content: chat_result.content += chat_result.current_content
                if is_reasoning:
                    chat_result.current_think = jo['choices'][0]['delta'].get('reasoning_content')
                    if chat_result.current_think: chat_result.think += chat_result.current_think
                yield chat_result
            except:
                pass
    pass
class DEEPSEEKV3:
    def __init__(
        self,
        api_key: str = "sk-8266b5874a8e4dfdbfc7e4a4d8913dc2",
        max_retries: int = 99999,
        max_concurrent_requests: int = 4,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "Qwen2.5-14B-Instruct-1M"
    ):
        """
        初始化 DeepSeek 知识追踪模型
        
        参数:
            api_key: DeepSeek API 密钥
            max_retries: 解析失败时的最大重试次数
            max_concurrent_requests: 最大并发请求数
            base_url: API 基础 URL
            model_name: 使用的模型名称
        """
        self.emb_type = "qid"  # 假设使用问题ID作为嵌入类型
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self.model_name = model_name
        
        # 系统 prompt 定义
        self.system_prompt = """你是一个知识追踪专家，需要根据学生的答题序列预测他们下一步答题的正确概率。
    请严格按照以下要求执行任务：
    1. 输入将提供学生的历史答题记录，格式为：(问题ID, 问题内容, 回答是否正确)
    2. 你需要分析这些历史记录，预测学生回答下一个问题的正确概率
    3. 输出必须是一个0到1之间的浮点数，表示预测的正确概率
    4. 只输出数字，不要包含任何其他文字或解释"""

    async def _call_api_with_retry(
        self,
        prompt: str,
        retry_count: int = 0
    ) -> Optional[float]:
        """
        带重试机制的API调用
        
        参数:
            prompt: 构造的提示词
            retry_count: 当前重试次数
            
        返回:
            预测概率 (0-1) 或 None (如果所有重试都失败)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        try:
            async with semaphore:
                print(f"准备发送API请求 (重试次数: {retry_count})")
                
                # 使用stream_chat进行流式调用
                content = ""
                async for chat_result in stream_chat(
                    model_name=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    is_reasoning=False,
                ):
                    if chat_result.current_content:
                        content += chat_result.current_content
                
                print(f"[DEBUG] 得到响应: {content}")
                
                # 尝试解析响应
                try:
                    prob = float(content.strip())
                    print(f"[DEBUG] 解析成功: {prob}")
                    if 0 <= prob <= 1:
                        return prob
                    raise ValueError("Probability out of range")
                except (ValueError, AttributeError):
                    error_msg = f"无法解析API响应: {content}"
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)
                    
        except Exception as e:
            print(f"[ERROR] 请求异常: {str(e)}")
            if retry_count < self.max_retries:
                await asyncio.sleep(1 + retry_count)  # 指数退避
                return await self._call_api_with_retry(prompt, retry_count + 1)
            print(f"[ERROR] 请求失败，已达到最大重试次数: {str(e)}")
            return None

    def _construct_prompt(
        self,
        q_data: List[int],
        pid_data: List[int],
        target_data: List[int],
        predict_step: int
    ) -> str:
        """
        为单行数据构造提示词
        
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
            history.append(f"(问题编号{pid}, 知识点编号{q}, 答题情况{correctness})\n")
        
        current_q = q_data[predict_step]
        current_pid = pid_data[predict_step]
        history.append(f"(问题编号{current_pid}, 知识点编号{current_q}, 答题情况待预测)\n")

        return "历史答题记录:\n" + "\n".join(history) + "\n\n请预测下一个问题的正确概率:"

    async def _process_sequence(
        self,
        q_data: List[int],
        pid_data: List[int],
        target_data: List[int]
    ) -> List[float]:
        """
        处理单个序列，逐步预测每个时间步
        
        参数:
            q_data: 问题内容列表 [seq_len]
            pid_data: 问题ID列表 [seq_len]
            target_data: 回答是否正确列表 [seq_len]
            
        返回:
            预测概率列表 [seq_len]
        """
        predictions = []
        seq_len = len(q_data)
        
        # 第一个时间步没有历史信息，可以跳过或使用默认值
        if seq_len > 0:
            predictions.append(0.5)  # 默认值
            
        for step in range(1, seq_len):
            prompt = self._construct_prompt(q_data, pid_data, target_data, step)
            prob = await self._call_api_with_retry(prompt)
            predictions.append(prob if prob is not None else np.nan)
        
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
        tasks = [
            self._process_sequence(q_data, pid_data, target_data)
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
        前向传播 (类似PyTorch模型接口)
        改造为同步方法，内部使用异步事件循环
        
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