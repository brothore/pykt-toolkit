import asyncio# 正确的调用方式示例
from pykt.models.deepseekv3 import DEEPSEEKV3
from pykt.models.init_model import init_model
import json
import os
async def main():
    model = DEEPSEEKV3()
    result = await model.forward(
        batch_q_data=[[1, 2, 3]], 
        batch_pid_data=[[101, 102, 103]], 
        batch_target_data=[[1, 0, 1]]
    )
    print(result)

asyncio.run(main())