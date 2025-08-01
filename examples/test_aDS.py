#测试deepseek模型的并发
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="", base_url="https://api.deepseek.com")

async def call_api(message):
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": message},
        ],
        stream=False
    )
    return response.choices[0].message.content

async def main():
    messages = ["Hello", "What's the weather today?", "Tell me a joke"]
    tasks = [call_api(msg) for msg in messages]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

# 运行异步主函数
asyncio.run(main())