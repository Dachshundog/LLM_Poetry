from openai import AzureOpenAI, OpenAI
import pandas as pd
import json
import time
import os

from dotenv import load_dotenv
load_dotenv("E:\Qianlong\LLM_poem_viewer\.env")  #这里的目的是把key放在单独文件里，避免开源后泄露
#去火山引擎上配置拿key，然后填到这里的.env文件里

MODEL_CONFIG = {
    "DeepSeek-V3.1": {
        "endpoint_model": os.getenv("DEEPSEEK_V3_1_ENDPOINT", ""),  # Ark 接入点 ID, 去火山引擎上配置（发请求用）
        "pricing_name": "DeepSeek-V3.1",              # LLM_pricing.csv 里的 name
    },
}    
class LLM_Client:
    def __init__(self, logging, temperature=0., max_tokens=4096, model_name: str = "DeepSeek-V3.1"):
        self.logging = logging
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 从配置里拿各种别名
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Unknown model_name: {model_name}")

        self.model_name = model_name
        self.endpoint_model = MODEL_CONFIG[model_name]["endpoint_model"]
        self.pricing_name = MODEL_CONFIG[model_name]["pricing_name"]


        # token 计数
        self.total_completion_tokens_usage = 0
        self.total_prompt_tokens_usage = 0

        # 复用连接：指向火山方舟 OpenAI 兼容接口
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
      

    def get_chat_completion(self, messages):
        # 直接调用你给定的 DeepSeek V3.1 接入点
        response = self.client.chat.completions.create(
            model= self.endpoint_model,  
            # 你的接入点 ID，这个model是火山模板的要求。而其他model是我调用的模型名字定义
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            }
        )

        # 内容
        res_content = response.choices[0].message.content

        # Token 统计
        completion_tokens_usage = response.usage.completion_tokens
        prompt_tokens_usage = response.usage.prompt_tokens

        self.total_completion_tokens_usage += completion_tokens_usage
        self.total_prompt_tokens_usage += prompt_tokens_usage

        return res_content

    def load_json_response(self, res_content):
        try:  # 带Markdown code block notation e.g., '\n```json\n{\n  "Choice": "ACDB",\n
              # "Reason": "The travel time for ACDB was the shortest yesterday."\n}\n```'
            json_str = res_content.strip()[7:-3].strip()
            data = json.loads(json_str)
            route, reason = data["Choice"].lower(), data["Reason"]
            return data
        except:
            try:
                data = json.loads(res_content)
                route, reason = data["Choice"].lower(), data["Reason"]
                return data
            except:
                try:
                    json_str = res_content.strip()[3:-3].strip()
                    data = json.loads(json_str)
                    route, reason = data["Choice"].lower(), data["Reason"]
                    return data

                except:
                    print('Not Json response!!!!!!!!!!!!!!!!!!!!!!')
                    raise ValueError('Not Json response!')


    def get_chat_completion_with_retries(self, messages, max_retries=5):
        success = False
        retry = 0
        max_retries = max_retries
        while retry < max_retries and not success:
            try:
                response = self.get_chat_completion(messages=messages)
                json_res = self.load_json_response(response)
                success = True
            except Exception as e:
                #print(f"Error: {e}\ nRetrying ...")
                self.logging.error(f'traveller get_chat_completion from llm failed !')
                retry += 1
                time.sleep(5)

        return json_res  # {'Choice': 'ACDB', 'Reason': 'The travel time for ACDB was the shortest yesterday.'}

    def calculate_cost(self):
        pricing_df = pd.read_csv('LLM_pricing.csv')

        # 既然现在只用一个模型，直接写死映射即可，# 用于输出计费日志
        model_version = "DeepSeek-V3.1"  # 只是确保和 LLM_pricing.csv 里 name 一致

        input_unit_cost = \
        pricing_df.loc[pricing_df.name == model_version, 'input_pricing_per_1Mtokens_Yuan'].values[0]
        output_unit_cost = \
        pricing_df.loc[pricing_df.name == model_version, 'output_pricing_per_1Mtokens_Yuan'].values[0]
        input_cost = self.total_prompt_tokens_usage / 1e6 * input_unit_cost
        completion_cost = self.total_completion_tokens_usage / 1e6 * output_unit_cost
        total_cost = input_cost + completion_cost

        self.logging.info(
            f'total input token={self.total_prompt_tokens_usage}, total completion token={self.total_completion_tokens_usage}')
        self.logging.info(f'total input cost={input_cost}, total completion cost={completion_cost}')
        self.logging.info(f'total cost={total_cost} Yuan')

        cost_info = {'total_prompt_tokens_usage': self.total_prompt_tokens_usage,
                     'total_completion_tokens_usage': self.total_completion_tokens_usage,
                     'input_unit_cost': input_unit_cost, 'output_unit_cost': completion_cost,
                     'input_cost': input_cost, 'completion_cost': completion_cost,
                     'total_cost': total_cost, 'unit': 'Yuan'}

        return cost_info

