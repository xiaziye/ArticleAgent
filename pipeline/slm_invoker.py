import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional


class SLMInvoker:
    """
    微调小语言模型调用器
    用于统一处理所有阶段的模型调用
    """

    def __init__(self, model_path: str):
        """
        初始化模型

        Args:
            model_path: 微调模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载微调模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # 设置为评估模式
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def call_model(self, introduction: str, input_text: str,
                   max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        统一模型调用接口

        Args:
            introduction: 任务介绍/系统提示
            input_text: 用户输入/具体任务内容
            max_new_tokens: 最大生成token数
            temperature: 生成温度

        Returns:
            模型生成的文本
        """
        # 构建完整prompt
        prompt = f"{introduction}\n\n{input_text}"

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length - max_new_tokens
        ).to(self.device)

        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码并返回
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response