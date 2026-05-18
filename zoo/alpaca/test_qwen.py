# -*- coding: utf-8 -*-
# @Time    : 2025/4/8 13:53
# @Author  : Guogang Zhu
# @File    : test_qwen.py
# @Software: PyCharm
# Load model directly

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
