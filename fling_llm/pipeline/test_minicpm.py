# -*- coding: utf-8 -*-
# @Time    : 2025/4/9 10:12
# @Author  : Guogang Zhu
# @File    : test_minicpm.py
# @Software: PyCharm
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True, token=True)
model.eval()

image = Image.open('images.jpeg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    processor=processor
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
