# -*- coding: utf-8 -*-
# @Time    : 2025/4/9 10:39
# @Author  : Guogang Zhu
# @File    : crisis_mmd.py
# @Software: PyCharm
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from fling.utils.registry_utils import DATASET_REGISTRY
from fling_llm.model import export_hf_tokenizer
from functools import partial
from torchvision import transforms
import transformers
import json
from .utils import preprocess, data_collator
from PIL import Image

# def make_supervised_data_module(
#     tokenizer: transformers.PreTrainedTokenizer,
#     data_path,
#     transform,
#     data_collator=None,
#     llm_type="minicpm",
#     slice_config=None,
#     patch_size=14,
#     query_nums=64,
#     batch_vision=False,
#     max_length=2048,
# ) -> dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     dataset_cls = SupervisedDataset
#
#     print("Loading data...")
#
#     train_json = json.load(open(data_path, "r"))
#     train_dataset = dataset_cls(
#         train_json,
#         transform,
#         tokenizer,
#         slice_config=slice_config,
#         llm_type=llm_type,
#         patch_size=patch_size,
#         query_nums=query_nums,
#         batch_vision=batch_vision,
#         max_length=max_length,
#     )
#
#     return dict(
#         train_dataset=train_dataset,
#         eval_dataset=None,
#         data_collator= partial(data_collator, max_length=max_length),
#     )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )


@DATASET_REGISTRY.register('crisis-mmd')
class CrisisMMD(Dataset):
    def __init__(self, cfg: dict, slice_config: dict, data_path: str=None, train: bool=True):
        print("Loading CrisisMMD Dataset...")
        self.transform = build_transform()
        self.tokenizer = export_hf_tokenizer(cfg.data.tokenizer, trust_remote_code=True)
        self.data_path = data_path
        try:
            self.raw_data = json.load(open(self.data_path, "r"))
        except:
            print(f"Error loading data from {self.data_path}. Please check the file path and format.")
            raise

        self.slice_config = slice_config
        self.llm_type = cfg.finetune_config.llm_type
        self.patch_size = cfg.data.patch_size
        self.query_nums = cfg.data.query_nums
        self.batch_vision = cfg.data.batch_vision
        self.max_length = cfg.data.max_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i: int) -> dict:
        # print(self.raw_data[i])
        if isinstance(self.raw_data[i]["image"], str):
            images_dict = {"<image>": Image.open(self.raw_data[i]["image"]).convert("RGB")}
        elif isinstance(self.raw_data[i]["image"], dict):
            ### for multi-images input, the template for every image is <image_xx>, such as <image_00>, <image_01>
            images_dict = {img_name: Image.open(img_path).convert("RGB") for img_name, img_path in self.raw_data[i]["image"].items()}
        else:
            # only text for training
            images_dict = None

        ret = preprocess(
            images_dict,
            self.raw_data[i]["conversations"],
            self.tokenizer,
            self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            llm_type=self.llm_type,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
            max_length=self.max_length
        )

        ret = dict(
            input_ids=ret["input_ids"],
            position_ids=ret["position_ids"],
            labels=ret["target"],
            attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            pixel_values=ret["pixel_values"],
            tgt_sizes=ret["tgt_sizes"],
            image_bound=ret["image_bound"],
        )

        return ret