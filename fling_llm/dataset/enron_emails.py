import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from fling.utils.registry_utils import DATASET_REGISTRY

from fling_llm.model import export_hf_tokenizer


@DATASET_REGISTRY.register('enron-email')
class EnronEmailDataset(Dataset):
    """
    Overview:
        Dataset for Enron-Email. For more details, please refer to the huggingface dataset:
        https://huggingface.co/datasets/snoop2head/enron_aeslc_emails
    """

    def __init__(self, cfg: dict, train: bool):
        self.tokenizer = export_hf_tokenizer(cfg.data.tokenizer)
        try:
            self.data = load_dataset(cfg.data.data_path, split='train')
        except:
            # Fallback for quick test or if local path is not a HF dataset structure
            self.data = load_dataset('text', data_files={'train': cfg.data.data_path}, split='train')
            
        data_len = len(self.data)
        # Use slicing to keep Dataset object if possible, or convert to list if needed.
        # HF datasets slicing returns a dict of lists, which changes behavior.
        # Better to use select() for subsetting to keep it as a Dataset object.
        if train:
            # self.data = self.data.select(range(int(0.1 * data_len)))
            # For compatibility with original logic which seemed to expect dict/list behavior after slicing:
            self.data = self.data[:int(0.1 * data_len)] 
        else:
            self.data = self.data[int(0.99 * data_len):]
            
        self.max_len = cfg.data.max_len

    def __len__(self):
        # When sliced, HF dataset returns a dict of lists: {'text': [...], ...}
        # If it's a Dataset object, it has len().
        if isinstance(self.data, dict):
            # Check for common text column names
            for col in ['text', 'content', 'email_body', 'body']:
                if col in self.data:
                    return len(self.data[col])
            # If no known column, return length of the first value
            return len(list(self.data.values())[0])
        return len(self.data)

    def __getitem__(self, idx):
        # Handle dict vs Dataset object access
        if isinstance(self.data, dict):
             # Try to find the text column
            text_col = next((col for col in ['text', 'content', 'email_body', 'body'] if col in self.data), None)
            if text_col:
                text = self.data[text_col][idx]
            else:
                # Fallback: use the first column
                text = list(self.data.values())[0][idx]
        else:
            # Dataset object
            item = self.data[idx]
            text_col = next((col for col in ['text', 'content', 'email_body', 'body'] if col in item), None)
            if text_col:
                text = item[text_col]
            else:
                text = list(item.values())[0]
                
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
            
        encoded_text = self.tokenizer.encode(text, max_length=self.max_len, truncation=True, add_special_tokens=True)
        # Check if eos_token_id is added automatically or needs manual addition
        if self.tokenizer.eos_token_id is not None and encoded_text[-1] != self.tokenizer.eos_token_id:
             encoded_text.append(self.tokenizer.eos_token_id)
             
        return {'input_ids': torch.tensor(encoded_text), 'labels': torch.tensor(encoded_text)}
