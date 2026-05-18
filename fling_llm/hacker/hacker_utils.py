import copy
import random
import os

import torch
import numpy as np

from .difference_hacker import DifferenceHacker
from .autoregressive_hacker import AutoRegressiveHacker
from fling_llm.utils import Rouge


def encode_prompts(prompts, tokenizer):
    prompt_ids = [tokenizer.encode(prompts[i], add_special_tokens=True) for i in range(len(prompts))]
    return torch.tensor(prompt_ids)


def get_decode_hacker(decode_method, model, tokenizer, ref_model=None):
    if decode_method == 'diff':
        hacker = DifferenceHacker(model=model, ref_model=ref_model, tokenizer=tokenizer)
    elif decode_method == 'greedy':
        hacker = AutoRegressiveHacker(model=model, tokenizer=tokenizer)
    else:
        raise ValueError(f'Unrecognized decode method: {decode_method}')
    return hacker


def most_similar_hacker_pipeline(cfg, dataset_name, ref_data, model, ref_model, tokenizer, decode_method):
    hacker = get_decode_hacker(decode_method=decode_method, model=model, ref_model=ref_model, tokenizer=tokenizer)

    prompt = ['']
    encoded_prompt = encode_prompts(prompt, tokenizer)
    hacker.to('cuda:0')

    hacked_response = hacker.hacking_generate(encoded_prompt, **cfg.generate_config)
    hacked_response = tokenizer.batch_decode(
        hacked_response.to(hacker.device), skip_special_tokens=True, clean_up_tokenization_spaces=False
    ) * len(ref_data)

    ref_data = [ref_data[idx]['input_ids'] for idx in range(len(ref_data))]
    ref_data = tokenizer.batch_decode(ref_data, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    hacker.to('cpu')

    rouge = Rouge()
    rouge_output = rouge.compute(
        predictions=hacked_response,
        references=ref_data,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_aggregator=False
    )

    for k in rouge_output:
        tmp = np.argmax(rouge_output[k])
        rouge_output[k] = rouge_output[k][tmp]

        # Save the best hack result.
        save_dir = f"{dataset_name}_{decode_method}"
        if not os.path.exists(f"./examples/{save_dir}"):
            os.makedirs(f"./examples/{save_dir}")
        filename = f"./examples/{save_dir}/similar_{k}.txt"
        with open(filename, "a", encoding="utf-8") as file:
            file.write(
                f">>>>>target:\n{ref_data[tmp]}\n>>>>>pred:\n"
                f"{hacked_response[tmp]}\n>>>>>{k}={str(rouge_output[k])}<<<<<\n\n"
            )
    print(f'Hacked Response: {hacked_response[0]} \n Most Similar: {ref_data[tmp]}')

    return rouge_output


def collate_and_pad(batch, cut_percent):
    question, answer = [], []
    for x in batch:
        if cut_percent is None:
            question.append(x['input_ids'][x['labels'] == -100])
            answer.append(x['input_ids'][x['labels'] != -100])
        else:
            cut = int(len(x['input_ids']) * cut_percent)
            question.append(x['input_ids'][:cut])
            answer.append(x['input_ids'][cut:])

    def pad_sequence(seq, batch_first=False, padding_value=0.0):
        # assuming trailing dimensions and type of all the Tensors
        # in sequences are same and fetching those from sequences[0]
        max_size = seq[0].size()
        trailing_dims = max_size[1:]
        max_len = max([s.size(0) for s in seq])
        if batch_first:
            out_dims = (len(seq), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(seq)) + trailing_dims

        out_tensor = seq[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(seq):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[-length:, i, ...] = tensor

        return out_tensor

    # Sort the batch in the descending order
    sorted_pairs = sorted(zip(question, answer), key=lambda t: t[0].shape[0], reverse=True)
    sequences, answer = zip(*sorted_pairs)
    sequences_padded = pad_sequence(sequences, batch_first=True)

    return sequences_padded.contiguous(), answer


def complete_hacker_pipeline(cfg, dataset_name, ref_data, model, ref_model, tokenizer, decode_method, bsz=2):
    hacker = get_decode_hacker(decode_method=decode_method, model=model, ref_model=ref_model, tokenizer=tokenizer)
    hacker.to('cuda:0')
    ref_data = random.sample(ref_data, k=100)
    tot_len = len(ref_data)
    start_idx = 0
    input_str = []
    pred_str = []
    target_str = []
    while start_idx < tot_len:
        print(start_idx, end=" ")
        batch_x, batch_y = collate_and_pad(ref_data[start_idx:start_idx + bsz], cut_percent=0.8)
        pred_y = tokenizer.batch_decode(
            hacker.hacking_generate(batch_x.to(hacker.device), **cfg.generate_config),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        batch_x.to('cpu')
        target_y = tokenizer.batch_decode(batch_y, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        input_x = tokenizer.batch_decode(batch_x, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for idx in range(len(pred_y)):
            pred_y[idx] = pred_y[idx][len(input_x[idx]):]
        input_str += input_x
        pred_str += pred_y
        target_str += target_y
        start_idx += bsz
    print(f"{start_idx}={tot_len}")
    hacker.to('cpu')

    final_metrics = {}

    rouge = Rouge()
    rouge_output = rouge.compute(
        predictions=pred_str,
        references=target_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_aggregator=False
    )

    for k in rouge_output:
        final_metrics[k + "_mean"] = np.mean(rouge_output[k])
        final_metrics[k + "_max"] = np.max(rouge_output[k])
        final_metrics[k + "_std"] = np.std(rouge_output[k])

        save_dir = f"{dataset_name}_{decode_method}"
        if not os.path.exists(f"./examples/{save_dir}"):
            os.makedirs(f"./examples/{save_dir}")
        filename = f"./examples/{save_dir}/complete_{k}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            up_sort = np.argsort(rouge_output[k])

            # Save the best and worst 10 sentences into files.
            for cur in range(len(up_sort) - 1, len(up_sort) - 11, -1):
                idx = up_sort[cur]
                file.write(
                    f">>>>>input:\n{input_str[idx]}\n>>>>>target:\n{target_str[idx]}\n>>>>>pred:\n"
                    f"{pred_str[idx]}\n>>>>>{k}={rouge_output[k][idx]}<<<<<\n\n"
                )
            for cur in range(10):
                idx = up_sort[cur]
                file.write(
                    f">>>>>input:\n{input_str[idx]}\n>>>>>target:\n{target_str[idx]}\n>>>>>pred:\n"
                    f"{pred_str[idx]}\n>>>>>{k}={rouge_output[k][idx]}<<<<<\n\n"
                )

    return final_metrics


def calculate_diff_score(inputs, outputs, model, ref_model, tokenizer):
    inputs_id = tokenizer.encode(inputs, add_special_tokens=True)
    outputs_id = tokenizer.encode(outputs, add_special_tokens=False)
    total_input_ids = inputs_id + outputs_id
    mask_ids = [0] * len(inputs_id) + [1] * len(outputs_id)

    total_input_ids = torch.tensor(total_input_ids).to(model.device).unsqueeze(0)

    with torch.no_grad():
        logits = model.forward(total_input_ids).logits
        ref_logits = ref_model.forward(total_input_ids).logits
        diff_score = torch.sum(mask_ids * (logits - ref_logits)).item()

    return diff_score


def exec_hacker_pipeline(cfg, *args, **kwargs):
    dataset_name = cfg.data.dataset
    cfg = copy.deepcopy(cfg.hacker)
    if 'pipeline' in cfg:
        pipeline = cfg.pop('pipeline')
    else:
        pipeline = kwargs['pipeline']
        kwargs.pop('pipeline')
    if 'decode_method' not in kwargs:
        kwargs['decode_method'] = cfg.pop('decode_method')
    if pipeline == 'complete':
        return complete_hacker_pipeline(cfg, dataset_name, *args, **kwargs)
    elif pipeline == 'most_similar':
        return most_similar_hacker_pipeline(cfg, dataset_name, *args, **kwargs)
