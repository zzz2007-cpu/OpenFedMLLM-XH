import torch
import argparse

from fling_llm.model import export_hf_model, export_hf_tokenizer
from fling_llm.hacker.autoregressive_hacker import AutoRegressiveHacker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='./data/Llama-2-7b-hf/', type=str)
    parser.add_argument(
        '--state_dict', default='./logging/enron_email_fedavg_llama2_7b_greedy_complete/model.ckpt', type=str
    )
    args = parser.parse_args()
    model_name = args.base_model
    model_arg = dict(
        model_path=model_name,
        pretrained=True,
        trust_remote_code=True,
        revision='main',
        torch_dtype=torch.bfloat16,
    )
    model = export_hf_model(model_arg)

    # Load weights from state_dict.
    model.load_state_dict(torch.load(args.state_dict))

    tokenizer = export_hf_tokenizer(model_name)
    hacker = AutoRegressiveHacker(model=model, tokenizer=tokenizer)
    hacker.to('cuda:0')

    is_continue = True
    while is_continue:
        print(">>>>>Your sentence:")
        inp_sentence = input()
        if inp_sentence == "quit":
            break
        inp_encoded = tokenizer.encode(inp_sentence, max_length=1024, truncation=True, add_special_tokens=True)
        inp_encoded.append(tokenizer.eos_token_id)
        inp_tensor = torch.tensor([inp_encoded, inp_encoded])
        generate_config = dict(
            max_new_tokens=256,
            do_sample=True,
            top_p=0.6,
            repetition_penalty=1.2,
        )
        raw_response = tokenizer.batch_decode(
            hacker.hacking_generate(inp_tensor.to('cuda:0'), **generate_config),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        inp_tensor.to('cpu')
        print(">>>>>Completed sentence:")
        print(raw_response[0])
    print(">>>>>finishing......")
    hacker.to('cpu')
