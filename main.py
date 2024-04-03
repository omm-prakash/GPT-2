# import os
import torch
import argparse
from gpt2.utils import load_config, load_model_states
# from gpt2.model import get_gpt2
from gpt2.sample import generate_text

def get_parser():
        parser = argparse.ArgumentParser(description='Texting GPT-2.')
        parser.add_argument("--text", type=str, required=True)
        parser.add_argument("--nsamples", type=int, default=1)
        parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.', default=False)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--length", type=int, default=-1)
        parser.add_argument("--config", type=str, required=False, default='config.yml')
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--load_pretrained", default=False, action='store_true')

        args = parser.parse_args()
        return args

if __name__ == '__main__':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params = get_parser()
        model = load_model_states(device, load_config(params.config), pretrained=params.load_pretrained)
        generate_text(model, load_config(params.config), device, 
                      text=params.text,
                      unconditional=params.unconditional,
                      temperature=params.temperature,
                      nsamples=params.nsamples,
                      batch_size=params.batch_size,
                      length=params.length,
                      top_k=params.top_k)