import os
import torch
import argparse
from gpt2.utils import load_openai_weight, load_config
from gpt2.model import get_gpt2
from gpt2.sample import generate_text

def get_parser():
        parser = argparse.ArgumentParser(description='Texting GPT-2.')
        parser.add_argument("--text", type=str, required=True)
        parser.add_argument("--nsamples", type=int, default=1)
        parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.', default=True)
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--length", type=int, default=-1)
        parser.add_argument("--config", type=str, required=False, default='config.yml')

        args = parser.parse_args()
        return args

def load_model_states(device, args):
        model = get_gpt2(args)        
        model.to(device)
        if not os.path.exists('assets/ops_gpt2_pretrained_states.pth'):
                assert os.path.exists('assets/gpt2-pytorch_model.bin'), \
                        "Download the openai gpt2 pretrained states. \nCommand: 'curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'"
                openai_state_dic = torch.load('assets/gpt2-pytorch_model.bin', map_location=device)
                try:
                        state_dict = load_openai_weight(model.state_dict(), openai_state_dic, args)
                        model.load_state_dict(state_dict)
                except:
                        print('OpenAI pretrained states are not compatible with OPS gpt-2.')
                torch.save(model.state_dict(), "./assets/ops_gpt2_pretrained_states.pth")
        else:
                state_dict = torch.load('assets/ops_gpt2_pretrained_states.pth', map_location=device)
                model.load_state_dict(state_dict)
        return model

if __name__ == '__main__':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        params = get_parser()
        model = load_model_states(device, load_config(params.config))
        generate_text(model, load_config(params.config), device, 
                      text=params.text,
                      unconditional=params.unconditional,
                      nsamples=params.nsamples,
                      batch_size=params.batch_size,
                      length=params.length)