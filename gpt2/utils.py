import os
import yaml
import torch
from pathlib import Path
from copy import deepcopy
from gpt2.model import get_gpt2
import time, datetime

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
            return DictToObject(config_data)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

class Error(Exception):
    def __init__(self, message):
        super().__init__(message)

def load_openai_weight(ops_state_dict, openai_state_dict, args):
    state = deepcopy(ops_state_dict)
    openai_weight_patterns_transpose = ['h.{n}.attn.c_attn.weight',
                                        'h.{n}.attn.c_proj.weight',
                                        'h.{n}.mlp.c_fc.weight',
                                        'h.{n}.mlp.c_proj.weight',]

    openai_weight_patterns = ['h.{n}.attn.c_attn.bias',
                              'h.{n}.attn.c_proj.bias',
                              'h.{n}.mlp.c_fc.bias',
                              'h.{n}.mlp.c_proj.bias',
                              'h.{n}.ln_1.weight', 
                              'h.{n}.ln_1.bias', 
                              'h.{n}.ln_2.weight', 
                              'h.{n}.ln_2.bias', ]
    
    # loading weight for attention and mlp layers for each block
    # mismatch was due to use of Conv1D in openai implementation
    print('Loading OpenAI GPT-2 Pretrained States on OPS GPT-2 Model...')
    for i in range(args.N):
        for key in openai_weight_patterns_transpose:
            ops_weight_key = 'transformer.decoder.decoder_blocks'+key.format(n=i)[1:]
            state[ops_weight_key] = openai_state_dict[key.format(n=i)].transpose(-1,-2)
        for key in openai_weight_patterns:
            ops_weight_key = 'transformer.decoder.decoder_blocks'+key.format(n=i)[1:]
            state[ops_weight_key] = openai_state_dict[key.format(n=i)] #.transpose(-1,-2)
            
    state['transformer.embedding.embedding.weight'] = openai_state_dict['wte.weight']
    state['projection.linear.weight'] = openai_state_dict['wte.weight']
    state['transformer.pos_embedding.pos_embedding.weight'] = openai_state_dict['wpe.weight']
    state['transformer.ln_f.weight'] = openai_state_dict['ln_f.weight']
    state['transformer.ln_f.bias'] = openai_state_dict['ln_f.bias']
    print('Done, all state loaded!!')
    return state

def load_model_states(device, args, pretrained=False):
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
        elif pretrained:
                if args.position_embedding_type == 'standard' and args.attention_type == 'transformer':
                        state_dict = torch.load('assets/ops_gpt2_pretrained_states.pth', map_location=device)
                        model.load_state_dict(state_dict)
                else:
                        print('\nMismatch in selected position_emebdding or attention_type with pretrained model.')
                        exit(1)
        return model

def get_weights_file_path(config, epoch: str):
        model_folder = f"{config.model_folder}"
        if not os.path.exists(model_folder):
                os.makedirs(model_folder)
        model_filename = f"{config.model_basename}@{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
        model_folder = f"{config.model_folder}"
        model_filename = f"{config.model_basename}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])

class Time():
    def __init__(self):
        self.begin = 0
        self.final = 0
    def now(self):
        return datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    def reset(self):
        self.begin = time.time()
        self.final = time.time()        
    def start(self, message=None):
        # if message:
        self.message = message
        self.begin = time.time()
    def end(self):
        self.final = time.time()
        tm = float(self.final-self.begin)
        unit = 'sec'
        if tm > 60:
            tm = tm/60
            unit = 'min'
        elif tm > 3600:
            tm = tm/3600
            unit = 'hr'
        if self.message:
            print('>> {}: Done!! Time taken: {:.4f} {}'.format(self.message, tm, unit))
        else:
            print('>> Done!! Time taken: {:.4f} {}'.format(tm, unit))
        self.message = None