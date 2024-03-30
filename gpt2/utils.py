import yaml
from copy import deepcopy

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

