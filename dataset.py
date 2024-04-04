"""Byte pair encoding utilities"""
import os
import json
import torch
import regex as re
from pathlib import Path
from gpt2.utils import Time
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(encoderf='./assets/encoder.json', vocabf='./assets/vocab.bpe'):
    with open(encoderf, 'r') as f:
        encoder = json.load(f)
    with open(vocabf, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges)

class GPT2Dataset(Dataset):
        def __init__(self, raw_dataset) -> None:
                super().__init__()
                self.raw_dataset = raw_dataset

                self.encoder = get_encoder()
                self.sos = self.encoder.encode('<|endoftext|>')   

        def __len__(self):
                return len(self.raw_dataset)
        
        def __getitem__(self, index):                
                sentence = self.raw_dataset[index]
                sentence_tokens = self.encoder.encode(sentence) # shape: (seq)

                # input padding
                input_tokens = torch.cat((sentence_tokens, self.sos), dim=0)
                input = torch.tensor(input_tokens, dtype=torch.int64)

                # label padding
                label_tokens = torch.cat((sentence_tokens[1:], self.sos, self.sos), dim=0)
                label = torch.tensor(label_tokens, dtype=torch.int64)

                assert input.size(0)==label.size(0)

                return {
                        'input': input, 
                        'label': label,
                        'text': sentence
                }

def get_data(config_file):
        try:
                with open(config_file, 'r') as file:
                        sentences = file.readlines()
                        for sentence in sentences:
                                sentence = sentence.strip()
                if sentences:
                        return sentences
                else:
                        print("No sentences were read from the file.")

        except FileNotFoundError:
                print(f"File '{config_file}' not found.")
                return []

def get_dataset(config, fsdp=False, rank=None, world_size=None):
        data_path = Path(os.path.join(os.getcwd(), config.data_path, config.dataset_filename))
        tm = Time()
        if data_path.exists():
                print(f'>> Data file found at {data_path}')
                print(">> Loading the data file.")
                dataset = get_data(data_path)
        else:
                print('>> Dataset not found. Terminating the process.')
                exit(1)

        print('\nLoading dataset..')
        tm.start('loading data')

        val_size = int(config.split_ratio*len(dataset))
        train_size = len(dataset)-val_size
        train_dataset_raw, val_dataset_raw = random_split(dataset, lengths=[train_size, val_size])

        train_dataset = GPT2Dataset(train_dataset_raw)
        val_dataset = GPT2Dataset(val_dataset_raw)
        
        if not fsdp:
                train_dataset = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                val_dataset = DataLoader(val_dataset, batch_size=1, shuffle=True)
        else:
                assert rank is not None, "Provide rank to load dataset while using FSDP."
                assert world_size is not None, "Provide GPU count to load dataset while using FSDP."
                sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
                sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

                train_kwargs = {'batch_size': config.batch_size, 'sampler': sampler1}
                test_kwargs = {'batch_size': 1, 'sampler': sampler2}
                cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}

                train_kwargs.update(cuda_kwargs)
                test_kwargs.update(cuda_kwargs)

                train_dataset = DataLoader(train_dataset, **train_kwargs)
                val_dataset = DataLoader(val_dataset, **test_kwargs)
                
        tm.end()
        return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
        }