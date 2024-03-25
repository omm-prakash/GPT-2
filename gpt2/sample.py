import torch
import numpy as np
from tqdm import trange
from gpt2.encoder import get_encoder
from gpt2.model import triangular_mask

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, device='cuda', sample=True):
        if start_token is None:
                assert context is not None, 'Specify exactly one of start_token and context!'
                context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        else:
                assert context is None, 'Specify exactly one of start_token and context!'
                context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        prev = context
        output = context
        print('prev shape: ', prev.size())
        past = None
        with torch.no_grad():
                for __ in trange(length):
                        mask = triangular_mask(prev.size(1))
                        logits, past = model(prev, mask, past=past) # shape: (batch, seq, vocab_size), [ (2, batch, head, seq, k_d)*N ]
                        log_probs = torch.nn.functional.softmax(logits, dim=-1) # shape: (batch, seq, vocab_size)
                        if sample:
                                prev = torch.multinomial(log_probs, num_samples=1)
                        else:
                                _, prev = torch.topk(log_probs, k=1, dim=-1)
                        output = torch.cat((output, prev), dim=1) 
        return output

def generate_text(model, args, device, text, unconditional, nsamples=1, batch_size=-1, length=-1, seed=21):
        if batch_size == -1:
                batch_size = 1
        assert nsamples % batch_size == 0

        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        if length == -1:
                length = args.seq_len // 2
        elif length > args.seq_len:
                raise ValueError("Can't get samples longer than window size: %s" % args.seq_len)
        
        print('Input text:',text)
        enc = get_encoder()
        context_tokens = enc.encode(text)

        model.eval()
        generated = 0
        for _ in range(nsamples // batch_size):
                out = sample_sequence(model, length, 
                                      start_token = enc.encoder['<|endoftext|>'] if unconditional else None,
                                      batch_size = batch_size,
                                      context = context_tokens  if not  unconditional else None,
                                      device = device,
                                      sample=True)

                out = out[:, len(context_tokens):].tolist()
                for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(text)

        return 
