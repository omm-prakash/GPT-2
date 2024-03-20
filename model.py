import torch
import torch.nn as nn
from utils import Error
from attention import MultiHeadAttention, GroupQueryAttention, SlidingWindowAttention
from embedding import Embedding, PositionEmbedding, TransformerPositionEmbedding, RotaryPositionEmbedding


class MLP(nn.Module):
        def __init__(self, d_model, mlp_scale, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.c_fc = nn.Linear(d_model, mlp_scale*d_model) # same as the conv1D in OpenAI implementation
                self.c_proj = nn.Linear(mlp_scale*d_model, d_model) # same as the conv1D by OpenAI implementation
                self.gelu = nn.GELU()
                
        def forward(self, x):
                x = self.gelu(self.c_fc(x)) # shape: (batch, seq, mlp_scale*d_model)
                return self.c_proj(x) # shape: (batch, seq, d_model)

class LayerNorm(nn.Module):
        def __init__(self, d_model, eps=10e-8, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.weight = nn.Parameter(torch.ones(int(d_model)))
                self.bias = nn.Parameter(torch.zeros(int(d_model)))
                self.eps = eps
        
        def forward(self, x):
                mean = torch.mean(x, dim=-1, keepdim=True) # shape: (batch, seq, 1) 
                std = torch.std(x, dim=-1, keepdim=True) # shape: (batch, seq, 1) 
                return self.weight*((x-mean)/(std+self.eps)) + self.bias # shape: (batch, seq, d_model)                

class DecoderBlock(nn.Module):
        def __init__(self, d_model, head, drop, eps, mlp_scale, attention_type, context, groups, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.attn = get_attention(attention_type, d_model, head, context, groups)
                self.mlp = MLP(d_model, mlp_scale)
                self.ln_1 = LayerNorm(d_model, eps)
                self.ln_2 = LayerNorm(d_model, eps)
                if drop is not None:
                        self.dropout = nn.Dropout(drop)
                else:
                        self.dropout = None

        def forward(self, x, mask, layer_past=None):
                hidden,present = self.attn(self.ln_1(x), mask, layer_past)
                x = x + hidden
                x = x + self.mlp(self.ln_2(x))
                x = self.dropout(x) if self.dropout is not None else x
                return x, present # shape: (batch, seq, d_model), (2, batch, head, seq, k_d)

class Decoder(nn.Module):
        def __init__(self, d_model, head, drop, eps, N, mlp_scale, attention_type, context, groups, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.N = N
                self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, head, drop, eps, mlp_scale, attention_type, context, groups) for _ in range(N)])
                
        def forward(self, x, mask, layer_pasts=None):
                presents = []
                if layer_pasts is not None:
                        assert len(layer_pasts)==self.N, "layer_pasts need to be same as N"
                for i in range(self.N):
                        x, present = self.decoder_blocks[i](x,mask,layer_pasts[i])
                        presents.append(present)
                return x, presents # shape: (batch, seq, d_model), [ (2, batch, head, seq, k_d)*N ]

class Projection(nn.Module): # same as GPT2LMHead in OpenAI implementation
        def __init__(self, d_model, vocab_size, weights, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                assert d_model==weights.shape[1], "d_model should match with 1st dimension of embedding weights"
                assert vocab_size==weights.shape[0], "vocab_size should match with 2nd dimension of embedding weights"
                self.linear = nn.Linear(d_model, vocab_size, bias=False)
                self.linear.weight.requires_grad = False
                self.set_embedding_weights(weights)
        
        def set_embedding_weights(self, weights):
                self.linear.weight = weights

        def forward(self, x):
                return self.linear(x) # shape: (batch, seq, vocab_size)

class Transformer(nn.Module): # same as GPT2Model in OpenAI implementations
        def __init__(self, d_model, head, drop, eps, N, vocab_size, seq_len, mlp_scale, position_embedding_type, base, attention_type, context, groups, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.embedding = Embedding(vocab_size, d_model)
                self.pos_embedding = get_position_embedding(position_embedding_type, d_model, seq_len, base, drop)
                self.decoder = Decoder(d_model, head, drop, eps, N, mlp_scale, attention_type, context, groups)
                self.ln_f = LayerNorm(d_model, eps)

        def forward(self, mask, past=None):
                x = self.pos_embedding(self.embedding(x)) # shape: (batch, seq, d_model)
                x, presents = self.decoder(x, mask, past) # shape: (batch, seq, d_model)
                x = self.ln_f(x) # shape: (batch, seq, d_model)
                return x, presents # shape: (batch, seq, d_model), [ (2, batch, head, seq, k_d)*N ]

class GPT2(nn.Module): # same as GPT2LMHeadModel in OpenAI implementations
        def __init__(self, d_model, head, drop, eps, N, vocab_size, seq_len, mlp_scale, position_embedding_type, base, attention_type, context, groups, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.transformer = Transformer(d_model, head, drop, eps, N, vocab_size, seq_len, mlp_scale, position_embedding_type, base, attention_type, context, groups)
                self.projection = Projection(d_model, vocab_size, self.transformer.embedding.get_embedding_weight())

        def forward(self, x, mask, past=None, labels=None):
                x, presents = self.transformer(x, mask, past)
                x = self.projection(x) # shape: (batch, seq, vocab_size)

                if labels is not None:
                        loss_fun = nn.CrossEntropyLoss(ignore_index=-1)
                        loss = loss_fun(x.view(-1, x.size(-1)), labels.view(-1))
                        return loss
                return x, presents  # shape: (batch, seq, vocab_size), [ (2, batch, head, seq, k_d)*N ]

def get_attention(name, d_model, head, context, groups):
        if name=='group-query':
                attention = GroupQueryAttention(d_model, head, groups)
        elif name=='sliding-window':
                attention = SlidingWindowAttention(d_model, head, context)
        elif name=='transformer':
                attention = MultiHeadAttention(d_model, head)
        else:
                raise Error('attention type not found!!')
        return attention

def get_position_embedding(name, d_model, seq_len, base, drop):
        if name=='standard':
                embedding = PositionEmbedding(seq_len, d_model)
        elif name=='rotary':
                embedding = RotaryPositionEmbedding(d_model, seq_len, base)
        elif name=='transformer':
                embedding = TransformerPositionEmbedding(seq_len, d_model, drop)
        else:
                raise Error('position embedding type not found!!')
        return embedding

def get_gpt2(args):
        model = GPT2(d_model = args.d_model,
                     head = args.head, 
                     drop = args.drop,
                     eps = args.eps,
                     N = args.N,
                     vocab_size = args.vocab_size,
                     seq_len = args.seq_len,
                     mlp_scale = args.mlp_scale,
                     position_embedding_type = args.position_embedding_type, 
                     base = args.base,
                     attention_type = args.attention_type, 
                     context = args.context, 
                     groups = args.groups)
        
        for param in model.parameters():
                if param.dim()>1:
                        nn.init.xavier_normal_(param)
        return model