import torch
import torch.nn as nn

class Embedding(nn.Module):
        def __init__(self, vocab_size, d_model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)

        def forward(self, x):
                x = self.embedding(x)
                return x*torch.sqrt(torch.tensor(self.d_model, dtype=torch.int, requires_grad=False))  # shape: (batch, seq, d_model)

class PositionEmbedding(nn.Module):
        def __init__(self, seq_len, d_model, drop, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.seq_len = seq_len
                self.d_model = d_model
                self.dropout = nn.Dropout(drop)
                
                embedding = torch.empty(self.seq_len, self.d_model) # shape: (seq, d_model)
                numerator = torch.arange(0, self.seq_len).unsqueeze(1) # shape: (seq, 1)
                denominator = torch.exp(-1*torch.arange(0, self.d_model, 2, dtype=torch.float)* \
                                        torch.log(torch.tensor(10000))/self.d_model) # shape: (d_model/2)

                embedding[:,0::2] = torch.sin(numerator*denominator) # shape: (seq, d_model/2)
                embedding[:,1::2] = torch.cos(numerator*denominator) # shape: (seq, d_model/2)

                self.embedding = embedding.unsqueeze(0) # shape: (1, seq, d_model)
                if not hasattr(self, 'embedding'):
                        self.register_buffer('embedding', embedding)

        def forward(self, x):
                x = x+self.embedding[:, :self.seq_len, :].requires_grad_(False)
                return self.dropout(x) # shape: (batch, seq, d_model)

class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, head, drop, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.d_model = d_model
                self.Wq = nn.Linear(d_model, d_model)
                self.Wv = nn.Linear(d_model, d_model)
                self.Wk = nn.Linear(d_model, d_model)
                self.Wo = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(drop)
                self.head = head

        def forward(self, V, Q, K, mask):
                # shape: (batch, seq, d_model)
                value = self.Wv(V)
                query = self.Wq(Q)
                key = self.Wk(K)

                k_d = self.d_model//self.head
                # shape: (batch, seq, head, k_d) --> (batch, head, seq, k_d)
                value = value.view(value.shape[0], value.shape[1], self.head, k_d).transpose(1,2)
                query = query.view(query.shape[0], query.shape[1], self.head, k_d).transpose(1,2)
                key = key.view(key.shape[0], key.shape[1], self.head, k_d).transpose(1,2)
                
                # shape: (batch, head, seq, seq)
                map = (torch.matmul(query, key.transpose(-1,-2)))/torch.sqrt(k_d)
                map = self.dropout(nn.functional.softmax(torch.mul(map, mask)))

                # shape: (batch, head, seq, seq) --> (batch, head, seq, k_d) --> (batch, seq, head, k_d) --> (batch, head, d_model)
                out = torch.matmul(map, value).transpose(1,2).contiguous().view(V.shape[0], V.shape[1], self.d_model)
                
                return self.Wo(out) # shape: (batch, seq, d_model)

class FeedForward(nn.Module):
        def __init__(self, d_model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.linear1 = nn.Linear(d_model, d_model)
                self.linear2 = nn.Linear(d_model, d_model)
                self.gelu = nn.GELU()
                
        def forward(self, x):
                x = self.gelu(self.linear1(x))
                return self.linear2(x) # shape: (batch, seq, d_model)

class LayerNorm(nn.Module):
        def __init__(self, eps=10e-8, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.alpha = nn.Parameter(torch.ones(1))
                self.bias = nn.Parameter(torch.zeros(1))
                self.eps = eps
        
        def forward(self, x):
                mean = torch.mean(x, dim=-1, keepdim=True)
                std = torch.std(x, dim=-1, keepdim=True)
                return self.alpha*((x-mean)/(std+self.eps)) + self.bias # shape: (batch, seq, d_model)                

class ResidualBlock(nn.Module):
        def __init__(self, eps, drop, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.norm = LayerNorm(eps)
                self.dropout = nn.Dropout(drop)
        
        def forward(self, x, sublayer):
                return x + self.dropout(sublayer(self.norm(x))) # shape: (batch, seq, d_model)

class Projection(nn.Module):
        def __init__(self, d_model, vocab_size, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.linear = nn.Linear(d_model, vocab_size)
                self.softmax = nn.LogSoftmax(dim=-1)
        
        def forward(self, x):
                return self.softmax(self.linear(x)) # shape: (batch, seq, vocab_size)

class DecoderBlock(nn.Module):
        def __init__(self, d_model, head, drop, eps, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.attention = MultiHeadAttention(d_model, head, drop)
                self.mlp = FeedForward(d_model)
                self.residual1 = ResidualBlock(eps, drop)
                self.residual2 = ResidualBlock(eps, drop)
        
        def forward(self, x, mask):
                x = self.residual1(x, lambda y: self.attention(y,y,y,mask))
                x = self.residual2(x, lambda y: self.mlp(y))
                return x # shape: (batch, seq, d_model)

class Decoder(nn.Module):
        def __init__(self, d_model, head, drop, eps, N, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.N = N
                self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, head, drop, eps) for _ in range(N)])
                
        def forward(self, x, mask):
                for i in range(self.N):
                        x = self.decoder_blocks[i](x,mask)
                return x # shape: (batch, seq, d_model)

class GPT_2(nn.Module):
        def __init__(self, d_model, head, drop, eps, N, vocab_size, seq_len, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.embedding = Embedding(vocab_size, d_model)
                self.pos_embedding = PositionEmbedding(seq_len, d_model, drop)
                self.decoder = Decoder(d_model, head, drop, eps, vocab_size, N)
                self.norm = LayerNorm(eps)
                self.projection = Projection(d_model, vocab_size)

        def forward(self, x, mask): # shape: (batch, seq, vocab_size)
                x = self.pos_embedding(self.embedding(x)) # shape: (batch, seq, d_model)
                x = self.decoder(x, mask) # shape: (batch, seq, d_model)
                x = self.norm(x) # shape: (batch, seq, d_model)
                x = self.projection(x) # shape: (batch, seq, vocab_size)
                return x

def get_gpt2(args):
        model = GPT_2(d_model = args.d_model,
                      head = args.head, 
                      drop = args.drop,
                      eps = args.eps,
                      N = args.N,
                      vocab_size = args.vocab_size,
                      seq_len = args.seq_len)
        
        for param in model.parameters():
                if param.dim()>1:
                        nn.init.xavier_normal_(param)

        return model