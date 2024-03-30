import torch
import torch.nn as nn

class Embedding(nn.Module):
        def __init__(self, vocab_size, d_model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)
        
        def get_embedding_weight(self):
                return self.embedding.weight # shape: (vocab_size, d_model)

        def forward(self, x):
                x = self.embedding(x)
                return x # shape: (batch, seq, d_model)


class PositionEmbedding(nn.Module):
        def __init__(self, seq_len, d_model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.seq_len = seq_len
                self.pos_embedding = nn.Embedding(seq_len, d_model)

        def forward(self, x, past_length):
                pos_ids = self.pos_embedding(torch.arange(past_length,x.size(1)+past_length)).unsqueeze(0)
                x = x + pos_ids
                return x # shape: (batch, seq, d_model)


class RotaryPositionEmbedding(nn.Module):
        def __init__(self, d_model, seq_len, base, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.d_model = d_model
                self.seq_len = seq_len
                self.base = base
                self.theta_matrix = None

        def compute_theta(self):
                if self.theta_matrix is None:
                        theta = torch.pow(self.base, -1*torch.arange(0,self.d_model,2,dtype=torch.double)/self.d_model).unsqueeze(0) # shape: (1, d_model//2)
                        sequence = torch.arange(0,self.seq_len,dtype=torch.double).unsqueeze(-1) # shape: (seq, 1)
                        theta = torch.matmul(sequence, theta) # shape: (seq, d_model//2)
                        cos_theta = torch.cos(theta) # shape: (seq, d_model//2)
                        sin_theta = torch.sin(theta) # shape: (seq, d_model//2)
                        theta_matrix = torch.stack([torch.stack([cos_theta, -1*sin_theta], dim=-1),\
                                                         torch.stack([sin_theta, cos_theta], dim=-1)], dim=-2) # shape: (seq, d_model//2, 2, 2)
                        theta_matrix = theta_matrix.transpose(-1,-2) # shape: (seq, d_model//2, 2, 2)
                        # if not hasattr(self, 'theta_matrix'):
                        self.register_buffer('theta_matrix', theta_matrix)

        def forward(self, x, past_length):
                self.compute_theta()
                shape = x.shape
                x = x.view(*shape[:-1],2,self.d_model//2) # shape: (batch, seq, 2, d_model//2)
                x = x.transpose(-1,-2).unsqueeze(-2) # shape: (batch, seq, d_model//2, 1, 2)

                x = torch.matmul(x, self.theta_matrix[past_length:x.size(1)+past_length,:,:,:]) # shape: (batch, seq, d_model//2, 1, 2)
                x = x.squeeze(-2) # shape: (batch, seq, d_model//2, 2)
                x = x.transpose(-1,-2).contiguous() # shape: (batch, seq, 2, d_model//2)
                x = x.view(*shape[:-1], self.d_model) # shape: (batch, seq, d_model)

                return x # shape: (batch, seq, d_model)


class TransformerPositionEmbedding(nn.Module):
        def __init__(self, seq_len, d_model, drop, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.seq_len = seq_len
                self.d_model = d_model
                self.dropout = nn.Dropout(drop if drop is not None else 0)
                
                embedding = torch.empty(self.seq_len, self.d_model) # shape: (seq, d_model)
                numerator = torch.arange(0, self.seq_len).unsqueeze(1) # shape: (seq, 1)
                denominator = torch.exp(-1*torch.arange(0, self.d_model, 2, dtype=torch.float)* \
                                        torch.log(torch.tensor(10000))/self.d_model) # shape: (d_model/2)

                embedding[:,0::2] = torch.sin(numerator*denominator) # shape: (seq, d_model/2)
                embedding[:,1::2] = torch.cos(numerator*denominator) # shape: (seq, d_model/2)

                self.embedding = embedding.unsqueeze(0) # shape: (1, seq, d_model)
                if not hasattr(self, 'embedding'):
                        self.register_buffer('embedding', embedding)

        def forward(self, x, past_length):
                x = x+self.embedding[:, past_length:x.size(1)+past_length, :].requires_grad_(False)
                return self.dropout(x) # shape: (batch, seq, d_model)
