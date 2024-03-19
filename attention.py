import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, head, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.head = head
                self.d_model = d_model
                assert d_model%head==0, "d_model should be divisible by head"
                self.k_d = self.d_model//self.head

                self.c_attn = nn.Linear(d_model, 3*d_model) # same as the conv1D by OpenAI implementation
                self.c_proj = nn.Linear(d_model, d_model) # same as the conv1D by OpenAI implementation

        def split_heads(self, value, query, key):
                # shape: (batch, seq, head, k_d) --> (batch, head, seq, k_d)
                value = value.view(value.shape[0], value.shape[1], self.head, self.k_d).transpose(1,2)
                query = query.view(query.shape[0], query.shape[1], self.head, self.k_d).transpose(1,2)
                key = key.view(key.shape[0], key.shape[1], self.head, self.k_d).transpose(1,2)

                return value,query,key # shape: (batch, head, seq, k_d)

        def score(self,query,key,mask):
                # k = 2 if layer_past is not None else 1
                attention_score = torch.matmul(query, key.transpose(-1,-2))/\
                        torch.sqrt(torch.tensor(self.k_d, dtype=torch.int, requires_grad=False)) # shape: (batch, head, seq, k*seq)
                k = attention_score.shape[-1]//mask.shape[-1]
                if k>1:
                        mask = torch.cat([mask for _ in range(k)], dim=-1)
                if mask is not None:
                        attention_score.masked_fill_(mask==0, -1*torch.inf) # shape: (batch, head, seq, k*seq)
                attention_score = nn.functional.softmax(attention_score) # shape: (batch, head, seq, k*seq)
                
                return attention_score

        @staticmethod
        def merge_past(layer_past,key,value):
                if layer_past is not None:
                        past_key, past_value = layer_past[0], layer_past[1]  # transpose back cf below
                        key = torch.cat((past_key, key), dim=-2) # shape: (batch, head, 2*seq, k_d)
                        value = torch.cat((past_value, value), dim=-2) # shape: (batch, head, 2*seq, k_d)
                return torch.stack((key, value)),key,value # shape: (2, batch, head, seq, k_d), (batch, head, 2*seq, k_d), (batch, head, 2*seq, k_d)

        def forward(self, x, mask, layer_past=None):
                x = self.c_attn(x) # shape: (batch, seq, 3*d_model)
                
                value,query,key = x.split(split_size=self.d_model, dim=-1) # shape: (batch, seq, d_model)
                value,query,key = self.split_heads(value,query,key) # shape: (batch, head, seq, k_d)

                present,key,value = MultiHeadAttention.merge_past(layer_past)

                attention_score = self.score(query,key,mask)
                out = torch.matmul(attention_score, value) # shape: (batch, head, seq, k_d)
                # shape: (batch, head, seq, k_d) --> (batch, seq, head, k_d) --> (batch, head, d_model)
                out = out.transpose(1,2).contiguous().view(value.shape[0], value.shape[1], self.d_model)
                out = self.c_proj(out) # shape: (batch, seq, d_model)
                
                return out, present # shape: (batch, seq, d_model), (2, batch, head, seq, k_d)


class GroupQueryAttention(nn.Module):
        def __init__(self, d_model, head, groups, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.head = head
                self.d_model = d_model
                assert d_model%head==0, "d_model should be divisible by head"
                self.k_d = self.d_model//self.head
                self.groups = groups

                self.Wq = nn.Linear(d_model, groups*d_model)
                self.Wv = nn.Linear(d_model, d_model)
                self.Wk = nn.Linear(d_model, d_model)
                self.c_proj = nn.Linear(d_model, d_model) # same as the conv1D by OpenAI implementation

        def split_heads(self, value, query, key):
                # shape: (batch, seq, head, k_d) --> (batch, head, seq, k_d)
                value = value.view(value.shape[0], value.shape[1], self.head, self.k_d).transpose(1,2)
                query = query.view(query.shape[0], query.shape[1], self.head*self.groups, self.k_d).transpose(1,2)
                key = key.view(key.shape[0], key.shape[1], self.head, self.k_d).transpose(1,2)

                return value,query,key # shape: (batch, head, seq, k_d)

        def forward(self, x, mask, layer_past=None):
                value = self.Wv(x) # shape: (batch, seq, d_model)
                query = self.Wq(x) # shape: (batch, seq, groups*d_model)
                key = self.Wk(x) # shape: (batch, seq, d_model)
                value,query,key = self.split_heads(value,query,key) # shape: (batch, *head, seq, k_d)
                # *head: head for query is groups*head else same
                query = query.view(query.shape[0],self.head,self.groups,query.shape[2],self.k_d).transpose(1,2) # shape: (batch, groups, head, seq, k_d)

                present,key,value = MultiHeadAttention.merge_past(layer_past)

                # k = 2 if layer_past is not None else 1
                attention_score = torch.matmul(query, key.transpose(-1,-2))/\
                        torch.sqrt(torch.tensor(self.k_d, dtype=torch.int, requires_grad=False)) # shape: (batch, groups, head, seq, k*seq)
                # mean pooling along group dimension
                attention_score = torch.mean(attention_score, dim=1) # shape: (batch, head, seq, k*seq)
                
                if mask is not None and layer_past is None:
                        attention_score.masked_fill_(mask==0, -1*torch.inf) # shape: (batch, head, seq, seq)
                attention_score = nn.functional.softmax(attention_score) # shape: (batch, head, seq, k*seq)
                
                out = torch.matmul(attention_score, value) # shape: (batch, head, seq, k_d)
                # shape: (batch, head, seq, k_d) --> (batch, seq, head, k_d) --> (batch, head, d_model)
                out = out.transpose(1,2).contiguous().view(value.shape[0], value.shape[1], self.d_model)
                out = self.c_proj(out) # shape: (batch, seq, d_model)
                
                return out, present # shape: (batch, seq, d_model), (2, batch, head, seq, k_d)
        

class SlidingWindowAttention(MultiHeadAttention):
        def __init__(self, d_model, head, context, *args, **kwargs) -> None:
                super().__init__(d_model, head, *args, **kwargs)
                self.context = context

        def forward(self, x, mask, layer_past=None):
                x = self.c_attn(x) # shape: (batch, seq, 3*d_model)
                
                value,query,key = x.split(split_size=self.d_model, dim=-1) # shape: (batch, seq, d_model)
                value,query,key = super().split_heads(value,query,key) # shape: (batch, head, seq, k_d)
                
                present,key,value = MultiHeadAttention.merge_past(layer_past)
                
                mask = mask-torch.tril(mask, diagonal=-self.context)
                attention_score = super().score(query,key,mask)
                
                out = torch.matmul(attention_score, value) # shape: (batch, head, seq, k_d)
                # shape: (batch, head, seq, k_d) --> (batch, seq, head, k_d) --> (batch, head, d_model)
                out = out.transpose(1,2).contiguous().view(value.shape[0], value.shape[1], self.d_model)
                out = self.c_proj(out) # shape: (batch, seq, d_model)
                
                return out, present # shape: (batch, seq, d_model), (2, batch, head, seq, k_d)
