import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MLP3(nn.Module):
    def __init__(self, input_size, hidden_size, outdim, format):
        super(MLP3, self).__init__()
        self.format = format
        assert(self.format == 'regAng')
        self.pl2weight = nn.Linear(1,1)
        self.gain = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x_):
        """
        input x is of shape (batch_size, seq_len, input_size)
        """
        B, L, D = x_.size()
        # convert pl scalar to a weight value
        w = self.pl2weight(x_.reshape(B*L,D)[:,2,None]).reshape(B,L,1)
        w = F.softmax(w,dim=1)
        x = x_[:,:,:2] * w
        o = x.mean(dim=1) * self.gain
        return o[:,1,None], w
    
class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, outdim, format):
        super(MLP2, self).__init__()
        self.format = format
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, outdim)
        
    def forward(self, x_):
        """
        input x is of shape (batch_size, seq_len, input_size)
        """
        B, L, D = x_.size()
        x = F.relu(self.fc1(x_))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(B, L, -1)
        o = x.sum(dim=1)
        return o, x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, outdim, format):
        super(MLP, self).__init__()
        self.format = format
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, outdim)
        
    def forward(self, x_):
        """
        input x is of shape (batch_size, seq_len, input_size)
        """
        B, L, D = x_.size()
        x = F.relu(self.fc1(x_))#[:,:,2:].reshape(-1,D-2)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(B, L, -1)
        x -= x.min(1, keepdim=True)[0]
        if self.format == 'regBoth':
            o = (x * x_[:,:,:2]).mean(dim=1)
        else:
            o = (x * x_[:,:,:1]).mean(dim=1)            
        return o, x

class QKVAttention(nn.Module):
    def __init__(self, input_size, hidden_size, outdim, num_heads=1,num_mha=1):
        super(QKVAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.preproj = nn.Linear(input_size, hidden_size)
        
        # Define multi-head attention layer
        self.multihead_attns = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(num_mha)])
        
        # Project the pooled output to the output size
        self.fc = nn.Linear(hidden_size, outdim)
        
    def forward(self, vectors):
        """
        vectors: Tensor of shape (batch_size, seq_len, input_size)
        """
        # Apply multi-head attention layers
        attended_vectors = self.preproj(vectors)
        for multihead_attn in self.multihead_attns:
            attended_vectors, _ = multihead_attn(attended_vectors, attended_vectors, attended_vectors)
        
        # Project the pooled output to the output size
        attended_vectors = self.fc(attended_vectors.mean(dim=1))
        return [attended_vectors]

def controlModel(input_size, hidden_size, outdim, format, model_type='mlp',num_heads=4):
    if model_type == 'mlp':
        return MLP(input_size, hidden_size, outdim, format)
    elif model_type == 'mlp2':
        return MLP2(input_size, hidden_size, outdim, format)
    elif model_type == 'qkv':
        return QKVAttention(input_size, hidden_size, outdim, num_heads)
    elif model_type == 'mlp3':
        return MLP3(input_size, hidden_size, outdim, format)
    else:
        raise ValueError("model_type must be one of 'mlp', 'mlp2', 'qkv'")

    