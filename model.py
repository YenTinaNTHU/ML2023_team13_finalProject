import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class FCNN(nn.Module):
    def __init__(self, input_dim, dim=32):
        super(FCNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),   
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),   
            nn.BatchNorm1d(dim),
            nn.ReLU(),       
            nn.Linear(dim, 1),   
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
# class FCNN512t512t512(nn.Module):
#     def __init__(self, input_dim):
#         super(FCNN, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),   
#             nn.ReLU(),
#             nn.Linear(512, 512),   
#             nn.ReLU(),
#             nn.Linear(512, 512),   
#             nn.ReLU(),             
#             nn.Linear(512, 1),   
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits    

# class FCNN256t256t256(nn.Module):
#     def __init__(self, input_dim):
#         super(FCNN, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),   
#             nn.ReLU(),
#             nn.Linear(256, 256),   
#             nn.ReLU(),            
#             nn.Linear(256, 1),   
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits

# class FCNN256t256t64(nn.Module):
#     def __init__(self, input_dim):
#         super(FCNN, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),   
#             nn.ReLU(),
#             nn.Linear(256, 64),   
#             nn.ReLU(),            
#             nn.Linear(64, 1),   
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits  

# class FCNN64t64t64(nn.Module):
#     def __init__(self, input_dim):
#         super(FCNN, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),   
#             nn.ReLU(),
#             nn.Linear(64, 64),   
#             nn.ReLU(),            
#             nn.Linear(64, 1),   
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits
    
# class FCNN64t64(nn.Module):
#     def __init__(self, input_dim):
#         super(FCNN, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),   
#             nn.ReLU(),
#             nn.Linear(64, 1),   
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits