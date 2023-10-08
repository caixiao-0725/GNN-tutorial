import torch
from torch_geometric.data import Data
x = torch.tensor([[2,1],[5,6],[3,7],[12,0]],dtype = torch.float)
y = torch.tensor([0,1,0,1],dtype = torch.long)
edge_index = torch.tensor([[0,1,2,0,3],
                           [1,0,1,3,2]],dtype = torch.long)


data = Data(x=x,y=y,edge_index=edge_index)
print(data)