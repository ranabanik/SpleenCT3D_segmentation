from model import UNet
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input = torch.random([10, 64, 64, 8]).to(device)

Net = UNet(in_dim=3, out_dim=3, num_filters=4)
# print(Net)

print(Net.up_4.children)