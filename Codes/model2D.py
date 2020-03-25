import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight = init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        m.weight = init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        m.bias.data.fill_(0)
        # if m.bias is not None:
            # init.kaiming_uniform_(m.bias)
        # print("Conv weight init")
    elif isinstance(m, nn.BatchNorm2d):
        # m.weight.fill_(1)
        m.weight.data.normal_(mean=1, std=0.02)
        m.bias.data.zero_()
        # print("BN weight init")
    elif isinstance(m, nn.Linear):
        stdv = 1/np.sqrt(m.in_features)
        m.weight = init.kaiming_uniform_(-stdv, stdv)
        # m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            # bound = 1/np.sqrt(fan_in)
            # init.uniform_(m.bias, -bound, bound)
            m.bias.uniform_(-stdv, stdv)

def conv_block_2d(in_dim, out_dim, activation): #do not change spatially
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation)

def conv_trans_block_2d(in_dim, out_dim, activation):
    "doubles the spatial dimension..."
    return nn.Sequential(
        # nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, output_padding=0), #2, 2, 0
        nn.BatchNorm2d(out_dim),
        activation)

def max_pooling_2d():
    "Halves the spatial dimension"
    # return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def conv_block_2_2d(in_dim, out_dim, activation): #used in bridge
    "Doesn't change spatial dimension. same as before just one Conv+BN layer more"
    return nn.Sequential(
        conv_block_2d(in_dim, out_dim, activation),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        """
        :param in_dim: number of channels: x = torch.Tensor(1, 2, 128, 128, 128) -> 2
        :param out_dim: out size: torch.Size([1, 5, 128, 128, 128]) -> 5
        :param num_filters: kernels, increments throught multiplication in down and
                            decrements vice versa in up and trans
        :return out: output should be a mask
        """
        super(UNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True) #LeakyRelu(0.2)
        # Down sampling
        # In every down sample block the number of filters double...
        self.down_1 = conv_block_2_2d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_2d()
        self.down_2 = conv_block_2_2d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_2d()
        self.down_3 = conv_block_2_2d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_2d()
        self.down_4 = conv_block_2_2d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_2d()
        # self.down_5 = conv_block_2_2d(self.num_filters * 8, self.num_filters * 16, activation)
        # self.pool_5 = max_pooling_2d()

        # Bridge
        self.bridge = conv_block_2_2d(self.num_filters * 8, self.num_filters * 16, activation) # 16 -> 32

        # Up sampling
        self.trans_1 = conv_trans_block_2d(self.num_filters * 16, self.num_filters * 8, activation) # 32 -> 32
        self.up_1 = conv_block_2_2d(self.num_filters * 16, self.num_filters * 8, activation)
        self.trans_2 = conv_trans_block_2d(self.num_filters * 8, self.num_filters * 4, activation)
        self.up_2 = conv_block_2_2d(self.num_filters * 8, self.num_filters * 4, activation)
        self.trans_3 = conv_trans_block_2d(self.num_filters * 4, self.num_filters * 2, activation)
        self.up_3 = conv_block_2_2d(self.num_filters * 4, self.num_filters * 2, activation)
        self.trans_4 = conv_trans_block_2d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_4 = conv_block_2_2d(self.num_filters * 2, self.num_filters * 1, activation)
        # self.trans_5 = conv_trans_block_2d(self.num_filters * 1, self.num_filters * 2, activation)
        # self.up_5 = conv_block_2_2d(self.num_filters * 1, self.out_dim, activation) # out_dim -> num_filters

        # Output
        self.out = conv_block_2d(self.num_filters, out_dim, activation)
        self.last = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # ->
        # print("down_1", down_1.shape)
        pool_1 = self.pool_1(down_1)  # ->
        # print("pool_1", pool_1.shape)
        down_2 = self.down_2(pool_1)  # ->
        # print("down_2", down_2.shape)
        pool_2 = self.pool_2(down_2)  # ->
        # print("pool_2", pool_2.shape)
        down_3 = self.down_3(pool_2)  # ->
        # print("down_3", down_3.shape)
        pool_3 = self.pool_3(down_3)  # ->
        # print("pool_3", pool_3.shape)
        down_4 = self.down_4(pool_3)  # ->
        # print("down_4", down_4.shape)
        pool_4 = self.pool_4(down_4)  # ->
        # print("pool_4", pool_4.shape)
        # down_5 = self.down_5(pool_4)  # ->
        # print("down_5", down_5.shape)
        # pool_5 = self.pool_5(down_5)  # ->
        # print("pool_5", pool_5.shape)
        # Bridge
        bridge = self.bridge(pool_4)  # ->
        # print("bridge", bridge.shape)
        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> ]
        # print("trans_1", trans_1.shape)
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # ->
        # print("concat_1", concat_1.shape)
        up_1 = self.up_1(concat_1)  # ->
        # print("up_1", up_1.shape)
        trans_2 = self.trans_2(up_1)  # ->
        # print("trans_2", trans_2.shape)
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # ->
        # print("concat_2", concat_2.shape)
        up_2 = self.up_2(concat_2)  # ->
        # print("up_2", up_2.shape)
        trans_3 = self.trans_3(up_2)  # ->
        # print("trans_3", trans_3.shape)
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # ->
        # print("concat_3", concat_3.shape)
        up_3 = self.up_3(concat_3)  # ->
        # print("up_3", up_3.shape)
        trans_4 = self.trans_4(up_3)  # ->
        # print("trans_4", trans_4.shape)
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # ->
        # print("concat_4", concat_4.shape)
        up_4 = self.up_4(concat_4)  # ->
        # print("up_4", up_4.shape)
        # trans_5 = self.trans_5(up_4)  # ->
        # print("trans_5", trans_5.shape)
        # concat_5 = torch.cat([trans_5, down_1], dim=1)  # ->
        # print("concat_5", concat_5.shape)
        # up_5 = self.up_5(concat_5)  # ->
        # print("up_5", up_5.shape)
        # Output
        out = self.out(up_4)  # ->
        # out = self.last(out)
        # out = F.sigmoid(out)
        out = torch.sigmoid(self.last(out))
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_dim=1, out_dim=1, num_filters=32).to(device)
    model.apply(weights_init)
    # print(model)
    inp = torch.Tensor(8, 1, 512, 512).to(device)
    # print(inp.size())
    inp.to(device)
    out = model(inp)
    print(out.size())

# if __name__ != '__main__':
#     encoder = models.vgg16() # .features
#     print(encoder)


