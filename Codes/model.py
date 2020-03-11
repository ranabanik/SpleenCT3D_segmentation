# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

# def weights_init(m):
#     if isinstance(m, nn.modules.conv._ConvNd): #nn.Conv3d
#         init.xavier_uniform_(m.weight, gain = np.sqrt(2.0)) #m.weight.data not used anymore?
#         init.zeros_(m.bias)#m.bias.data.fill_(0)
#         # torch.nn.init.xavier_uniform_(m.bias.data)
#         print("Conv happens")
#     elif isinstance(m, nn.modules.batchnorm._BatchNorm):
#         m.weight.data.normal_(mean=1.0, std=0.02)
#         m.bias.data.fill_(0)
#         print("batch happens")
#     elif isinstance(m, nn.Linear):
#         # m.weight.data.normal_(0.0, 0.02)
#         # init.xavier_uniform_(m.weight.data)
#         y = 1/np.sqrt(m.in_features)
#         m.weight.data.uniform_(-y, y)
#         m.bias.data.fill_(0) #0.01


# for m in self.modules():
#     if isinstance(m, nn.Conv3d):
#         m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
#     elif isinstance(m, nn.BatchNorm3d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        m.weight = init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        m.bias.data.fill_(0)
        # if m.bias is not None:
            # init.kaiming_uniform_(m.bias)
        print("Conv weight init")
    elif isinstance(m, nn.BatchNorm3d):
        # m.weight.fill_(1)
        m.weight.data.normal_(mean=1, std=0.02)
        m.bias.data.zero_()
        print("BN weight init")
    elif isinstance(m, nn.Linear):
        stdv = 1/np.sqrt(m.in_features)
        m.weight = init.kaiming_uniform_(-stdv, stdv)
        # m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            # bound = 1/np.sqrt(fan_in)
            # init.uniform_(m.bias, -bound, bound)
            m.bias.uniform_(-stdv, stdv)


def conv_block_3d(in_dim, out_dim, activation): #do not change spatially
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation)

def conv_trans_block_3d(in_dim, out_dim, activation):
    "doubles the spatial dimension..."
    return nn.Sequential(
        # nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=(2, 2, 1), padding=0, output_padding=0),
        nn.BatchNorm3d(out_dim),
        activation)

def max_pooling_3d():
    "Halves the spatial dimension"
    # return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return nn.MaxPool3d(kernel_size=2, stride=(2, 2, 1), padding=0)

def conv_block_2_3d(in_dim, out_dim, activation): #used in bridge
    "Doesn't change spatial dimension. same as before just one Conv+BN layer more"
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

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
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        # In every down sample block the number of filters double...
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)
        self.last = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 2, 1), stride=(2, 2, 1))
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        print("down_1", down_1.shape)
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        print("pool_1", pool_1.shape)
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        print("down_2", down_2.shape)
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        print("pool_2", pool_2.shape)
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        print("down_3", down_3.shape)
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        print("pool_3", pool_3.shape)
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        print("down_4", down_4.shape)
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]
        print("pool_4", pool_4.shape)
        down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        print("down_5", down_5.shape)
        pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
        print("pool_5", pool_5.shape)
        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]
        print("bridge", bridge.shape)
        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        print("trans_1", trans_1.shape)
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        print("concat_1", concat_1.shape)
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        print("up_1", up_1.shape)
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        print("trans_2", trans_2.shape)
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        print("concat_2", concat_2.shape)
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        print("up_2", up_2.shape)
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        print("trans_3", trans_3.shape)
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        print("concat_3", concat_3.shape)
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        print("up_3", up_3.shape)
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        print("trans_4", trans_4.shape)
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        print("concat_4", concat_4.shape)
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        print("up_4", up_4.shape)
        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        print("trans_5", trans_5.shape)
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        print("concat_5", concat_5.shape)
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        print("up_5", up_5.shape)
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        out = self.last(out)
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # image_size = 128
    # x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x = torch.Tensor(1, 1, 64, 64, 8)
    x.to(device)
    print("x size: {}".format(x.size()))
    #
    model = UNet(in_dim=1, out_dim=1, num_filters=2)
    model.apply(weights_init)
    # print(model)
    out = model(x)
    # lastConv = nn.Conv3d(1, 1, kernel_size=(3, 3, 1), stride=(1, 1, 1))
    # out = lastConv(out)
    print("out size: {}".format(out.size()))

if __name__ != '__main__':
    conv2 = conv_block_2_3d(1, 2, nn.LeakyReLU(0.2, inplace=True))
    # input = torch.randn(2, 1, 64, 64, 8)
    # out = conv2(input)
    # print(out.shape)

    # convT = conv_trans_block_3d(5, 1, nn.ReLU())

    # out = convT(input)
    # pool1 = max_pooling_3d()
    # out = pool1(input)
    # print(out.shape)
    # print(conv2)
    x = torch.Tensor(32, 32, 2, 2, 3).to(device)
    # down_5->'torch.Size([1, 16, 4, 4, 4])'
    trans_1 = conv_trans_block_3d(32, 32, activation=nn.LeakyReLU(0.2, inplace=True)).to(device)
    print(trans_1)
    out = trans_1(x)
    print(out.shape)

if __name__ != '__main__':
    # print(nn.Conv3d.bias.data)
    torch.manual_seed(10)
    # A = nn.Conv3d(1, 1, 3)
    # # print(A.weight.mean())
    # A.weight = init.kaiming_normal_(())
    # print(A.weight.mean())
    w = torch.empty(3, 5)
    # nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='leaky_relu')
    w = nn.BatchNorm3d(5)
    w.weight.data.fill_(1)
    print(w)