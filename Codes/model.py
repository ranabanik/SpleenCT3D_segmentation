# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn


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
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        
        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # image_size = 128
    # x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x = torch.Tensor(1, 1, 64, 64, 8)
    x.to(device)
    print("x size: {}".format(x.size()))
    #
    model = UNet(in_dim=1, out_dim=1, num_filters=1)
    # print(model)
    out = model(x)
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