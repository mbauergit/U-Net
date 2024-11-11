import torch
import torch.nn as nn

# Standard double convultional network used in down and upsampling
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=0),
        nn.ReLU(inplace=True)
    )

# Function to crop outputs from encoder side to be concatenated to inputs on decoder side
def crop_image(orig_tensor, target_tensor):
    target_size = target_tensor.size()[-1]
    original_size = orig_tensor.size()[-1]
    diff = original_size - target_size
    diff = diff // 2
    return orig_tensor[:, :, diff:original_size-diff, diff:original_size-diff]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        self.up_conv1 = double_conv(1024, 512)
        self.up_conv2 = double_conv(512, 256)
        self.up_conv3 = double_conv(256, 128)
        self.up_conv4 = double_conv(128, 64)
        self.up_conv5 = double_conv(64, 1)

        self.up_trans1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_trans3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_trans4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.last_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1), padding=0)

    def forward(self, input):
        # Encoder
        x1 = self.down_conv1(input) #
        x2 = self.max_pool_2x2(x1)
        x2 = self.down_conv2(x2) #
        x3 = self.max_pool_2x2(x2)
        x3 = self.down_conv3(x3) #
        x4 = self.max_pool_2x2(x3)
        x4 = self.down_conv4(x4) #
        x5 = self.max_pool_2x2(x4)
        x5 = self.down_conv5(x5)

        # Decoder
        x6 = self.up_trans1(x5)
        x4 = crop_image(x4, x6)
        x6 = torch.cat((x4, x6), dim=1)
        x6 = self.up_conv1(x6)

        x7 = self.up_trans2(x6)
        x3 = crop_image(x3, x7)
        x7 = torch.cat((x3, x7), dim=1)
        x7 = self.up_conv2(x7)

        x8 = self.up_trans3(x7)
        x2 = crop_image(x2, x8)
        x8 = torch.cat((x2, x8), dim=1)
        x8 = self.up_conv3(x8)

        x9 = self.up_trans4(x8)
        x1 = crop_image(x1, x9)
        x9 = torch.cat((x1, x9), dim=1)
        x9 = self.up_conv4(x9)

        res = self.last_conv(x9)
        print(res.size())

if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    unet = UNet()
    unet(image)