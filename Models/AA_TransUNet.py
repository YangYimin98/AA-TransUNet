"""AA_TransUNet.py"""

"""
Ref: https://github.com/The-AI-Summer/self-attention-cv
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Built AA_TransUNet
"""

from einops import rearrange
from bottleneck import *
from Precipitation_Forecasting.precipitation_lightning_base import Precip_regression_base
from encoder import ViT
from decoder import SingleConv, UpDS
from cbam import *


class AA_TransUnet(Precip_regression_base):
    def __init__(self, hparams
                 ):
        """
        Args:
            img_dim: the img dimension
            in_channels: channels of the input
            classes: desired segmentation classes
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
            vit_transformer: pass your own version of vit
            vit_channels: the channels of your pretrained vit. default is 128*8
            patch_dim: for image patches of the vit
        """
        super(AA_TransUnet, self).__init__(hparams=hparams)

        self.inplanes = 128
        self.patch_size = hparams['patch_size']
        self.vit_transformer_dim = hparams['vit_transformer_dim']
        vit_channels = self.inplanes * 8 if hparams['vit_channels'] is None else hparams['vit_channels']

        in_conv1 = nn.Conv2d(hparams['in_channels'], self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)

        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.CBAM6 = CBAM(256)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.CBAM7 = CBAM(512)

        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)
        self.CBAM8 = CBAM(1024)

        self.img_dim_vit = hparams['img_dim'] // 16

        assert (self.img_dim_vit % hparams['patch_size'] == 0), "Vit patch_dim not divisible"
        #
        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # input features' channels (encoder)
                       patch_dim=hparams['patch_size'],
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=hparams['vit_transformer_dim'],
                       blocks=hparams['vit_blocks'],
                       heads=hparams['vit_heads'],
                       dim_linear_block=hparams['vit_dim_linear_mhsa_block'],
                       classification=False) if hparams['vit_transformer'] is None else hparams['vit_transformer']

        # to project patches back - undoes vit's patchification
        token_dim = vit_channels * (hparams['patch_size'] ** 2)
        self.project_patches_back = nn.Linear(hparams['vit_transformer_dim'], token_dim)
        # upsampling path
        self.vit_conv = SingleConv(in_ch=vit_channels, out_ch=512)
        self.cbam1 = CBAM(512)
        self.dec1 = UpDS(vit_channels, 256)
        self.cbam2 = CBAM(256)
        self.dec2 = UpDS(512, 128)
        self.cbam3 = CBAM(128)
        self.dec3 = UpDS(256, 64)
        self.cbam4 = CBAM(64)
        self.dec4 = UpDS(64, 16)
        self.cbam5 = CBAM(16)
        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=hparams['classes'], kernel_size=1)

    def forward(self, x):
        # ResNet 50-like encoder
        x2 = self.init_conv(x)
        x4 = self.conv1(x2)
        x4 = self.CBAM6(x4)
        x8 = self.conv2(x4)
        x8 = self.CBAM7(x8)
        x16 = self.conv3(x8)  # out shape of 1024, img_dim_vit, img_dim_vit
        x16 = self.CBAM8(x16)

        y = self.vit(x16)  # out shape of number_of_patches, vit_transformer_dim

        # from [number_of_patches, vit_transformer_dim] -> [number_of_patches, token_dim]
        y = self.project_patches_back(y)

        # from [batch, number_of_patches, token_dim] -> [batch, channels, img_dim_vit, img_dim_vit]
        y = rearrange(y, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=self.img_dim_vit // self.patch_size, y=self.img_dim_vit // self.patch_size,
                      patch_x=self.patch_size, patch_y=self.patch_size)

        y = self.vit_conv(y)
        y = self.cbam1(y)
        y = self.dec1(y, x8)
        y = self.cbam2(y)
        y = self.dec2(y, x4)
        y = self.cbam3(y)
        y = self.dec3(y, x2)
        y = self.cbam4(y)
        y = self.dec4(y)
        y = self.cbam5(y)

        return self.conv1x1(y)