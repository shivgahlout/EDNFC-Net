
import torch
import torch.nn as nn


class bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=16, upsample=False, downsample=False):
        super(bottleneck, self).__init__()
        self.conv=nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.upsample=upsample
        self.downsample=downsample

    def forward(self,x):
        if self.downsample:
            x=nn.functional.max_pool2d(x,2,2)
        elif self.upsample:
            x=nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            
        x=nn.functional.relu(self.conv(x))
        return x

class FCC(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, downsample=False, upsample=False, feature_reuse=False):
        super(FCC, self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels,3,1,1)
        self.conv2=nn.Conv2d(out_channels, out_channels,3,1,1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
        self.upsample=upsample
        self.feature_reuse=feature_reuse

    def forward(self, x_pooled, x_reuse):
        x1=nn.functional.relu(self.bn1(self.conv1(x_pooled)))
        x2=nn.functional.relu(self.bn2(self.conv2(x1)))

        x=torch.cat((x1,x2),1)

        if self.feature_reuse:
            for x_reuse_ in x_reuse:
                x=torch.cat((x,x_reuse_),1)

        if self.downsample:
            x_pooled=nn.functional.max_pool2d(x,2,2)

            return x_pooled, x
        
        elif self.upsample:
            x_pooled=nn.functional.interpolate(x, scale_factor=2, mode='bilinear')

            return x_pooled, x

        return x

class UpsampleAndConcat(nn.Module):
    def __init__(self, scale_factor=2):
        super(UpsampleAndConcat, self).__init__()
        self.scale_factor=scale_factor

    def forward(self, x_1, x_2):
        x_1=nn.functional.interpolate(x_1,scale_factor=self.scale_factor, mode='bilinear')
        x=torch.cat((x_1,x_2),1)

        return x


class EDNFC(nn.Module):
    def __init__(self, in_channels=3, n_classes=1,out_channels=64, bottleneck_channels=16):
        """ in_channels --> no of input channels
        out_channels --> no of channels in first FCC. The remaining FCCs will have multiples of out_channels
        n_classes --> no of output classes
        bottleneck_channels --> inter-FCC bottleneck """
        
        super(EDNFC, self).__init__()
        self.encoder_block_1=FCC(in_channels,out_channels,downsample=True)
        in_channels=out_channels*2
        self.bottleneck_1_1=bottleneck(in_channels, bottleneck_channels, downsample=True)
        self.bottleneck_1_2=bottleneck(bottleneck_channels, bottleneck_channels,downsample=True)
        out_channels*=2
        self.encoder_block_2=FCC(in_channels,out_channels,downsample=True, feature_reuse=True)
        in_channels=out_channels*2+bottleneck_channels
        out_channels*=2
        self.bottleneck_2_1=bottleneck(in_channels, bottleneck_channels, downsample=True)
        self.bottleneck_2_2=bottleneck(bottleneck_channels, bottleneck_channels,downsample=True)
        self.encoder_block_3=FCC(in_channels,out_channels,downsample=True,feature_reuse=True)
        in_channels=out_channels*2+bottleneck_channels*2
        out_channels*=2
        self.bottleneck_3_1=bottleneck(in_channels, bottleneck_channels, downsample=True)
        self.bottleneck_3_2=bottleneck(bottleneck_channels, bottleneck_channels,downsample=True)
        self.encoder_block_4=FCC(in_channels,out_channels,downsample=True, feature_reuse=True)
        in_channels=out_channels*2+bottleneck_channels*3
        out_channels*=2
        self.bottleneck_4_1=bottleneck(in_channels, bottleneck_channels, downsample=True)
        self.encoder_block_5=FCC(in_channels,out_channels,downsample=False, feature_reuse=True)


        self.upsample_and_concat=UpsampleAndConcat()

        out_channels/=2
        in_channels=(out_channels*6)+(bottleneck_channels*7)
        self.decoder_block_1=FCC(int(in_channels),int(out_channels),upsample=True)
        self.bottleneck_4_1_up=bottleneck(int(out_channels*2), bottleneck_channels, upsample=True)
        self.bottleneck_4_2_up=bottleneck(bottleneck_channels, bottleneck_channels,upsample=True)
        out_channels/=2
        in_channels=(out_channels*6)+(bottleneck_channels*2)
        self.decoder_block_2=FCC(int(in_channels),int(out_channels),upsample=True, feature_reuse=True)
        self.bottleneck_3_1_up=bottleneck(int(out_channels*2+bottleneck_channels), bottleneck_channels, upsample=True)
        self.bottleneck_3_2_up=bottleneck(bottleneck_channels, bottleneck_channels, upsample=True)
        out_channels/=2
        in_channels=(out_channels*6)+(bottleneck_channels*2)
        self.decoder_block_3=FCC(int(in_channels),int(out_channels),upsample=True, feature_reuse=True)
        self.bottleneck_2_1_up=bottleneck(int(out_channels*2+2*bottleneck_channels), bottleneck_channels,upsample=True)
        self.bottleneck_2_2_up=bottleneck(bottleneck_channels, bottleneck_channels, upsample=True)
        out_channels/=2
        in_channels=(out_channels*6)+(bottleneck_channels*2)
        self.decoder_block_4=FCC(int(in_channels),int(out_channels),upsample=True, feature_reuse=True)
        in_channels=(out_channels*2)+(bottleneck_channels*3)
        self.output_layer=nn.Conv2d(int(in_channels),n_classes,3,1,1)

    
    def forward(self,x):
        x_pooled_1, x_1=self.encoder_block_1(x,x)
        x_1_1=self.bottleneck_1_1(x_1)
        x_pooled_2, x_2=self.encoder_block_2(x_pooled_1, [x_1_1])

        x_1_2=self.bottleneck_1_2(x_1_1)
        x_2_1=self.bottleneck_2_1(x_2)
        x_pooled_3, x_3=self.encoder_block_3(x_pooled_2,[x_1_2,x_2_1])

        x_1_3=self.bottleneck_1_2(x_1_2)
        x_2_2=self.bottleneck_2_2(x_2_1)
        x_3_1=self.bottleneck_3_1(x_3)
        x_pooled_4, x_4=self.encoder_block_4(x_pooled_3,[x_1_3,x_2_2,x_3_1])
 
        x_1_4=self.bottleneck_1_2(x_1_3)
        x_2_3=self.bottleneck_2_2(x_2_2)
        x_3_2=self.bottleneck_3_2(x_3_1)
        x_4_1=self.bottleneck_4_1(x_4)
        x_5=self.encoder_block_5(x_pooled_4,[x_1_4,x_2_3,x_3_2,x_4_1])

        x_5_upsampled=self.upsample_and_concat(x_5,x_4)
        _, x_4_upsampled=self.decoder_block_1(x_5_upsampled,x_5_upsampled)
        x_4_upsampled_=self.upsample_and_concat(x_4_upsampled,x_3)
        x_4_1_up=self.bottleneck_4_1_up(x_4_upsampled)

        _, x_3_upsampled=self.decoder_block_2(x_4_upsampled_, [x_4_1_up])
        x_3_upsampled_=self.upsample_and_concat(x_3_upsampled,x_2)
        x_4_2_up=self.bottleneck_4_2_up(x_4_1_up)
        x_3_1_up=self.bottleneck_3_1_up(x_3_upsampled)

        _, x_2_upsampled=self.decoder_block_3(x_3_upsampled_, [x_4_2_up, x_3_1_up])
        x_2_upsampled_=self.upsample_and_concat(x_2_upsampled,x_1)
        x_4_3_up=self.bottleneck_4_2_up(x_4_2_up)
        x_3_2_up=self.bottleneck_3_2_up(x_3_1_up)
        x_2_1_up=self.bottleneck_2_1_up(x_2_upsampled)

        _, x_1_upsampled=self.decoder_block_4(x_2_upsampled_, [x_4_3_up, x_3_2_up, x_2_1_up])

        output_mask=self.output_layer(x_1_upsampled)

        return output_mask

    
