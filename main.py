import torch 
import torch.nn as nn 

'''
|Encoding : 
    During encoding we find the important features of image 
    
|Decoding : 
    During Decoding we use that features to construct segmented image 
    
|skip connection : 
    We use skip connection to forward the encoder blocks inforamtion to Decoder 
    blocks which help the decoders to remember the every detail captured by encoder.This helps
    decoder to construct detailed and accurate segmented image 
    
    ## In U-net during skip connection we use concatination instead of adding (these are different from Resnet skip connections)

|steping in Encoder path :
    Every time we move one step down during encoding we use max_pool to reduce image resolution
    
|steping in Decoder path :
    When we move one step above in Decoder block we use Transpose convolution, We construct 
    segmented image in decoding process using the high level details of bottleneck volume and
    low level details from the encoder blocks using the skip connections

|Why volume height and width decreases but the depth increase as we move down ?
    * small volume in shallow layers because shallow layeres contain low level inforamtion about image like edges,corners,straight lines,faces,doors location
    * But as we move down deeper layers use low level information to detect high level features in image like what image is telling
    (why smiley face,why balck door) or say intelligent details  
    * high depth volumes represent that we contain detailed information | High Depth => high details (what and why) |
    * Low depth volumes represent that we contain low level information | Low depth => low details (where)|
    
    ## thats why the bottleneck layer contain most important details of image (intelligent information)

'''

class U_NET(nn.Module):
    
    def block(self,in_c,out_c):
        
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,padding=1,bias=False),
            nn.ReLU(),
        )
     
    def __init__(self):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2,2)
        
        # Enocder path 
        self.e1 = self.block(1,64)
        self.e2 = self.block(64,128)
        self.e3 = self.block(128,256)
        self.e4 = self.block(256,512)
        
        # bottle neck 
        self.bottle_neck = self.block(512,1024)

        # Decoder Path 
        # In decoder path when we move one step up we will use transpose convolution 
        self.d1u = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.d1 = self.block(1024,512) # 1024 => (512 skip connection) and (512 original)
        
        self.d2u = nn.ConvTranspose2d(512,256,2,stride=2)
        self.d2 = self.block(512,256)
        
        self.d3u = nn.ConvTranspose2d(256,128,2,stride=2)
        self.d3 = self.block(256,128)
        
        self.d4u = nn.ConvTranspose2d(128,64,2,stride=2)
        self.d4 = self.block(128,64)
        
        self.final = nn.Conv2d(64,2,1)
        
    def forward (self,x):
        # encoding
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        # bottle neck
        bottle_neck = self.bottle_neck(self.pool(e4))
        
        # Decoding 
        d1u = self.d1u(bottle_neck)
        # first we join skip connection tensor and transpose conv tensor and then put into d1
        d1 = self.d1(torch.cat([e4,d1u],dim=1)) # skip connection | [batch,channel,height,width] dim=1 means concatenate along channels i.e (256+256) resulting channels become 512
        
        d2u = self.d2u(d1)
        d2 = self.d2(torch.cat(e3,d2u),dim=1)
        
        d3u = self.d3u(d2)
        d3 = self.d3(torch.cat([e2,d3u],dim=1))
        
        d4u = self.d4(d3)
        d4 = self.d4(torch.cat([e1,d4u],dim=1))
        # puting output into 1x1 convolution 
        out = self.final(d4)
        
        return out
        
