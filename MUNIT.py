#Multimodal UNsupervised Image-to-image Translation
#不需要成對數據（unpaired data）的情況下，將一張圖像從一個風格或領域轉換成另一個
import os
import time
import torch
import datetime
import argparse
import itertools

import glob
import time
import datetime

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from PIL import Image

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# original dataset: 'edges2shoes'
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")#!!#
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d,nn.BatchNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def sample_images(imgtag):
    imgs = next(iter(val_dataloader))
    img_samples = None
    Enc1.eval()
    Dec2.eval()
    with torch.inference_mode():
        for img1, img2 in zip(imgs["A"], imgs["B"]):
            X1 = img1.unsqueeze(0).repeat(8, 1, 1, 1)  #(3,h,w) =>(8,3,h,w)
            X1 = X1.to(dtype=torch.float32,device=device)
            #隨機風格 因為MLP會展平,所以送(b,8,1,1)/(b,8)都行
            s_code = torch.empty((8, opt.style_dim), device=device).uniform_(-1, 1)
            c_code_1, _ = Enc1(X1)

            X12 = Dec2(c_code_1, s_code)
            X12 = torch.cat([x for x in X12.data.cpu()], -1) # 8 into 1
            img_sample = torch.cat((img1, X12), -1).unsqueeze(0) #save_image ask (b,c,h,w)
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples,f'./samples/MUNIT_{imgtag}.png', normalize=True)
    Enc1.train()
    Dec1.train()


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# input:style(b,c,h,w) => (b,output_dim)
#(Decoder引用)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

#(num_features,affine)
#用途:先對每個樣本做正規化(均值=0、標準差=1)，然後如果 affine=True，用可學習的gamma/beta把它轉換回去某種新風格
# 解碼器穩定訓練
#(Decoder引用)
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            #可訓練參數
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())#標準差縮放因子
            self.beta = nn.Parameter(torch.zeros(num_features))#偏移量

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1) # [-1] + [1]*(4-1)=[-1,1,1,1] /-1表示自動
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2) #[1,-1,1,1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


#(num_features)
#外部需要指定.weight/.bias兩數才能使用(由Decoder中assign_adain_params提供)
#用途:風格移植，把內容圖的 feature 調成指定風格的樣子
#(ResidualBlock引用)
class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # 模型需要但不訓練 的資料('name',tensor)
        # 實際計算不使用，而使用weight/bias
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # x.contiguous()：確保 x 在記憶體中是連續儲存的
        # 把這個batch的所有通道都視為同一物件的貢獻
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        #兩步驟函數:
        # 1.先減去 x_reshaped 自己算出來的 mean，再除以它自己的 std
        # 2.再乘上 self.weight(風格 std)加上 self.bias(風格 mean)
        # running_變數是為了滿足.batch_norm的要求製作的
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )
        #.batch_norm( input, 不使用 , 不使用, 風格的std(手動),風格的mean(手動),強制使用輸入而非running stats, 因為不使用running所以也不使用, 避免除以0)
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

#(features,norm='in/adain')  / 這裡features會建立adapt...中的num_features
#(ContentEncoder/Decoder引用)
class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),#可視為變更Conv2d的padding方式
            nn.Conv2d(features, features, 3),#padding=0
            norm_layer(features),

            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )
    def forward(self, x):
        return x + self.block(x)

#input(b,c,h,w) => (b,256,h/4,w/4)
#(Encoder引用)
class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]#(b,64,h,w)

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2
        #(b,256,h/4,w/4)

        # Residual blocks(卷積+保留)
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#input(b,c,h,w) => (b,8,1,1)
#(Encoder引用)
class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()

        # Initial conv block
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]

        # Average pool and output layer
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]
        #平均池化成 [B, 256, 1, 1] => 1X1卷積
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#input:img(b,c,h,w) => content_(b,256,h/4,w/4) and style_(b,8,1,1)
class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super(Encoder, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)

    def forward(self, x):
        content_code = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return content_code, style_code

#input:content_code, style_code => img(b,c,h,w)
#用途: 賦予content_codeu 一個style,再上採樣到h,w
# kernal_size漸大: 小核讓結果'局部細節誇張',大核讓結果'平滑過渡、整體自然'
class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample #256
        # Residual blocks 
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")] #設定adpt...num_features=64

        # Upsampling (b,256,h/4,w/4)
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer(b,64,h,w)
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        #(b,3,)
        self.model = nn.Sequential(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params #Decoder中的adapt...設定64*2*層數

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():#給所有adapt層賦予mean/std(從style_code)
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer(如果adapt超過1層需要)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, content_code, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img

#call :compute_loss(x,gt)
#input: img, target => loss
class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = [] 
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs #(b,1,hw/16),(b,1,hw/32),(b,1,hw/64)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}
    def __len__(self):
        return len(self.files) 

#object (64,2,3,8)
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(device)
Dec1 = Decoder(dim=opt.dim, n_upsample  =opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(device)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(device)
Dec2 = Decoder(dim=opt.dim, n_upsample  =opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim).to(device)
D1 = MultiDiscriminator().to(device)
D2 = MultiDiscriminator().to(device)
#loss
criterion_recon = torch.nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),lr=opt.lr,betas=(opt.b1, opt.b2))
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#LR schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

transforms_seq = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transforms_seq),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)
val_dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transforms_seq, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

if opt.epoch != 0:
    # Load pretrained models
    Enc1.load_state_dict(torch.load(f"./models/MUNIT_enc1_{opt.epoch}",weights_only=True))
    Dec1.load_state_dict(torch.load(f"./models/MUNIT_dec1_{opt.epoch}",weights_only=True))
    Enc2.load_state_dict(torch.load(f"./models/MUNIT_enc2_{opt.epoch}",weights_only=True))
    Dec2.load_state_dict(torch.load(f"./models/MUNIT_dec2_{opt.epoch}",weights_only=True))
    D1.load_state_dict(torch.load(f"./models/MUNIT_d1_{opt.epoch}",weights_only=True))
    D2.load_state_dict(torch.load(f"./models/MUNIT_d2_{opt.epoch}",weights_only=True))
else:
    # Initialize weights
    Enc1.apply(custom_weights_init)
    Dec1.apply(custom_weights_init)
    Enc2.apply(custom_weights_init)
    Dec2.apply(custom_weights_init)
    D1.apply  (custom_weights_init)
    D2.apply  (custom_weights_init)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

valid = 1
fake = 0

for epoch in range(opt.epoch, opt.n_epochs):
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        X1 = batch['A'].to(dtype=torch.float32,device=device)
        X2 = batch['B'].to(dtype=torch.float32,device=device)
        style_1 = torch.empty((X1.size(0), opt.style_dim, 1, 1), device=device).uniform_(-1, 1)#(1,8,1,1)
        style_2 = torch.empty((X1.size(0), opt.style_dim, 1, 1), device=device).uniform_(-1, 1)

        # ---------------
        #  Train E and G
        # ---------------
        optimizer_G.zero_grad()
        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

        # Total loss
        loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------
        optimizer_D1.zero_grad()
        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()
        # -----------------------
        #  Train Discriminator 2
        # -----------------------
        optimizer_D2.zero_grad()
        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

        loss_D2.backward()
        optimizer_D2.step()

    time_1round = datetime.timedelta(minutes=time.time()-prev_time)
    print(f'[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {(loss_D1 + loss_D2).item():.4f}][G loss: {loss_G.item():.4f}]--{time_1round}')   
    
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if epoch % 50 == 0 or epoch == 199:
        sample_images(imgtag=epoch)

    if epoch % 199 == 0 :
        torch.save(Enc1.state_dict(), f"./models/MUNIT_enc1_{epoch}")
        torch.save(Dec1.state_dict(), f"./models/MUNIT_dec1_{epoch}")
        torch.save(Enc2.state_dict(), f"./models/MUNIT_enc2_{epoch}")
        torch.save(Dec2.state_dict(), f"./models/MUNIT_dec2_{epoch}")
        torch.save(D1.state_dict(), f"./models/MUNIT_d1_{epoch}")
        torch.save(D2.state_dict(), f"./models/MUNIT_d2_{epoch}")