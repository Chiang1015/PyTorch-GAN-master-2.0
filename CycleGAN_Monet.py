# This model's training is heavy loading for household device, recommend to put it to Kaggle or Colab
# I put trained model in file 'models', use them
import itertools
import argparse
import random
import torch
import glob
import os
import datetime
import time

import torch.nn as nn

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=199, help="epoch to start training from") # If you want to retrain, default = 0

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

input_shape = (opt.channels, opt.img_height, opt.img_width) #(3,256,256)
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

#(real_A, fake_B, real_B, fake_A)由上而下
def sample_images(imgtag):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader)) #batch=5
    G_AB.eval()
    G_BA.eval()
    with torch.inference_mode():
        real_A = imgs["A"].to(dtype=torch.float32,device=device)
        fake_B = G_AB(real_A)
        real_B = imgs["B"].to(dtype=torch.float32,device=device)
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True,scale_each=True) #=>(3,256,256*5) remove 'batch'
        real_B = make_grid(real_B, nrow=5, normalize=True,scale_each=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True,scale_each=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True,scale_each=True)
    G_AB.train()
    G_BA.train()
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid,f'./samples/CycleGAN_{imgtag}.png', normalize=False)

#input:(b,in,H,W) => output:(b,in,H,W)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),#(H+2,w+2)
            nn.Conv2d(in_features, in_features, 3),#=>(H,w)
            nn.InstanceNorm2d(in_features),

            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features) )
    def forward(self, x):
        return x + self.block(x)
    
#input(b,3,256,256) =>(b,3,256,256)
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0] #3
        out_features = 64

        model = [
            nn.ReflectionPad2d(channels), #(256+2*3)
            nn.Conv2d(channels, out_features, 7), #=>(batch,64,256,256)
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)  ]
        in_features = out_features

        # Downsampling(2)
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ] #(batch,64,256,256) =>(batch,128,128,128)=>(batch,256,64,64)
            in_features = out_features

        # Residual blocks(N)
        # in:(batch,256,64,64) => out:(batch,256,64,64)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling(2)
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]#(batch,256,64,64) =>(batch,128,128,128)=>(batch,64,256,256)
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        # =>(batch,3,256,256)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#input(b,3,256,256) => output:(b,1,16,16) no sig
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape #(3,256,256)

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        #(1,16,16)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),#(batch,512,16,16)
            nn.ZeroPad2d((1, 0, 1, 0)),#(left, right, top, bottom)(batch,512,17,17)
            nn.Conv2d(512, 1, 4, padding=1)#(batch,1,16,16)
        )

    def forward(self, img):
        return self.model(img)

#Learning Rate(linear decay)
class LambdaLR_fn:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the end of training!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# input:shape[B, C, H, W] => output:shape[B, C, H, W]
# input/output的batch 相同但是output元素可能被舊資料替代
class ReplayBuffer:
    def __init__(self, max_size: int = 50):
        assert max_size > 0, "Buffer size must be positive."
        self.max_size = max_size
        self.data = [] #候選名單
    #batch: torch.Tensor) -> torch.Tensor:表示傳進Tensor，傳出Tensor(僅註解沒有實際檢查功能，不會報錯)
    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        to_return = [] #每次執行都清空為0
        for element in batch:
            element = element.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    old = self.data[i].clone()#從候選抽一
                    self.data[i] = element #候選更新
                    to_return.append(old) 
                else:
                    to_return.append(element)

        return torch.cat(to_return, dim=0)  # Return [B, C, H, W]


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)#新空白圖RGB模式,大小同原圖
    rgb_image.paste(image)#原圖貼上
    return rgb_image
    # return image.convert("RGB") 同等功能

#return {"A": item_A, "B": item_B}
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))#file A
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))#file B

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])#A必循環
        #因為A是畫，數量有限
        #B是真實照片，數量多，預設不循環
        if self.unaligned: #不循環
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


#dataset
transform_seq = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC), #變大
    transforms.RandomCrop((opt.img_height, opt.img_width)), #裁切
    transforms.RandomHorizontalFlip(), #隨機翻轉
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transform_seq, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0)
val_dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transform_seq, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=0)


#object
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks).to(device)
D_A = Discriminator(input_shape).to(device)
D_B = Discriminator(input_shape).to(device)
#loss
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_identity = torch.nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#Learning rate(linear decay to 0)
#epoch: start from / n_epoch: max round
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR_fn(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR_fn(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR_fn(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

#假資料重播緩衝區:讓判別器不要只看到最新的生成器輸出，而是可以看到一段時間內的歷史假樣本
fake_A_buffer = ReplayBuffer() #store A type
fake_B_buffer = ReplayBuffer() #store B type

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("./models/Cycle_G_AB_%d.pth" % (opt.epoch)))
    G_BA.load_state_dict(torch.load("./models/Cycle_G_BA_%d.pth" % (opt.epoch)))
    D_A.load_state_dict(torch.load("./models/Cycle_D_A_%d.pth" % (opt.epoch)))
    D_B.load_state_dict(torch.load("./models/Cycle_D_B_%d.pth" % (opt.epoch)))
else:
    #for first time
    G_AB.apply(custom_weights_init)
    G_BA.apply(custom_weights_init)
    D_A.apply(custom_weights_init)
    D_B.apply(custom_weights_init)

for epoch in range(opt.epoch+1, opt.n_epochs):
    prev_time = time.time() 
    for i, batch in enumerate(dataloader):
        real_A = batch["A"].to(dtype=torch.float32,device=device)
        real_B = batch["B"].to(dtype=torch.float32,device=device)

        valid = torch.ones((real_A.size(0),*D_A.output_shape),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((real_B.size(0),1,16,16),dtype = torch.float32 , device=device ,requires_grad=False)
        # ------------------
        #  Train Generators
        # ------------------
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss(1*GAN loss  +  10*Cycle loss + 5*Identity)
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

    # Update learning rates(lr_scheduler內建計數器epoch並傳至lr_lambda指定的函數中)
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if epoch % 2 ==0 :
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "./models/G_AB_%d.pth" % (epoch))
        torch.save(G_BA.state_dict(), "./models/G_BA_%d.pth" % (epoch))
        torch.save(D_A.state_dict(), "./models/D_A_%d.pth" % (epoch))
        torch.save(D_B.state_dict(), "./models/D_B_%d.pth" % (epoch))

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {loss_D.item():.4f}][G loss: {loss_G.item():.4f}]",end='')
    print(f"[GAN loss: {loss_GAN.item():.4f}][CYC loss: {loss_cycle.item():.4f}][IDE loss: {loss_identity.item():.4f}]--{time1round}")

    if epoch %50 == 0 or epoch ==199:
        sample_images(imgtag=epoch)

for i in range(5):
    sample_images(imgtag=f'final_{i}')
   