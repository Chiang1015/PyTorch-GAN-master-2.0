#修正CrossEntropyLoss用法:(no sig , Number)
#z範圍變更為-1~1
import torch
import argparse
import itertools
import time
import datetime

from torchvision import datasets #現成的資料集庫
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_images(imgtag):
    generator.eval()
    # vary z
    n_row = 10
    z = torch.randn((n_row**2, opt.latent_dim), device=device)

    with torch.inference_mode():
        static_sample = generator(z, static_label, static_code)   
        # Get varied c1 and c2
        steps = torch.linspace(-1, 1, n_row, device=device).repeat_interleave(n_row).view(-1, 1)  # shape: (n_row^2, 1)
        zeros = torch.zeros_like(steps, device=device)  # shape: (n_row^2, 1)

        # 控制變數：c1 與 c2
        c1 = torch.cat((steps, zeros), dim=1)  # 第1維變，第2維為0
        c2 = torch.cat((zeros, steps), dim=1)  # 第2維變，第1維為0
        sample1 = generator(static_z, static_label, c1)
        sample2 = generator(static_z, static_label, c2)

    save_image(static_sample.data, f'./samples/infoGAN_{imgtag}_z.png', nrow=n_row, normalize=True) # 隨機z , 左到右0~9 , 全0
    save_image(sample1.data, f'./samples/infoGAN_{imgtag}_c1.png', nrow=n_row, normalize=True)      # 全0   , 左到右0~9 ,左到右相同/上到下遞增
    save_image(sample2.data, f'./samples/infoGAN_{imgtag}_c2.png', nrow=n_row, normalize=True)      # 全0   , 左到右0~9 ,左到右相同/上到下遞增
    
    generator.train()

# (label,opt.n_classes,device,torch.float32)
def to_categorical(y, num_classes=None, device=device, dtype=torch.float32):   
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:
        y = y.to(device=device)

    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    return F.one_hot(y, num_classes=num_classes).to(dtype=dtype)

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



#input:(batch,1,62+10+2)=> img(batch,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim #62+10+2=74

        self.init_size = opt.img_size // 4  # 8
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), dim=-1) #(batch,1,74)
        out = self.l1(gen_input) #(batch,1,128*8*8)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) #(batch,128,8,8)
        img = self.conv_blocks(out) #(batch,1,32,32)
        return img
    
#input(batch,1,32,32)=> val(batch,1) / label(batch,10)/ latent_code(batch,2)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4 #2

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes))
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img) #(batch,128,2,2)
        out = out.view(out.shape[0], -1)#(batch,128*2*2)
        #同時收束
        validity = self.adv_layer(out)#(batch,1)
        label = self.aux_layer(out)#(batch,10)
        latent_code = self.latent_layer(out)#(batch,2)

        return validity, label, latent_code

#object
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#Loss
adversarial_loss = torch.nn.MSELoss().to(device)
categorical_loss = torch.nn.CrossEntropyLoss().to(device)
continuous_loss = torch.nn.MSELoss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#special design for infoGAN
optimizer_info = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader1 = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)

n= opt.n_classes
static_label = to_categorical([i for _ in range(n) for i in range(n)],opt.n_classes,device,torch.float32)
static_z = torch.zeros((n * n, opt.latent_dim), device=device)
static_code = torch.zeros((n * n, opt.code_dim), device=device)

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize weights
generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)

for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, labels) in enumerate(dataloader1):
        batch_size = imgs.shape[0]
        valid = torch.ones((batch_size,1),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((batch_size,1),dtype = torch.float32 , device=device ,requires_grad=False)

        real_imgs = imgs.to(dtype=torch.float32,device=device)
        labels = to_categorical(labels,opt.n_classes,device,torch.float32)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn((batch_size, opt.latent_dim), device=device) #(batch,62)

        z10 = torch.randint(0,opt.n_classes,(batch_size,),device=device) 
        label_input = to_categorical(z10,opt.n_classes,device,torch.float32) #(batch,10)
        code_input = torch.randn((batch_size,opt.code_dim),device=device)

        gen_imgs = generator(z, label_input, code_input)
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid) #MSE

        g_loss.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_pred, _, _ = discriminator(real_imgs)
        fake_pred, _, _ = discriminator(gen_imgs.detach())

        d_real_loss = adversarial_loss(real_pred, valid)
        d_fake_loss = adversarial_loss(fake_pred, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
       
        # ------------------
        # Information Loss
        # ------------------
        optimizer_info.zero_grad()
        z = torch.randn((batch_size, opt.latent_dim), device=device)
        z10 = torch.randint(0,opt.n_classes,(batch_size,),device=device)

        label_input = to_categorical(z10,opt.n_classes,device,torch.float32)
        code_input = torch.randn((batch_size,opt.code_dim),device=device)
        
        gt_labels = z10.to(dtype=torch.long,device=device)

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels)+ lambda_con * continuous_loss( pred_code, code_input)
        info_loss.backward()
        optimizer_info.step()

    time1round = datetime.timedelta(seconds = time.time() - prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}]",end='')
    print(f'[info_loss: {info_loss.item():.4f}]--{time1round}')

    if epoch % 50 ==0 or epoch ==199:
        sample_images(imgtag=epoch)

        







