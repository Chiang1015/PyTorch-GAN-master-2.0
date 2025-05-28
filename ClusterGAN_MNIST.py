import torch
import argparse
import datetime
import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch.autograd import grad as torch_grad
from itertools import chain as ichain

import numpy as np 
import torch.nn as nn
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
parser.add_argument("-i", "--img_size", dest="img_size", type=int, default=28, help="Size of image dimension")
parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=30, type=int, help="Dimension of latent space")
parser.add_argument("-l", "--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-w", "--wass_flag", dest="wass_flag", action='store_true', help="Flag for Wasserstein metric")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
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

#for Wasserstein GAN version
def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (b,c, h, w),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

#input:(batch,latent_dim,n_c,fix_class)=> 
#output: zn(b,30),zc_FT(b,10) ,zc_idx(b,1)
#原版錯誤設定latent_dim=10 => 30
def sample_z(batch=64, latent_dim=30, n_c=10, fix_class=-1, req_grad=False):
    #如fix_class=-1 or 0~9是合法範圍，超出就顯示後面訊息
    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c) ), f"Require fix_class:-1 or 0~9, {fix_class}given."
    
    # Sample noise(generator input):zn(batch,30)
    zn= torch.tensor(0.75*np.random.normal(0, 1, (batch, latent_dim)), dtype=torch.float32, device=device, requires_grad=req_grad)
    
    #類別:數字FT與one-hot(idx)
    zc_FT = torch.zeros((batch, n_c), dtype=torch.float32, device=device,requires_grad=req_grad)
    zc_idx = torch.empty(batch, dtype=torch.long,device=device)
    if (fix_class == -1): #無指定類別
        zc_idx = zc_idx.random_(n_c)#0~9 填滿(batch)
        #.scatter_(dim,position,value)
        #.unsqueeze(dim) => (batch,1)
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:#指定類別
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

    return zn, zc_FT, zc_idx

def sample_image(n_row, imgtag):
    #Save sample(from random)
    stack_imgs = []
    for idx in range(opt.n_classes):
        zn_samp, zc_samp, _ = sample_z(batch=opt.n_classes,latent_dim=30,n_c=opt.n_classes,fix_class=idx)#?
        gen_imgs_samp = generator(zn_samp, zc_samp)

        if (len(stack_imgs) == 0):#for 1st round
            stack_imgs = gen_imgs_samp
        else:
            stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)
    save_image(stack_imgs,f"./samples/ClusterGAN_{imgtag}.png", nrow=n_row, normalize=True)
    #Rebuild loss:從真實圖片經encoder再由generator重建的loss
    #Zn loss:從隨機數經generator再由encoder重建的loss(Zn:噪音)
    #Zc loss:從隨機數經generator再由encoder重建的loss(Zc:標籤資訊)
    

#input:zn(b,30),zc(b,10)=>img(b,1,28,28)
class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, n_c, x_shape):
        super(Generator_CNN, self).__init__()
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape #(1,28,28)
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        
        self.model_01 = nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),#30+10 =>1024
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels), #1024 => 128*7*7
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),       
        )
        
        self.model_02 = nn.Sequential( #input:(batch,128,7,7)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),#(batch,64,14,14)
            
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
            #(batch,1,28,28)
        )
           
    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model_01(z)
        x_gen = x_gen.view(x_gen.size(0), *self.ishape)
        x_gen = self.model_02(x_gen)

        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen

#imput:(b,1,28,28) => zn(b,30),zc(b,10) softmax版,zc_logits(batch,10)原版
class Encoder_CNN(nn.Module):
    def __init__(self, latent_dim, n_c):
        super(Encoder_CNN, self).__init__()
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5) 
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        
        self.model_01 = nn.Sequential(#(batch,1,28,28)
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),#(batch,128,5,5)
            nn.LeakyReLU(0.2, inplace=True),    
        )

        self.model_02 = nn.Sequential(
            torch.nn.Linear(self.iels, 1024),#(batch,128*7*7)=>(batch,1024)
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, latent_dim + n_c)#=>(batch,30+10)
        )

        self.model_soft = nn.Softmax(dim=1)
        
    def forward(self, in_feat):
        z_img = self.model_01(in_feat)
        z_img = z_img.view(z_img.size(0),-1)
        z_img = self.model_02(z_img)

        # Reshape for output
        z = z_img.view(z_img.size(0), -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim] #前30個數字

        zc_logits = z[:, self.latent_dim:] #後10個數字
        zc = self.model_soft(zc_logits) #softmax

        return zn, zc, zc_logits


#input:(b,1,28,28)=>val(b,1)
class Discriminator_CNN(nn.Module):         
    def __init__(self, wass_metric=False):
        super(Discriminator_CNN, self).__init__()
        self.channels = 1
        self.cshape = (128, 5, 5) # floor (28-4)/2 + 1 =13 => floor (13-4)/2 +1 = 5
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        
        
        self.model_01 = nn.Sequential(#(batch,1,28,28)
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),#(batch,128,7,7)
        )

        self.model_02 = nn.Sequential(
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 1),#(batch,1)
        )
        
        #Wasserstein metric:基於Earth Mover’s Distance，需要輸出真實度分數，不用機率
        if (not self.wass):
            self.model_02 = nn.Sequential(self.model_02, torch.nn.Sigmoid())

    def forward(self, img):
        img = self.model_01(img)
        img = img.view(img.size(0), -1)
        validity = self.model_02(img)
        return validity

n_c = 10
x_shape = (opt.channels, opt.img_size, opt.img_size)#(1,28,28)
#object
generator = Generator_CNN(opt.latent_dim, opt.n_classes, x_shape).to(device)#(30,10,(1,28,28))
encoder = Encoder_CNN(opt.latent_dim, opt.n_classes).to(device)#(30,10)
discriminator = Discriminator_CNN(wass_metric=False).to(device)
#Loss
bce_loss = torch.nn.BCELoss().to(device)
xe_loss = torch.nn.CrossEntropyLoss().to(device)
mse_loss = torch.nn.MSELoss().to(device)
#optimizers
decay = 2.5*1e-5
ge_chain = ichain(generator.parameters(),encoder.parameters())
optimizer_GE = torch.optim.Adam(ge_chain, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#initial weight
generator.apply(custom_weights_init)
encoder.apply(custom_weights_init)
discriminator.apply(custom_weights_init)

#Dataloader(Train/Test)
transform_seq = transforms.Compose([transforms.ToTensor()])

train_mnist_data = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq)
train_dataloader = DataLoader(train_mnist_data, batch_size=opt.batch_size, shuffle=True)

test_mnist_data = datasets.MNIST("./Resorces", train=False, download=False, transform=transform_seq)
test_dataloader = DataLoader(train_mnist_data, batch_size=opt.batch_size, shuffle=True)
test_imgs, test_labels = next(iter(test_dataloader))
test_imgs = test_imgs.to(dtype=torch.float32,device=device)


n_skip_iter = 5 #延遲G/E訓練
betan = 10
betac = 10
for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, itruth_label) in enumerate(train_dataloader):
        generator.train()
        encoder.train()

        generator.zero_grad()
        encoder.zero_grad()
        discriminator.zero_grad()

        real_imgs = imgs.to(dtype=torch.float32,device=device)
        valid = torch.ones((real_imgs.size(0), 1), dtype=torch.float32, device=device,requires_grad=False)
        fake = torch.zeros((real_imgs.size(0), 1), dtype=torch.float32, device=device,requires_grad=False)

        # ---------------------------
        #  Train Generator + Encoder
        # ---------------------------
        optimizer_GE.zero_grad()
        #zn(batch,30),zc_FT(batch,10) ,zc_idx(batch,1)
        zn, zc_FT, zc_idx = sample_z(batch=real_imgs.size(0),latent_dim=opt.latent_dim,n_c=opt.n_classes)      
        gen_imgs = generator(zn, zc_FT)
        D_gen = discriminator(gen_imgs)
        
        #encoder training less freqent
        if (i % n_skip_iter == 0):
            # Encode the generated images
            enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

            # Calculate losses for z_n, z_c
            zn_loss = mse_loss(enc_gen_zn, zn)
            zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

            # Check requested metric
            if opt.wass_flag:
                # Wasserstein GAN loss
                ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
            else:
                # Vanilla GAN loss               
                v_loss = bce_loss(D_gen, valid)
                ge_loss = v_loss + betan * zn_loss + betac * zc_loss

            ge_loss.backward(retain_graph=True)
            optimizer_GE.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        D_real = discriminator(real_imgs)

        if opt.wass_flag:
            # Gradient penalty term
            grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
            # Wasserstein GAN loss w/gradient penalty
            d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
            
        else:
            # Vanilla GAN loss(原版valid定義在if中，前4輪為未定義)
            #故將fake/valid移動至訓練前           
            real_loss = bce_loss(D_real, valid)
            fake_loss = bce_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


    #Rebuild test
    #1st: real ->encoder ->generator -> img_mse_loss
    #2nd: random ->generator -> encoder ->lat_mse_loss ,lat_xe_loss
    N_test = 25
    N_root = 5
    #--1st--
    generator.eval()
    encoder.eval()
    t_imgs, t_label = test_imgs.detach(), test_labels
    e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)

    teg_imgs = generator(e_tzn, e_tzc)
    img_mse_loss = mse_loss(t_imgs, teg_imgs)

    #---2nd--
    #sample_z(batch, latent_dim=, n_c=10, fix_class=-1, req_grad=False)
    zn_samp, zc_samp, zc_samp_idx = sample_z(batch=N_test,latent_dim=30, n_c=10)

    gen_imgs_samp = generator(zn_samp, zc_samp)
    zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)

    lat_mse_loss = mse_loss(zn_e, zn_samp)
    lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)  

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {ge_loss.item():.4f}]",end='')
    print(f'[Rebuild loss:{img_mse_loss.item():.4f}][Zn loss:{lat_mse_loss.item():4f}][Zc loss:{lat_xe_loss.item():4f}]--{time1round}')
    
    if epoch % 50 == 0 or epoch == 199:
        sample_image(n_row=10, imgtag=epoch)
    