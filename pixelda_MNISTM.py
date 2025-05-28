#Pixel-Level Domain Adaptation

import torch
import os
import argparse
import itertools
import torch.nn as nn
import datetime
import time

from torchvision import datasets #現成的資料集庫
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
opt = parser.parse_args()
print(opt)

patch = (1,2,2)
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
    row = 10
    dataiter = iter(dataloader3)
    images, _ = next(dataiter)
    images = images.to(dtype=torch.float32,device=device).expand(row**2, 3, opt.img_size, opt.img_size)
    z = torch.empty((row**2, opt.latent_dim), device=device).uniform_(-1, 1)
    generator.eval()
    with torch.inference_mode():
        sample = generator(images,z)
        samplewithseed = torch.cat((images,sample),-2)
    save_image(samplewithseed,f'./samples/pixelda_{imgtag}.png', nrow=row, normalize=True)
    generator.train()

#修正辭意:最後2個更為out_features
#shape unchange : (b,64,h,w)
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, out_features, 3, 1, 1),
            nn.BatchNorm2d(out_features))
    def forward(self, x):
        return x + self.block(x)
#input:(b,3,32,32)/(b,10) => img(b,3,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels * opt.img_size ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(opt.channels * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)#(b,6,32,32)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)#(b,3,32,32)

        return img_

#input:(b,3,32,32) => val(b,1,2,2)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),#(b,512,2,2)
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity#(b,1,2,2)

#input:(b,3,32,32) => one-hot_like(b,10)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512))#(b,512,2,2)

        input_size = opt.img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, opt.n_classes), nn.Softmax(dim=-1))

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label   

class MNISTM(Dataset):
    def __init__(self, root,  mode='train', transform=None, target_transform=None):        
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.data, self.targets = torch.load(os.path.join(self.root,f'mnist_m_{mode}.pt'),weights_only=True)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = img.permute(2, 0, 1)
        img = to_pil_image(img)

        #img = Image.fromarray(img.numpy(), mode="RGB")       
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.data)

#object
generator = Generator().to(device)
discriminator = Discriminator().to(device)
classifier = Classifier().to(device)
#loss
adversarial_loss = torch.nn.MSELoss().to(device)
task_loss = torch.nn.CrossEntropyLoss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(), classifier.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# normal
transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader1 = DataLoader(dataset_1, batch_size=64, shuffle=True)
# color
transform_seq2 = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
dataset_2 = MNISTM(root="./Resorces/mnist_m_data", mode='train', transform=transform_seq2)
dataloader2 = DataLoader(dataset_2,batch_size=64, shuffle=True)
# for inference
dataset_3 = datasets.MNIST("./Resorces", train=False, download=False, transform=transform_seq1)
dataloader3 = DataLoader(dataset_3, batch_size=100, shuffle=True)

generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)
classifier.apply(custom_weights_init)

task_performance = []
target_performance = []
lambda_adv = 1
lambda_task = 0.1
for epoch in range(opt.n_epochs):
    prev_time = time.time() 
    prev_time = time.time()
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader1, dataloader2)):
        batch_size = imgs_A.size(0)
        valid = torch.ones((batch_size,*patch),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((batch_size,*patch),dtype = torch.float32 , device=device ,requires_grad=False)

        imgs_A = imgs_A.to(dtype=torch.float32,device=device).expand(batch_size, 3, opt.img_size, opt.img_size)
        #灰階c=1 => expand => c=3 且不佔額外記憶體
        labels_A = labels_A.to(dtype=torch.long,device=device)

        imgs_B = imgs_B.to(dtype=torch.float32,device=device)
         # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.empty((batch_size, opt.latent_dim), device=device).uniform_(-1, 1)
        fake_B = generator(imgs_A, z)
        label_fake = classifier(fake_B)

        task_loss_ = (task_loss(label_fake, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2 #content
        adv_loss = adversarial_loss(discriminator(fake_B), valid) #style
        g_loss = lambda_adv * adv_loss + lambda_task * task_loss_

        g_loss.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs_B), valid)
        fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}]",end='')
    print(f'[_content:{task_loss_.item():.4f}][_style:{adv_loss.item():.4f}]--{time1round}')

    if epoch % 50 == 0 or epoch == 199:
        sample_images(imgtag=epoch)
    


