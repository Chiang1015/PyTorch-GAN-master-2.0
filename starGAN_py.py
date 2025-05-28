# Star Generative Adversarial Network
# 多個 domain 間的轉換
# medal loading model, trained models in 'models'
import torch
import glob
import time
import argparse
import datetime

import torch.nn as nn
import torch.autograd as autograd

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="celebA", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument(
    "--selected_attrs",
    "--list",
    nargs="+",
    help="selected attributes for the CelebA dataset",
    default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
)
#othrer trait option:
# 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair 
# Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones 
# Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline 
# Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace 
# Wearing_Necktie Young
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
opt = parser.parse_args()
print(opt)

c_dim = len(opt.selected_attrs)
img_shape = (opt.channels, opt.img_height, opt.img_width) #(3,128,128)
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

def compute_gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10.0):
    batch_size = real_data.size(0)
    device = real_data.device

    # 隨機係數 alpha 用來插值 real 和 fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    d_interpolated,_ = discriminator(interpolated)
    ones = torch.ones_like(d_interpolated)

    gradients = autograd.grad(
        outputs=d_interpolated, #微分的目標,如y(x)
        inputs=interpolated, #微分的變數,如dx,數量有32*32個
        grad_outputs=ones, #起始梯度
        create_graph=True, #保留計算圖,可以對這個梯度再做微分
        retain_graph=True, #保留整個計算圖
        only_inputs=True, #只針對 inputs 求梯度，而不考慮其他會牽涉的中間變數
    )[0]
    #autograd.grad() 會傳回一個 tuple，裡面是每個 inputs 對應的梯度
    #這裡只有一個 inputs（interpolated），我們只取 [0]
    gradients = gradients.view(batch_size, -1) #(batch_size, num_features=32*32)
    gradient_norm = gradients.norm(2, dim=1)
    #L2 norm對dim=1 => (batch_size,) 代表梯度強度
    penalty =  ((gradient_norm - 1) ** 2).mean()
    #梯度強度與初始梯度1的差距取平均
    return penalty

#label_changes須配合opt.selected_attrs依序指定:
#前三碼(髮色),第四碼性別,第五碼年齡
#eg. [1,0,0,1,1]=> Black_Hair /Male / Young
def sample_image(imgtag):
    label_changes = torch.tensor([
    [1,0,0,1,1],
    [0,1,0,1,1],
    [0,0,1,1,1],
    [1,0,0,0,1],
    [0,1,0,0,1]], dtype=torch.float32, device=device)
    var_n = label_changes.size(0) 

    val_imgs, val_labels = next(iter(val_dataloader))
    val_imgs = val_imgs.to(dtype=torch.float32,device=device)
    val_labels = val_labels.to(dtype=torch.float32,device=device)
    img_samples = None

    generator.eval()
    with torch.inference_mode():
        for i in range(10):
            img, label = val_imgs[i], val_labels[i]
            # img(1,3,128,128) => imgs (5,3,128,128)
            imgs = img.repeat(var_n, 1, 1, 1)
            labels = label.repeat(var_n, 1) #(5,c_dim)
            #修正指定方式
            labels = label_changes#抽換原label(保留shape)
        
            # glue along x-axis
            gen_imgs = generator(imgs, labels)
            gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_imgs), -1) #img:原始圖
            # along y-axis
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

    save_image(img_samples.view(1, *img_samples.shape), f'./samples/starGAN_{imgtag}.png', normalize=True)
    generator.train()

#input: (b,in,h,w) => (b,in,h,w)
#track_running_stats=True 推理時使用統計數據
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

#input: (b,3,128,128)+(b,c_dim) => img(b,3,128,128)
class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, _, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)]
            
            curr_dim *= 2
        #(b,256,32,32)
        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        #(b,256,32,32)
        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        #(b,64,128,128)
        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1) #(b,c_dim) =>(b,c_dim,1,1)
        c = c.repeat(1, 1, x.size(2), x.size(3)).to(device)
        x = torch.cat((x, c), 1)
        return self.model(x) #(b,3,128,128)

#input:(b,3,128,128) => (b,1,2,2) / (b,5)
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4,stride=2, padding=1)
                      ,nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        #(b,64,64,64)
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2
        #(b,2048,2,2)
        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided #128/2^6 = 2
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)#(b,2048,2,2)
        out_adv = self.out1(feature_repr)#(b,1,2,2)
        out_cls = self.out2(feature_repr)#(b,5,1,1)
        return out_adv, out_cls.view(out_cls.size(0), -1)

#root : pic file
#output: img / label(w/o appoint trait)
class CelebADataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", attributes=None):
        self.transform = transforms.Compose(transforms_)

        self.selected_attrs = attributes #appoint trait
        self.label_path = glob.glob(root +"/*.txt")[0] 
        self.annotations = self.get_annotations() #get dic form file 30000 only

        
        self.files = sorted(glob.glob(root + "/celebA/*.png"))
        
        split = len(self.files) - 2000 #2000 for 'val'      
        self.files = self.files[:split] if mode == "train" else self.files[split:]
        
    #return dic to show how the picture match appoint traits
    # {000001.png': [0, 0, 1, 0, 1]} mean it has 3rd'Brown_Hair' and 5th'Young' trait
    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, "r")]
        self.label_names = lines[1].split()
        max_entries = 30000 #資料圖片集數量
        for _, line in enumerate(lines[2:max_entries+2]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1"))
            annotations[filename] = labels
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split("\\")[-1]

        img = self.transform(Image.open(filepath).convert("RGB"))

        label = torch.tensor(self.annotations[filename], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.files)

#object   
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks, c_dim=c_dim).to(device)
discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim).to(device)
#loss
criterion_cls = nn.BCEWithLogitsLoss(reduction='mean').to(device)
criterion_cycle = nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

train_transforms = [
    transforms.Resize(int(1.12 * opt.img_height), Image.BICUBIC),
    transforms.RandomCrop(opt.img_height),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
val_transforms = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

dataloader = DataLoader(
    CelebADataset("./Resorces/face64X64", transforms_=train_transforms, mode="train", attributes=opt.selected_attrs),
    batch_size=opt.batch_size,
    shuffle=True, #16
    num_workers=0)
val_dataloader = DataLoader(
    CelebADataset("./Resorces/face64X64" , transforms_=val_transforms, mode="val", attributes=opt.selected_attrs),
    batch_size=10,
    shuffle=True,
    num_workers=0,
)

if opt.epoch == 0:
    generator.apply(custom_weights_init)
    discriminator.apply(custom_weights_init)
else :
    generator.load_state_dict(torch.load(f"./models/starGAN_G_{opt.epoch}"))
    discriminator.load_state_dict(torch.load(f"./models/starGAN_D_{opt.epoch}"))
# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

for epoch in range(opt.epoch, opt.n_epochs):
    prev_time = time.time()
    
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(dtype=torch.float32,device=device)
        labels = labels.to(dtype=torch.float32,device=device)

        z = torch.randint(0, 2, (imgs.size(0), c_dim), dtype=torch.float32,device=device)
        gen_imgs = generator(imgs, z)
        # Train_D:Train_G = 5:1
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_validity, pred_cls = discriminator(imgs)
        fake_validity, _ = discriminator(gen_imgs.detach())

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, gen_imgs.data)
        # Adversarial loss:真實度
        loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity)#移除gradient_penalty
        # Classification loss (b,5)&(b,5):標籤相似度
        loss_D_cls = criterion_cls(pred_cls, labels)

        loss_D = loss_D_adv + lambda_cls * loss_D_cls+ lambda_gp * gradient_penalty
        loss_D.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        if i % opt.n_critic == 0:
            # Translate and reconstruct image
            gen_imgs = generator(imgs, z)
            recov_imgs = generator(gen_imgs, labels)
            fake_validity, pred_cls = discriminator(gen_imgs)

            # Adversarial loss
            loss_G_adv = -torch.mean(fake_validity)
            # Classification loss
            loss_G_cls = criterion_cls(pred_cls, z)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recov_imgs, imgs)

            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec
            loss_G.backward()
            optimizer_G.step()
    
    time1round = datetime.timedelta(seconds = time.time()-prev_time)
    print(f'[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss:(10gp{gradient_penalty.item():.4f})+(1adv{loss_D_adv.item():.4f})+(1cls{loss_D_cls.item():.4f})={loss_D.item():.4f}]',end='')
    print(f'[G loss:(10rec{loss_G_rec.item():.4f})+(1adv{loss_G_adv.item():.4f})+(1cls{loss_G_cls.item():.4f})={loss_G.item():.4f}]--{time1round}')

    if epoch % 50 ==0 or epoch ==199 :
        sample_image(imgtag=epoch)
    
    if epoch == 199:
        torch.save(generator.state_dict(), f"./models/starGAN_G_{epoch}" )
        torch.save(discriminator.state_dict(), f"./models/starGAN_D_{epoch}" )

