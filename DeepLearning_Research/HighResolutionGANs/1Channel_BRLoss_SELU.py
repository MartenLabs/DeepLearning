# nohup python 1Channel_BRLoss_SELU.py > 1Channel_BRLoss_SELU.log 2>&1 &

import os
import torch
import torchvision
import ignite

print(*map(lambda m: ": ".join((m.__name__, m.__version__)), (torch, torchvision, ignite)), sep="\n")

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

if 'CUDA_LAUNCH_BLOCKING' in os.environ:
    del os.environ['CUDA_LAUNCH_BLOCKING']


import os
import logging
import matplotlib.pyplot as plt

import cv2
import numpy as np
from PIL import Image

from torchsummary import summary

# import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.utils as vutils
import torch.autograd as autograd

from ignite.engine import Engine, Events
import ignite.distributed as idist


resolution_list = ["4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256", "512x512"]
# channel_list = [256, 128, 128, 64, 64, 32, 16, 8]
channel_list = [256, 128, 128, 64, 32, 32, 16, 8]



import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        # self.bias.shape: (out_channels)
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    
    def forward(self, x):
        out = self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        return out



class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8


    def forward(self, x):
        # (batch_size, C, H, W) / (batch_size, 1, H, W)
        out = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
        return out



class UpDownSampling(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size


    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.size, mode="nearest")
        return out



class GeneratorConvBlock(nn.Module):

    def __init__(self, step, scale_size):
        super().__init__()
        self.up_sampling = UpDownSampling(size=scale_size)

        # (C_(step-1), H, W) -> (C_step, H, W)
        self.conv1 = WSConv2d(in_channels=channel_list[step-1], out_channels=channel_list[step], kernel_size=3, stride=1, padding=1)

        # (C_step, H, W) -> (C_step, H, W)
        self.conv2 = WSConv2d(in_channels=channel_list[step], out_channels=channel_list[step], kernel_size=3, stride=1, padding=1)

        self.selu = nn.SELU(True)

        self.pn = PixelNorm()


    def forward(self, x):
        self.scaled = self.up_sampling(x)
        
        out = self.conv1(self.scaled)
        out = self.selu(out)
        out = self.pn(out)

        out = self.conv2(out)
        out = self.selu(out)
        out = self.pn(out)

        return out



class Generator(nn.Module):

    def __init__(self, steps):
        super().__init__()

        self.steps = steps

        self.init = nn.Sequential(
            PixelNorm(),

            # (z_dim, 1, 1) -> (C_0, 4, 4)
            nn.ConvTranspose2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),

            # (C_0, 4, 4) -> (C_0, 4, 4)
            WSConv2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.init_torgb = WSConv2d(in_channels=channel_list[0], out_channels=1, kernel_size=1, stride=1, padding=0)

        self.prog_blocks = nn.ModuleList([self.init])
        self.torgb_layers = nn.ModuleList([self.init_torgb])
        
        # append blocks that are not init block.
        for step in range(1, self.steps+1):
            self.prog_blocks.append(GeneratorConvBlock(step, scale_size=2))
            self.torgb_layers.append(WSConv2d(in_channels=channel_list[step], out_channels=1, kernel_size=1, stride=1, padding=0))


    def fade_in(self, alpha, upsampling, generated):
        return alpha * generated + (1 - alpha) * upsampling


    def forward(self, x, alpha):
        out = self.prog_blocks[0](x)

        if self.steps == 0:
            return self.torgb_layers[0](out)

        for step in range(1, self.steps+1):
            out = self.prog_blocks[step](out)

        upsampling = self.torgb_layers[step-1](self.prog_blocks[step].scaled)
        generated = self.torgb_layers[step](out)

        return self.fade_in(alpha, upsampling, generated)



class DiscriminatorConvBlock(nn.Module):

    def __init__(self, step):
        super().__init__()

        # (C_step, H, W) -> (C_step, H, W)
        self.conv1 = WSConv2d(in_channels=channel_list[step], out_channels=channel_list[step], kernel_size=3, stride=1, padding=1)

        # (C_step, H, W) -> (C_(step-1), H, W)
        self.conv2 = WSConv2d(in_channels=channel_list[step], out_channels=channel_list[step-1], kernel_size=3, stride=1, padding=1)

        # (C_(step-1), H/2, W/2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.selu = nn.SELU(True)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.selu(out)

        out = self.conv2(out)
        out = self.selu(out)

        out = self.downsample(out)

        return out



class MinibatchStd(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):
        # mean of minibatch's std
        # (1) -> (batch_size, 1, H, W)
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.size(0), 1, x.size(2), x.size(3)))

        # (batch_size, C, H, W) -> (batch_size, C+1, H, W)
        return torch.cat((x, batch_statistics), dim=1)



class Discriminator(nn.Module):

    def __init__(self, steps):
        super().__init__()
        # progressive growing blocks
        self.prog_blocks = nn.ModuleList([])

        # fromrgb layers
        self.fromrgb_layers = nn.ModuleList([])

        self.selu = nn.SELU(True)

        self.steps = steps
        
        # append blocks that are not final block.
        for step in range(steps, 0, -1):
            self.prog_blocks.append(DiscriminatorConvBlock(step))
            self.fromrgb_layers.append(WSConv2d(in_channels=1, out_channels=channel_list[step], kernel_size=1, stride=1, padding=0))

        # append final block
        self.fromrgb_layers.append(
            WSConv2d(in_channels=1, out_channels=channel_list[0], kernel_size=1, stride=1, padding=0)
        )

        # append final block
        self.prog_blocks.append(
            nn.Sequential(
                MinibatchStd(),
                WSConv2d(in_channels=channel_list[0]+1, out_channels=channel_list[0], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                WSConv2d(in_channels=channel_list[0], out_channels=channel_list[0], kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2),
                WSConv2d(in_channels=channel_list[0], out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        )

        # downsample
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    
    def forward(self, x, alpha):
        # (3, H, W) -> (C, H, W)
        out = self.selu(self.fromrgb_layers[0](x))

        if self.steps == 0: # i.e, image size is 4x4
            
            # (C, 4, 4) -> (1, 1, 1)
            out = self.prog_blocks[-1](out)

            # (1, 1, 1) -> (1)
            # out.size(0) = batch_size
            return out.view(out.size(0), -1)
        

        downscaled = self.selu(self.fromrgb_layers[1](self.avgpool(x)))
        out = self.prog_blocks[0](out)

        out = self.fade_in(alpha, downscaled, out)
        
        for i in range(1, self.steps+1):
            out = self.prog_blocks[i](out)

        return out.view(out.size(0), -1)



from PIL import Image
from torchvision import transforms
import os
import torch
import numpy as np



class Dataset:

    def __init__(self, directory_list, resolution):
        self.directory_list = directory_list
        self.resolution = resolution


    def image_to_tensor(self, path, res):
        img = Image.open(path).convert('L').resize(res)

        tensor_img = transforms.ToTensor()(img)
        tensor_img = tensor_img.type(torch.float16)

        return tensor_img


    def dataset_to_tensor(self, directory_path):
        files = os.listdir(directory_path)
        tensor_dataset = torch.zeros((len(files), 1, *self.resolution)).type(torch.float16)

        for i in range(len(files)):
            tensor_dataset[i] = self.image_to_tensor(f"{directory_path}/{files[i]}", self.resolution)
        
        return tensor_dataset


    def extract_dataset(self):
        dataset_pair = []

        for directory_path in self.directory_list:
            dataset_pair.append(self.dataset_to_tensor(directory_path))

        return dataset_pair
    

def make_gif(paths, save_path, fps=500):
    img, *imgs = [Image.open(path) for path in paths]
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=fps, loop=1)


def merge_test_pred(pred):

    test_size = pred.size(0)
    
    # ex) test_size = 30 -> height = 5, weight = 6
    for i in range(int(np.sqrt(test_size)), test_size+1):
        if test_size % i == 0:
            n_height = max(i, test_size//i)
            n_weight = min(i, test_size//i)
            break
    
    image_size = (
        1024 - (1024 % n_weight),
        1024 - (1024 % n_height)
    )

    one_image_size = (image_size[0] // n_weight, image_size[1] // n_height)

    image = Image.new('RGB', image_size)

    for w in range(n_weight):
        for h in range(n_height):
            img = transforms.ToPILImage()(pred[n_height*w + h])
            img = img.resize(one_image_size)

            image.paste(img, (one_image_size[0] * w, one_image_size[1] * h))
    
    return image



def brightness_loss(y_true, y_pred):
    squared_difference = torch.square(y_true - y_pred)
    brightness_loss = torch.mean(squared_difference)
    
    return brightness_loss

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import gc

dataset_path = [f"HighResolution/torch/{i}" for i in resolution_list]
model_state_dict_path = [f"model_state_dict_br_selu/{i}" for i in resolution_list]

lambda_brightness = 1


class Trainer():

    def __init__(self,
                steps: int,
                batch_size: int,
                device: torch.device,
                test_size: int
            ):

        self.steps = steps
        self.batch_size = batch_size
        self.device = device
        self.test_size = test_size

        directory_path = dataset_path[self.steps]

        self.trainloader = DataLoader(torch.cat((torch.load(f"{directory_path}/train_0.pt")), dim=0).type(torch.float32), batch_size=self.batch_size, shuffle=True)
      
        self.generator = Generator(steps=self.steps).to(self.device)
        self.discriminator = Discriminator(steps=self.steps).to(self.device)

        self.criterion = nn.BCELoss()
        self.generator_optim = Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.discriminator_optim = Adam(self.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

        # It will be used for testing.
        self.test_z = torch.randn((self.test_size, 256, 1, 1)).to(self.device)

        self.load_model()


    # def save_model(self):
    #     # ------------------------------------------------------------------ generator model ---------------------------------------------------------------------------
    #     for i in range(self.steps+1):
    #         torch.save(self.generator.prog_blocks[i].state_dict(), f"{model_state_dict_path[self.steps]}/generator_model/prog_blocks_{i}.pt")
    #         torch.save(self.generator.torgb_layers[i].state_dict(), f"{model_state_dict_path[self.steps]}/generator_model/torgb_layers_{i}.pt")

    #     # ---------------------------------------------------------------- discriminator model -------------------------------------------------------------------------
    #     for i in range(self.steps+1):
    #         torch.save(self.discriminator.prog_blocks[i].state_dict(), f"{model_state_dict_path[self.steps]}/discriminator_model/prog_blocks_{i}.pt")
    #         torch.save(self.discriminator.fromrgb_layers[i].state_dict(), f"{model_state_dict_path[self.steps]}/discriminator_model/fromrgb_layers_{i}.pt")
    
    def save_model(self):
        self.generator.eval()
        self.discriminator.eval()

        # 모델 상태를 저장할 기본 경로
        base_path = "model_state_dict_br_selu"

        for i in range(self.steps + 1):
            # 생성자와 판별자의 상태 저장 경로 설정
            gen_path = f"{base_path}/{resolution_list[self.steps]}/generator_model"
            disc_path = f"{base_path}/{resolution_list[self.steps]}/discriminator_model"

            # 디렉토리가 없으면 생성
            os.makedirs(gen_path, exist_ok=True)
            os.makedirs(disc_path, exist_ok=True)

            # 생성자 모델 상태 저장
            torch.save(self.generator.prog_blocks[i].state_dict(), f"{gen_path}/prog_blocks_{i}.pt")
            torch.save(self.generator.torgb_layers[i].state_dict(), f"{gen_path}/torgb_layers_{i}.pt")

            # 판별자 모델 상태 저장
            torch.save(self.discriminator.prog_blocks[i].state_dict(), f"{disc_path}/prog_blocks_{i}.pt")
            torch.save(self.discriminator.fromrgb_layers[i].state_dict(), f"{disc_path}/fromrgb_layers_{i}.pt")


    def load_model(self):
        if self.steps == 0:
            return

        # ------------------------------------------------------------------ generator model ---------------------------------------------------------------------------
        for i in range(self.steps):
            self.generator.prog_blocks[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/generator_model/prog_blocks_{i}.pt"))
            self.generator.torgb_layers[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/generator_model/torgb_layers_{i}.pt"))

        # ---------------------------------------------------------------- discriminator model -------------------------------------------------------------------------
        for i in range(1, self.steps+1):
            self.discriminator.prog_blocks[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/discriminator_model/prog_blocks_{i-1}.pt"))
            self.discriminator.fromrgb_layers[i].load_state_dict(torch.load(f"{model_state_dict_path[self.steps-1]}/discriminator_model/fromrgb_layers_{i-1}.pt"))
    

    def clear_cuda_memory(self):
        gc.collect()
        torch.cuda.empty_cache()



    def test(self, epoch):
        self.generator.eval()
        self.discriminator.eval()

        pred = self.generator(self.test_z, alpha=self.alpha)
        pred = pred.detach().cpu()

        # 이미지를 저장할 경로를 생성합니다.
        save_path = f"./1Channel_BRLoss_SELU_Log/{resolution_list[self.steps]}"
        os.makedirs(save_path, exist_ok=True)  # 해당 경로에 폴더가 없으면 생성합니다.

        test_image = merge_test_pred(pred)
        test_image.save(fp=f"{save_path}/epoch-{epoch}.jpg")



    def train(self):
        self.generator.train()
        self.discriminator.train()

        generator_avg_loss = 0
        discriminator_avg_loss = 0

        for _ in range(len(self.trainloader)):
            self.alpha += self.alpha_gap

            real_image = next(iter(self.trainloader)).to(self.device)

            real_label = torch.full((real_image.size(0), 1), 1).type(torch.float).to(self.device)
            fake_label = torch.full((real_image.size(0), 1), 0).type(torch.float).to(self.device)

            # ---------------------------------------------------------- discriminator train ------------------------------------------------------------
            z = torch.randn(real_image.size(0), 256, 1, 1).to(self.device)

            fake_image = self.generator(z, alpha=self.alpha)
            
            d_fake_pred = self.discriminator(fake_image, alpha=self.alpha)
            d_fake_loss = self.criterion(d_fake_pred, fake_label)

            d_real_pred = self.discriminator(real_image, alpha=self.alpha)
            d_real_loss = self.criterion(d_real_pred, real_label)

            d_loss = d_fake_loss + d_real_loss

            self.discriminator_optim.zero_grad()
            d_loss.backward()
            self.discriminator_optim.step()

            discriminator_avg_loss += (d_loss.item() / 2)

            # ---------------------------------------------------------- generator train -----------------------------------------------------------------
            z = torch.randn(real_image.size(0), 256, 1, 1).to(self.device)

            fake_image = self.generator(z, alpha=self.alpha)

            d_fake_pred = self.discriminator(fake_image, alpha=self.alpha)
            g_loss = self.criterion(d_fake_pred, real_label) + lambda_brightness * brightness_loss(real_image, fake_image)

            self.generator_optim.zero_grad()
            g_loss.backward()
            self.generator_optim.step()

            generator_avg_loss += g_loss.item()


            self.clear_cuda_memory()

        generator_avg_loss /= len(self.trainloader)
        discriminator_avg_loss /= len(self.trainloader)

        return generator_avg_loss, discriminator_avg_loss


    def run(self, epochs):
        train_history = []

        self.alpha = 0
        self.alpha_gap = 1 / (len(self.trainloader) * (epochs[1] - epochs[0]))

        for epoch in range(*epochs):
            print("-"*100 + "\n" + f"Epoch: {epoch}")

            train_history.append(self.train())
            print(f"\tTrain\n\t\tG Loss: {train_history[-1][0]},\tD Loss: {train_history[-1][1]}")

            self.test(epoch)
    
        return train_history

    
if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    for steps in range(8):
        trainer = Trainer(steps=steps, batch_size=16, device=device, test_size=16)
        train_history = trainer.run((0, 256))
        trainer.save_model()


