import torchvision.transforms as transforms
import argparse
import torch

from diffusion_model.unet import create_model
from dataset import JPGPairImageGenerator
from diffusion_model.trainer import GaussianDiffusion, Trainer

# Testing Dataset Information
input_folder = "dataset/test/images"
input_size = 256

# Dataset Object
dataset = JPGPairImageGenerator(
    input_folder,
    input_size
)

# Diffusion Model
in_channels = 6
out_channels = 3

num_channels = 64
num_res_blocks = 2

model = create_model(input_size, num_channels, num_res_blocks).cuda()

timesteps = 250

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    timesteps = timesteps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

# Trainer Object
trainer = Trainer(
    diffusion,
    dataset,
    train_batch_size=4,
    save_and_sample_every=100,#10,
    train_num_steps = 10000,#30,
    fp16 = True
)

def main():
    # Train
    trainer.train()

if __name__ == '__main__':
    main()
