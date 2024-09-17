import torchvision.transforms as transforms
import argparse
import torch

from diffusion_model.unet import create_model
from dataset import JPGImageGenerator
from diffusion_model.trainer import GaussianDiffusion, Trainer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Testing Dataset Information
input_folder = "dataset/sample"
input_size = 64

def scale_to_minus_one_to_one(tensor):
    return (tensor * 2) - 1

# Define your transformations including rotating, scaling, and shifting
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.RandomRotation(degrees=45), 
    #transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2)),  
    #transforms.RandomHorizontalFlip(),  
    #transforms.Lambda(lambda t: (t * 2) - 1)
    transforms.Lambda(scale_to_minus_one_to_one),
])

# Dataset Object
dataset = JPGImageGenerator(
    input_folder,
    input_size,
    transform = transform
)

# Diffusion Model
in_channels = 6
out_channels = 3

num_channels = 64
num_res_blocks = 2

model = create_model(input_size, num_channels, num_res_blocks, class_cond=True).cuda()

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
    train_batch_size=32,
    save_and_sample_every=100,#10,
    train_num_steps = 10000,#30,
    fp16 = True,
    results_iteration = 1
)

def main():
    # Train
    trainer.train()

if __name__ == '__main__':
    main()
