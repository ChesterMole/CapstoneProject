import torchvision.transforms as transforms
import argparse
import torch

from diffusion_model.unet import create_model
from dataset import JPGImageGenerator
from diffusion_model.trainer import GaussianDiffusion, Trainer

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
from glob import glob
import re

def scale_to_minus_one_to_one(tensor):
    return (tensor * 2) - 1


def main():
    # Results Folder
    results_folder = 'results'
    results_iteration = 14

    # Testing Dataset Information
    input_folder = "dataset/LowCam"
    input_size = 640

    # Define your transformations including rotating, scaling, and shifting
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(scale_to_minus_one_to_one)
    ])

    # Dataset Object
    dataset = JPGImageGenerator(
        input_folder,
        input_size,
        transform = transform
    )
    dataset.save_inputs(results_folder, results_iteration)

    # Diffusion Model
    in_channels = 6
    out_channels = 3

    num_channels = 64
    num_res_blocks = 2

    model = create_model(input_size, num_channels, num_res_blocks, class_cond=True, linear_cond=True).cuda()

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
        train_batch_size=2,
        save_and_sample_every=1,
        train_num_steps = 10000,
        fp16 = True,
        results_iteration = results_iteration
    )

    # Train
    trainer.train()

if __name__ == '__main__':
    main()
