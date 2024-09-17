import os
from glob import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch

class JPGPairImageGenerator(Dataset):
    def __init__(
            self,
            input_folder: str,
            input_size: int,
            transform=None
    ):
        self.input_folder = input_folder
        self.input_size = input_size
        self.transform = transform
        self.inputs = self.build_inputs()

    def build_inputs(self):
        path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        input_check = os.path.join(path, self.input_folder, '*.jpg')
        input_files = sorted(glob(input_check))
        inputs = []

        for input_file in input_files:
            inputs.append(input_file)
        return inputs

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = self.build_inputs()

        input_tensors = []
        for i in indexes:
            input_img = self.read_image(input_files[i])
            #input_img = np.moveaxis(input_img, 0, -1)
            input_img = torch.from_numpy(input_img)
            if self.transform:
                print("Transforms not setup")
                #input_img = self.transform(input_img)
                #input_tensors.append(input_img)
            input_tensors.append(input_img)

        return torch.stack(input_tensors).cuda()

    def read_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((256, 256))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        img = np.moveaxis(img, -1, 0)
        return img

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_file = self.inputs[index]
        input_img = self.read_image(input_file)
        return input_img
