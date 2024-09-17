import os
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch

class JPGImageGenerator(Dataset):
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
        #input_check = os.path.join(path, self.input_folder, '*.jpg')
        input_check = os.path.join(path, self.input_folder,'*','*.jpg')
        input_files = sorted(glob(input_check))
        inputs = []

        for input_file in input_files:
            label = os.path.split(os.path.split(input_file)[0])[1]
            inputs.append((input_file, label))
        return inputs

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        inputs = self.build_inputs()

        input_tensors = []
        labels = []

        for i in indexes:
            input_file, label = inputs[i]

            # load and prepare image
            input_img = self.read_image(input_file)

            if self.transform:
                input_img = self.transform(input_img)
            else:
                input_img = torch.from_numpy(input_img)
                
            # Append sample conditions
            input_tensors.append(input_img)
            labels.append(int(label))

        input_tensors = torch.stack(input_tensors).cuda()
        labels = torch.tensor(labels).cuda()
        return input_tensors, labels

    def read_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((self.input_size, self.input_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        return img

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_file, label = self.inputs[index]
        input_img = self.read_image(input_file)

        if self.transform:
            input_img = self.transform(input_img)

        return {'input': input_img, 'label': int(label)}
