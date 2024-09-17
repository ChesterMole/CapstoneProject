import os
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
import re

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
        input_check = os.path.join(path, self.input_folder,'*.jpg')
        input_files = sorted(glob(input_check))
        inputs = []

        # Search linear folder for xlsx
        frame_check = os.path.join(path, self.input_folder,'*.xlsx')
        frame_files = glob(frame_check)[0]
        frames = pd.read_excel(frame_files)

        for input_file in input_files:
            # Get frame number
            frame_num = os.path.split(input_file)[1]
            frame_num = re.split(r'[._]', frame_num)
            step = frame_num[2]
            num = frame_num[3]

            # Find frame values
            frame = frames[(frames['number'] == int(num)) & (frames['step'] == int(step))]

            r = frame['r'].iloc[0]
            g = frame['g'].iloc[0]
            b = frame['b'].iloc[0]
            conditions = [r, g, b]

            conditions = torch.tensor(conditions, dtype=torch.float32)

            label = frame['number'].iloc[0]
            inputs.append((input_file, label, conditions))
        return inputs

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        inputs = self.build_inputs()

        input_tensors = []
        labels = []
        linear_conditions = []

        for i in indexes:
            input_file, label, linear_condition = inputs[i]

            # Load and prepare image
            input_img = self.read_image(input_file)

            # Apply transforms
            if self.transform:
                input_img = self.transform(input_img)
            else:
                input_img = torch.from_numpy(input_img)
                
            # Append label conditions
            input_tensors.append(input_img)
            labels.append(int(label))

            # Append linear conditions
            linear_conditions.append(linear_condition)

        input_tensors = torch.stack(input_tensors).cuda()
        labels = torch.tensor(labels).cuda()
        linear_conditions = torch.stack(linear_conditions).cuda()
        print(labels)
        print(linear_conditions)
        return input_tensors, labels, linear_conditions

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
        input_file, label, linear_condition = self.inputs[index]
        input_img = self.read_image(input_file)

        if self.transform:
            input_img = self.transform(input_img)

        return {'input': input_img, 'label': int(label), 'linear_condition': linear_condition}
