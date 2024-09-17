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
        input_check = os.path.join(path, self.input_folder,'Frames','*.jpg')
        input_files = sorted(glob(input_check))
        inputs = []

        # Search linear folder for xlsx
        frame_check = os.path.join(path, self.input_folder,'Poses','*.xlsx')
        frame_files = glob(frame_check)[0]
        frames = pd.read_excel(frame_files)

        for input_file in input_files:
            # Get frame number
            frame_num = os.path.split(input_file)[1]
            frame_num = re.split(r'[._]', frame_num)[1]

            # Find frame values
            frame = frames[frames['ImageFrame'] == int(frame_num)]

            # Translation data
            trans_x = frame['trans_x'].iloc[0]
            trans_y = frame['trans_y'].iloc[0]
            trans_z = frame['trans_z'].iloc[0]
            translation = [trans_x, trans_y, trans_z]
            translation = torch.tensor(translation, dtype=torch.float32)

            # Quaternion data
            qout_x = frame['quot_x'].iloc[0]
            qout_y = frame['quot_y'].iloc[0]
            qout_z = frame['quot_z'].iloc[0]
            quaternion = [qout_x, qout_y, qout_z]
            quaternion = torch.tensor(quaternion, dtype=torch.float32)

            linear_conditions = torch.cat((translation, quaternion))

            label = 1#frame['number'].iloc[0]
            inputs.append((input_file, label, linear_conditions))
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
    
    def resize_with_padding(self, image, target_size):
        img_ratio = image.width / image.height
        target_ratio = target_size / target_size

        if img_ratio > target_ratio:
            new_width = target_size
            new_height = int(target_size / img_ratio)

        elif img_ratio < target_ratio:
            new_width = int(target_size * img_ratio)
            new_height = target_size

        else:
            new_height = target_size
            new_width = target_size
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        paste_x = (target_size-new_width)//2
        paste_y = (target_size-new_height)//2
        padded_image.paste(resized_image, (paste_x,paste_y))

        return padded_image

    def read_image(self, file_path):
        img = Image.open(file_path)
        #img = img.resize((self.input_size, self.input_size))
        
        img = self.resize_with_padding(img, self.input_size)
        
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
