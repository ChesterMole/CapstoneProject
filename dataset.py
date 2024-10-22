import os
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def strip_label(path):
    return path.split('\\')[-1].split('-')[0]

class JPGImageDataset(Dataset):
    def __init__(
            self,
            inputs,
            input_size,
            transform
    ):
        self.inputs = inputs
        self.input_size = input_size
        self.transform = transform

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
    
    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)

        input_tensors = []
        labels = []
        linear_conditions = []

        for i in indexes:
            input_file, label, linear_condition = self.inputs[i]

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

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_file, label, linear_condition = self.inputs[index]
        input_img = self.read_image(input_file)

        if self.transform:
            input_img = self.transform(input_img)

        return {'input': input_img, 'label': int(label), 'linear_condition': linear_condition}

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

        # Get body section labels
        path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        label_check = os.path.join(path, self.input_folder,'*')
        labels_paths = glob(label_check)
        labels = [strip_label(path) for path in labels_paths]
        self.labels = list(dict.fromkeys(labels))
        self.label_dict = {label: idx for idx, label in enumerate(self.labels)}

        self.train_inputs, self.val_inputs, self.test_inputs = self.build_inputs()

        self.train = JPGImageDataset(self.train_inputs, self.input_size, self.transform)
        self.validation = JPGImageDataset(self.val_inputs, self.input_size, self.transform)
        self.test = JPGImageDataset(self.test_inputs, self.input_size, self.transform)

    def build_inputs(self):
        # Get paths for all body sections
        path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        section_path = os.path.join(path, self.input_folder, '*')
        sections = sorted(glob(section_path))

        # Store inputs
        inputs = []

        # Iterate through all body sections 
        for section in sections:
            # Get label
            label = strip_label(section)
            label_index = self.label_dict[label]

            # Get paths for all trajectories
            trajectory_check = os.path.join(section,'*')
            trajectories = sorted(glob(trajectory_check))
            
            # Iterate through all trajectories
            for trajectory in trajectories:
                input_check = os.path.join(path, self.input_folder, section, trajectory, 'Frames','*.jpg')
                input_files = sorted(glob(input_check))

                # Search linear folder for xlsx
                frame_check = os.path.join(path, self.input_folder, section, trajectory, 'Poses','*.xlsx')
                frame_files = glob(frame_check)[0]
                frames = pd.read_excel(frame_files, )

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

                    inputs.append((input_file, label_index, linear_conditions))

        # Split training data
        labels = [item[1] for item in inputs]
        train_inputs, temp_inputs = train_test_split(inputs, test_size=0.3, stratify=labels, random_state=42)

        # Split validation and testing data
        temp_labels = [item[1] for item in temp_inputs]
        val_inputs, test_inputs = train_test_split(temp_inputs, test_size=0.5, stratify=temp_labels, random_state=42)

        return (train_inputs, val_inputs, test_inputs)
    
    def save_inputs(self, results_folder, results_iterations):
        # Get results path
        path = os.path.dirname(os.path.abspath(__file__))
        results_name = 'results-'+str(results_iterations)
        results_path = str(os.path.join(path, results_folder, results_name))
        
        # Create dir if not
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # File paths for inputs
        train_inputs = str(os.path.join(results_path, "train_inputs.txt"))
        val_inputs = str(os.path.join(results_path, "val_inputs.txt"))
        test_inputs = str(os.path.join(results_path, "test_inputs.txt"))

        # Write inputs to file
        with open(train_inputs, 'w') as file:
            for item in self.train_inputs:
                file.write(f"{item}\n")
        with open(val_inputs, 'w') as file:
            for item in self.val_inputs:
                file.write(f"{item}\n")
        with open(test_inputs, 'w') as file:
            for item in self.test_inputs:
                file.write(f"{item}\n")

        print("Saved Inputs to Results")
        return