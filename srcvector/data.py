import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from srcvector.consts import VELOCITY_SCALE
from tqdm import tqdm

def read_velocity(filepath, velocity_scale=1000):
    vel = np.load(filepath)['arr_0'].astype(np.float32)
    vel = np.moveaxis(vel, 2, 0)
    vel = vel * velocity_scale
    return vel

def read_structure(filepath):
    img = Image.open(filepath)
    img_palette = img.getpalette()
    img = np.array(img).astype(np.float32)
    if img_palette[0] == 0:
        print(f'Inversing color palette for {filepath}')
        img = 1-img
    return img


class RandomRoll(torch.nn.Module):
    def __init__(self, ratio_x: float = 0, ratio_y: float = 0):
        super().__init__()
        self.ratio_y = ratio_y / 2.0
        self.ratio_x = ratio_x / 2.0

    def forward(self, structure, velocity):
        if self.ratio_y < 1e-4 and self.ratio_x < 1e-4:
            return structure, velocity
        _, size_x, size_y = structure.shape
        size_y = int(size_y * self.ratio_y)
        size_x = int(size_x * self.ratio_x)
        rand_roll_y = np.random.randint(low=-size_y, high=size_y)
        rand_roll_x = np.random.randint(low=-size_x, high=size_x)
        structure = torch.roll(structure, rand_roll_x, 1)
        structure = torch.roll(structure, rand_roll_y, 2)
        velocity = torch.roll(velocity, rand_roll_x, 1)
        velocity = torch.roll(velocity, rand_roll_y, 2)
        return structure, velocity

class RandomFlip(torch.nn.Module):
    def __init__(self, p_flip: float):
        super().__init__()
        self.p_flip = p_flip
    def forward(self, structure, velocity):
        if self.p_flip < 1e-5:
            return structure, velocity
        if np.random.random() < self.p_flip:
            structure = torch.flip(structure, dims=[1])
            velocity = torch.flip(velocity, dims=[1])
            velocity[1, :, :] = -velocity[1, :, :]
        return structure, velocity

class PeriodicWrap(torch.nn.Module):
    def __init__(self, margin: int):
        super().__init__()
        self.margin = margin
    def forward(self, structure, velocity):
        if self.margin > 0:
            structure = torch.cat([structure, structure[:,  :self.margin, :]], axis=1)
            structure = torch.cat([structure, structure[:, :, :self.margin]], axis=2)
            velocity = torch.cat([velocity, velocity[:, :self.margin, :]], axis=1)
            velocity = torch.cat([velocity, velocity[:, :, :self.margin]], axis=2)
        return structure, velocity

class PorousDataset(Dataset):
    def __init__(self, filelist, structures_dirpath, velocity_dirpath, augmentations=None):
        self.filelist = filelist
        self.structures_dirpath = structures_dirpath
        self.velocity_dirpath = velocity_dirpath
        self.velocity_scale = VELOCITY_SCALE
        self.structures = []
        self.velocities = []
        self.augmentations = augmentations
        # read structures
        for structure_filename in tqdm(filelist, desc='Reading structures'):
            self.structures.append(self._read_structure(filename=structure_filename))
        # read velocities
        for velocity_filename in tqdm(filelist, desc='Reading velocities'):
            self.velocities.append(self._read_velocity(filename=velocity_filename))
        assert len(self.structures) == len(self.velocities), "Structure and velocity lists have different lengths"

    def __len__(self):
        return len(self.structures)

    def _read_structure(self, filename):
        try:
            filepath = os.path.join(self.structures_dirpath, f'{filename}.gif')
            img = read_structure(filepath)
            # img = Image.open(filepath)
            # img = np.array(img).astype(np.float32)
        except:
            filepath = os.path.join(self.structures_dirpath, f'{filename}.npy')
            img = np.load(filepath)
            img = img.astype(np.float32)
            img[img > 0] += 2.0
            img /= 25
        img = np.repeat(img[np.newaxis, :, :], 3, axis=0)
        return img

    def _read_velocity(self, filename):
        filepath = os.path.join(self.velocity_dirpath, f'{filename}.gif.vel.npz')
        return read_velocity(filepath=filepath, velocity_scale=self.velocity_scale)
        # vel = np.load(filepath)['arr_0'].astype(np.float32)
        # vel = np.moveaxis(vel, 2, 0)
        # vel = vel * self.velocity_scale
        # return vel

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        structure = torch.Tensor(self.structures[idx])
        velocity = torch.Tensor(self.velocities[idx])
        # augmentation here are ugly implemented due to torch.Compose problem with multiple inputs
        if self.augmentations:
            for aug in self.augmentations:
                structure, velocity = aug(structure, velocity)
        return structure, velocity, filename

