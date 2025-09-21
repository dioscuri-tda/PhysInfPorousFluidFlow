import os
from datetime import datetime
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def experiment_name_with_timestamp(experiment_name: str) -> str:
    timestamp = str(datetime.now())[:19].replace(' ', '-')
    return f'{timestamp}_{experiment_name}'


def create_experiment_dir(output_dir: str, experiment_name: str):
    experiment_path = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    return experiment_path

def dict_to_json(filepath: str, data):
    with open(filepath, 'w') as fp:
        json.dump(data, fp, indent=4)


def velocity_to_vtk(matrix: np.array, filename: str):
    array = np.zeros((256, 256, 3))
    array[:, :, 0] = matrix[0, :, :]
    array[:, :, 1] = matrix[1, :, :]
    array = array.reshape(-1, 3)
    n_pts = array.shape[0]
    header = f'# vtk DataFile Version 2.0\nLBM+ML\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS 256 256 1\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA {n_pts}\nVECTORS VelocityField double'
    np.savetxt(filename, array, fmt='%.6f', header=header, comments='')

def tensor_smooth_gaussian(tensor, sigma: float):
    if sigma <= 0:
        return tensor
    array = tensor.detach().cpu().numpy()
    for arr in array:
        arr[0, :, :] = gaussian_filter(arr[0, :, :], sigma=sigma, mode='wrap')
        arr[1, :, :] = gaussian_filter(arr[1, :, :], sigma=sigma, mode='wrap')
    return torch.Tensor(array).to(tensor.device)

