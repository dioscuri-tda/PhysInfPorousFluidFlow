import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from srcvector.utils import velocity_to_vtk, tensor_smooth_gaussian


def plot_history(df_train: pd.DataFrame, df_val: pd.DataFrame, experiment_path: str):
    plot_path = os.path.join(experiment_path, 'history.png')
    n_metrics = df_train.shape[1]
    fig, axs = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    for ax, metric in zip(axs, df_train.columns):
        ax.plot(df_train[metric], 'o-', label='Train')
        ax.plot(df_val[metric], 'o-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric)
    plt.legend()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def plot_velocity_with_vector_field(u, v, ax, grid_subsample=8):
    vel = np.sqrt(u**2 + v**2)
    u = u[::grid_subsample, ::grid_subsample]
    v = v[::grid_subsample, ::grid_subsample]
    xs, ys = np.meshgrid(range(0, vel.shape[0], grid_subsample), range(0, vel.shape[1], grid_subsample))
    image_colorbar = ax.imshow(vel, origin='lower')
    ax.quiver(xs.reshape(-1, 1), ys.reshape(-1, 1), u.reshape(-1, 1), v.reshape(-1, 1), color='white')
    return image_colorbar


def plot_sample_predictions(model, dataloader, experiment_path, device, dataloader_name, crop_to_256=False):
    model.eval()
    imgfilepath = os.path.join(experiment_path, f'predictions_{dataloader_name}.png')
    if crop_to_256:
        imgfilepath = os.path.join(experiment_path, f'predictions_crop256_{dataloader_name}.png')
    sample = next(iter(dataloader))
    imgs = sample[0]
    vels = sample[1]
    pred = model(imgs.to(device)).data.cpu().numpy()
    if crop_to_256:
        imgs = imgs[:, :, :256, :256]
        vels = vels[:, :, :256, :256]
        pred = pred[:, :, :256, :256]
    batch_size = len(sample[0])
    batch_size = min(batch_size, 10)
    fig, axs = plt.subplots(batch_size, 4, figsize=(30, 6 * batch_size))
    for batch_id in range(batch_size):
        axs[batch_id][0].imshow(imgs[batch_id, 0, :, :], origin='lower')
        im1 = plot_velocity_with_vector_field(u = vels[batch_id, 0, :, :], v = vels[batch_id, 1, :, :], ax=axs[batch_id][1])
        im2 = plot_velocity_with_vector_field(u = pred[batch_id, 0, :, :], v = pred[batch_id, 1, :, :], ax=axs[batch_id][2])
        im3 = plot_velocity_with_vector_field(u = vels[batch_id, 0, :, :] - pred[batch_id, 0, :, :], v = vels[batch_id, 1, :, :] - pred[batch_id, 1, :, :], ax=axs[batch_id][3])
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.colorbar(im3)
    plt.savefig(imgfilepath, bbox_inches='tight')
    plt.close()
    model.train()

def vtk_sample_predictions(model, dataloader, experiment_path, device, dataloader_name, crop_to_256=False, post_smooth_sigma=0.0):
    model.eval()
    sample = next(iter(dataloader))
    imgs = sample[0]
    vels = sample[1]

    pred = model(imgs.to(device))
    pred = tensor_smooth_gaussian(tensor=pred, sigma=post_smooth_sigma)
    pred = pred.data.cpu().numpy()

    if crop_to_256:
        pred = pred[:, :, :256, :256]
        vels = vels[:, :, :256, :256]

    batch_size = len(sample[0])
    batch_size = min(batch_size, 10)

    for batch_id in range(batch_size):
        # save prediction
        vtkfilepath = os.path.join(experiment_path, f'predictions_{dataloader_name}_{batch_id}.vtk')
        if crop_to_256:
            vtkfilepath = os.path.join(experiment_path, f'predictions_crop256_{dataloader_name}_{batch_id}.vtk')
        velocity_to_vtk(pred[batch_id, :, :, :], filename=vtkfilepath)
        # save ground truth
        vtkfilepath = os.path.join(experiment_path, f'target_{dataloader_name}_{batch_id}.vtk')
        if crop_to_256:
            vtkfilepath = os.path.join(experiment_path, f'target_crop256_{dataloader_name}_{batch_id}.vtk')
        velocity_to_vtk(vels[batch_id, :, :, :], filename=vtkfilepath)

    model.train()