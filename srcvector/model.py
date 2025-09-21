import copy
import os
import time
from collections import defaultdict
import pickle

from fontTools.ttLib.tables.D_S_I_G_ import pem_spam
from sympy.printing.codeprinter import requires
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import torch
from srcvector.consts import FX, MU, MAX_TORTUOSITY, MIN_TORTUOSITY
import numpy as np
from tqdm import tqdm
import sys

from srcvector.plots import plot_history

class RandomRollTensor(torch.nn.Module):
    def __init__(self, ratio_x: float = 0, ratio_y: float = 0):
        super().__init__()
        self.ratio_y = ratio_y / 2.0
        self.ratio_x = ratio_x / 2.0
        self.rand_roll_y = 0
        self.rand_roll_x = 0


    def roll(self, structure, velocity):
        if self.ratio_y < 1e-4 and self.ratio_x < 1e-4:
            return structure, velocity
        _, _, size_x, size_y = structure.shape
        size_y = int(size_y * self.ratio_y)
        size_x = int(size_x * self.ratio_x)

        self.rand_roll_y = np.random.randint(low=-size_y, high=size_y)
        self.rand_roll_x = np.random.randint(low=-size_x, high=size_x)

        structure = torch.roll(structure, self.rand_roll_x, 2)
        structure = torch.roll(structure, self.rand_roll_y, 3)
        velocity = torch.roll(velocity, self.rand_roll_x, 2)
        velocity = torch.roll(velocity, self.rand_roll_y, 3)
        return structure, velocity

    def rollback_velocity(self, velocity):
        velocity = torch.roll(velocity, -self.rand_roll_x, 2)
        velocity = torch.roll(velocity, -self.rand_roll_y, 3)
        return velocity

class RollPorous(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def roll(self, structure, velocity, roll_x, roll_y):
        _, _, size_x, size_y = structure.shape

        structure = torch.roll(structure, roll_x, 2)
        structure = torch.roll(structure, roll_y, 3)
        velocity = torch.roll(velocity, roll_x, 2)
        velocity = torch.roll(velocity, roll_y, 3)
        return structure, velocity

#random_roller = RandomRollTensor(ratio_y=0.5, ratio_x=0.5)
roller = RollPorous()

def penalty_nonzero_inside(pred, mask, aggregate=True):
    penalty = (torch.abs(pred[:, 0, :, :]) + torch.abs(pred[:, 1, :, :])) *(1.0-mask)
    if aggregate:
        return  torch.mean(penalty)
    else:
        return penalty


def get_two_random_points(max_x, max_y):
    p0x, p0y = np.random.randint(0, max_x), np.random.randint(0, max_y)
    p1x, p1y = np.random.randint(0, max_x), np.random.randint(0, max_y)
    p0x, p1x = min(p0x, p1x), max(p0x, p1x)
    p0y, p1y = min(p0y, p1y), max(p0y, p1y)
    if p0x == p1x or p0y == p1y:
        return get_two_random_points(max_x, max_y)
    else:
        return p0x, p0y, p1x, p1y

def mass_constrain_penalty(pred):
    n_integration_loops = 20
    integrals = torch.zeros(len(pred), requires_grad=True).to(pred.device)
    for _ in range(n_integration_loops):
        # FIXME: add wrap_margins25
        col1, row1, col2, row2 = get_two_random_points(max_x=256, max_y=256)
        #integrals += torch.sum(pred[:, 0, row1:(row2+1), col2], axis=1)-torch.sum(pred[:, 0, row1:(row2+1), col1], axis=1) -torch.sum(pred[:, 1, row1, col1:(col2+1)], axis=1) +torch.sum(pred[:, 1, row2, col1:(col2+1)], axis=1)
        I1 = (pred[:, 0, row1:row2, col2] + pred[:, 0, (row1 + 1):(row2 + 1), col2]) / 2
        I3 = (pred[:, 0, row1:row2, col1] + pred[:, 0, (row1 + 1):(row2 + 1), col1]) / 2
        I2 = (pred[:, 1, row1, col1:col2] + pred[:, 1, row1, (col1 + 1):(col2 + 1)]) / 2
        I4 = (pred[:, 1, row2, col1:col2] + pred[:, 1, row2, (col1 + 1):(col2 + 1)]) / 2
        # I1 - I2 - I3 + I4
        integrals += torch.square(torch.sum(I1 - I3, axis=1) + torch.sum(I4 - I2, axis=1))
    loss = torch.mean(integrals)
    return loss

def divergence_penalty(pred, aggregate=True):
    Fxdx = (torch.roll(pred[:, 1, :, :], shifts=1, dims=1) - torch.roll(pred[:, 1, :, :], shifts=-1, dims=1)) / 2
    Fydy = (torch.roll(pred[:, 0, :, :], shifts=1, dims=2) - torch.roll(pred[:, 0, :, :], shifts=-1, dims=2)) / 2
    divF = Fxdx + Fydy
    divF = torch.square(divF)
    if aggregate:
        return torch.mean(divF)
    else:
        return divF

def margin_penalty(pred):
    margin = pred.shape[2]-256
    if margin == 0:
        return torch.tensor(0, requires_grad=False)
    penalty = torch.mean(torch.square(pred[:, 0, :margin, :] - pred[:, 0, -margin:, :]) + torch.square(pred[:, 1, :margin, :] - pred[:, 1, -margin:, :]))
    penalty += torch.mean(torch.square(pred[:, 0, :, :margin] - pred[:, 0, :, -margin:]) + torch.square(pred[:, 1, :, :margin] - pred[:, 1, :, -margin:]))
    return penalty/2.0

def tortuosity(X, mask):
    #V = X[:, 1, :, :]
    #U = X[:, 0, :, :]
    #V = V*mask
    #U = U*mask
    nominator = torch.sum(torch.sqrt(X[:, 0, :, :]**2 + X[:, 1, :, :]**2)*mask, axis=[1, 2])
    denominator = (torch.sum(X[:, 0, :, :]*mask, axis=[1, 2])+1e-10)
    ratio = nominator/denominator
    return ratio
    #return torch.clamp(ratio, min=MIN_TORTUOSITY, max=MAX_TORTUOSITY)
    #return torch.sum(torch.sqrt(X[:, 0, :, :]**2 + X[:, 1, :, :]**2)*mask, axis=[1, 2]) / (torch.sum(X[:, 0, :, :]*mask, axis=[1, 2])+1e-10)

def permeability(X):
    # U = X[:,0, :, : ]
    def volumeflux2d(X):
        A = 256
        dA = 1
        i = A // 2
        Q = torch.sum(X[:, 0, :, i], axis=1)*dA/A
        return Q
    return volumeflux2d(X)*MU/FX


def tortuosity_penalty(target, pred, mask):
    return torch.mean(torch.square(tortuosity(target, mask) - tortuosity(pred, mask)))

def periodic_penalty(input, pred, model):
    # penalties = []
    # for roll_x in [256 // 2]:
    #     input_rolled, velocity_rolled = roller.roll(input, pred, roll_x=roll_x, roll_y=roll_x)
    #     velocity_rolled_pred = model(input_rolled)
    #     penalty = torch.mean(torch.square(velocity_rolled - velocity_rolled_pred))
    #     penalties.append(penalty)
    # penalties_tensor = torch.Tensor(penalties, requires_grad=True)
    # return torch.mean(penalties_tensor)
    roll_x = 256 // 2
    input_rolled, velocity_rolled = roller.roll(input, pred, roll_x=roll_x, roll_y=roll_x)
    velocity_rolled_pred = model(input_rolled)
    penalty = torch.mean(torch.square(velocity_rolled - velocity_rolled_pred))
    return penalty


def hessian(X):
    # x = torch.nn.functional.pad(pred, pad=(1,1,1,1), mode='circular')
    # https://en.wikipedia.org/wiki/Finite_difference
    # https://mathoverflow.net/questions/450606/what-are-the-best-definitions-for-smoothness-of-a-2d-curve-real-valued-function
    xp = torch.roll(X, 1, dims=2)
    xm = torch.roll(X, -1, dims=2)
    yp = torch.roll(X, 1, dims=3)
    ym = torch.roll(X, -1, dims=3)
    xpyp = torch.roll(X, 1, dims=3)
    xmym = torch.roll(X, -1, dims=3)
    # formulas according to wikipedia
    fxx = xp - 2 * X + xm
    fyy = yp - 2 * X + ym
    fxy = (xpyp - xp - yp + 2 * X - xm - ym - xmym) / 2.0
    return fxx, fyy, fxy


def hessian_penalty(target, pred):
    pred_hess_xx, pred_hess_yy, pred_hess_xy = hessian(pred)
    target_hess_xx, target_hess_yy, target_hess_xy = hessian(target)
    # weighted frobenius norm
    norm = torch.norm(target, dim=1)
    frob = torch.mean(torch.square(pred_hess_xx-target_hess_xx)+torch.square(pred_hess_yy-target_hess_yy)+2*torch.square(pred_hess_xy-target_hess_xy), dim=1)
    frob = torch.mean(frob*norm)
    return frob

def calc_loss(input, pred, target, metrics, model, factor_inside=1.0, factor_mass=1.0, factor_div=1.0, factor_tortuosity=1.0, factor_periodic=1.0, factor_margin=0.0, factor_hessian=0.0, calculate_all_metrics=False):
    basic_loss = torch.mean(torch.square(pred-target))
    mask = (input[:, 0, :, :] < 0.5)*1.0 # maska to obszar przeplywu nie przeszkody
    #mask = (input[:, 0, :, :] > 0 )*1.0 # maska to obszar przeplywu nie przeszkody
    if factor_inside > 0 or calculate_all_metrics:
        penalty_inside = penalty_nonzero_inside(pred, mask)
    else:
        penalty_inside = torch.tensor(0, requires_grad=False)
    if factor_mass > 0 or calculate_all_metrics:
        penalty_mass = mass_constrain_penalty(pred)
    else:
        penalty_mass = torch.tensor(0, requires_grad=False)
    if factor_div > 0 or calculate_all_metrics:
        penalty_div = divergence_penalty(pred)
    else:
        penalty_div = torch.tensor(0, requires_grad=False)
    if factor_tortuosity > 0 or calculate_all_metrics:
        penalty_tortuosity = tortuosity_penalty(target, pred, mask)
    else:
        penalty_tortuosity = torch.tensor(0, requires_grad=False)
    if factor_periodic > 0 or calculate_all_metrics:
        penalty_periodic = periodic_penalty(input, pred, model=model)
    else:
        penalty_periodic = torch.tensor(0, requires_grad=False)
    if factor_margin > 0 or calculate_all_metrics:
        penalty_margin = margin_penalty(pred)
    else:
        penalty_margin = torch.tensor(0, requires_grad=False)
    if factor_hessian >0 or calculate_all_metrics:
        penalty_hessian = hessian_penalty(target, pred)
    else:
        penalty_hessian = torch.tensor(0, requires_grad=False)

    loss = basic_loss + factor_inside*penalty_inside + factor_mass*penalty_mass + factor_div*penalty_div  + factor_periodic*penalty_periodic + factor_margin*penalty_margin + factor_hessian*penalty_hessian
    # add to gradient only tortuosity penalty is non-zero
    if not np.isnan(penalty_tortuosity.data.cpu().numpy()):
        loss += factor_tortuosity * penalty_tortuosity
    else:
        print(f'!!! Skipping toruosity penelaty')


    metrics['loss'] += loss.data.cpu().numpy() * len(pred)
    metrics['mse'] += basic_loss.data.cpu().numpy() * len(pred)
    metrics['penalty_inside'] += penalty_inside.data.cpu().numpy() * len(pred)
    metrics['penalty_mass'] += penalty_mass.data.cpu().numpy() * len(pred)
    metrics['penalty_div'] += penalty_div.data.cpu().numpy() * len(pred)
    metrics['penalty_tortuosity'] += penalty_tortuosity.data.cpu().numpy() * len(pred)
    metrics['penalty_periodic'] += penalty_periodic.data.cpu().numpy() * len(pred)
    metrics['penalty_margin'] += penalty_margin.data.cpu().numpy() * len(pred)
    metrics['penalty_hessian'] += penalty_hessian.data.cpu().numpy() * len(pred)
    metrics['loss_inside'] += penalty_inside.data.cpu().numpy() * len(pred)*factor_inside
    metrics['loss_mass'] += penalty_mass.data.cpu().numpy() * len(pred) * factor_mass
    metrics['loss_div'] += penalty_div.data.cpu().numpy() * len(pred) * factor_div
    metrics['loss_tortuosity'] += penalty_tortuosity.data.cpu().numpy() * len(pred) * factor_tortuosity
    metrics['loss_periodic'] += penalty_periodic.data.cpu().numpy() * len(pred) * factor_periodic
    metrics['loss_margin'] += penalty_margin.data.cpu().numpy() * len(pred) * factor_periodic
    metrics['loss_hessian'] += penalty_hessian.data.cpu().numpy() * len(pred) * factor_hessian
    metrics['n'] += len(pred)
    return loss

def calc_prediction_info(inputs, preds, targets, filenames):
    mses = torch.mean(torch.square(preds-targets), axis=[1,2,3]).detach().cpu().tolist()
    masks = (inputs[:, 0, :, :] < 0.5)*1.0 # maska to obszar przeplywu nie przeszkody
    penalty_inside = torch.mean(penalty_nonzero_inside(preds, masks, aggregate=False), axis=[1,2]).detach().cpu().tolist()
    penalty_div = torch.mean(divergence_penalty(preds, aggregate=False), axis=[1,2]).detach().cpu().tolist()
    tortuosity_pred = tortuosity(preds, masks).detach().cpu().tolist()
    tortuosity_target = tortuosity(targets, masks).detach().cpu().tolist()
    permeability_pred = permeability(preds).detach().cpu().tolist()
    permeability_target = permeability(targets).detach().cpu().tolist()
    output = {}
    for filename, mse, pen_inside, pen_div, tort_target, tort_pred, perm_target, perm_pred in zip(filenames, mses, penalty_inside, penalty_div, tortuosity_target, tortuosity_pred, permeability_target, permeability_pred):
        output[filename] = {'mse': mse,
                            'penalty_inside': pen_inside,
                            'penalty_div': pen_div,
                            'tortuosity_target': tort_target,
                            'tortuosity_pred': tort_pred,
                            'permeability_target': perm_target,
                            'permeability_pred': perm_pred,
                            }
    return output





def print_metrics(metrics, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k]))
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, dataloaders, optimizer, scheduler, num_epochs, experiment_path, device, loss_factor_mass, loss_factor_inside, loss_factor_div, loss_factor_tortuosity, loss_factor_periodic, loss_factor_margin, loss_factor_hessian, gradient_clip=0):
    best_model_wts = copy.deepcopy(model.state_dict())
    model_path = os.path.join(experiment_path, 'best_model.pth')
    history_path = os.path.join(experiment_path, 'history.csv')
    tensorboard_path = os.path.join(experiment_path, "tensorboard")
    writer = SummaryWriter(tensorboard_path)

    best_loss = np.Inf
    best_loss_epoch = 0
    early_stopping_mercy = 20
    metrics_history = {'train': [],
                       'val': []}

    for epoch in range(num_epochs):
        print('')
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, velocities, _ in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                velocities = velocities.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(inputs, outputs, velocities, metrics, factor_mass=loss_factor_mass,
                                     factor_inside=loss_factor_inside, factor_div=loss_factor_div,
                                     factor_tortuosity=loss_factor_tortuosity, factor_periodic=loss_factor_periodic,
                                     factor_margin=loss_factor_margin, factor_hessian=loss_factor_hessian,
                                     model=model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            # average metrics
            metrics_n = metrics['n']
            metrics = {x: y / epoch_samples for x, y in metrics.items()}
            metrics_history[phase].append(metrics)
            metrics['n'] = metrics_n
            epoch_loss = metrics['loss']

            # save metrics to tensorboard
            for metric_key in metrics:
                writer.add_scalar(f"{metric_key}/{phase}", metrics[metric_key], epoch)


            if phase == 'val':
                print_metrics(metrics, phase)
                print(f'Best loss epoch: {best_loss_epoch} loss={best_loss}')


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_loss_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path)
            if epoch - best_loss_epoch > early_stopping_mercy:
                print('Early stopping!')
                model.load_state_dict(best_model_wts)
                # exit due to early stopping
                return model, best_loss, metrics_history
            # TODO: MAKE THIS MORE GENERAL
            # if epoch == 30 and metrics['mse'] > 2e-4:
            #     print('Too slow progress!')
            #     model.load_state_dict(best_model_wts)
            #     # exit due to early stopping
            #     return model, best_loss, metrics_history
        # no early stopping continue with usual flow
        df_history_train = pd.DataFrame(metrics_history['train'])
        df_history_train.to_csv(history_path.replace('.csv', '_train.csv'), index=False)
        df_history_val = pd.DataFrame(metrics_history['val'])
        df_history_val.to_csv(history_path.replace('.csv', '_val.csv'), index=False)
        plot_history(df_history_train, df_history_val, experiment_path=experiment_path)

        scheduler.step()
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, metrics_history
