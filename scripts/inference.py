import argparse
import os
import random
import json

import numpy as np
import torch
import time
from torch.utils.data import DataLoader

from srcvector.data import PorousDataset, PeriodicWrap
from srcvector.utils import experiment_name_with_timestamp, create_experiment_dir, dict_to_json
from srcvector.eval import evaluate
from backbonedunet.backboned_unet import Unet
from vectorpredictor import BACKBONES

# set seeds
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)



def run_inference(args):
    experiment_path = args["experiment_path"]
    model_path = os.path.join(experiment_path, 'best_model.pth')
    experiment_params = json.load(open(os.path.join(experiment_path, 'arguments.json')))
    backbone_name = experiment_params['backbone_name']
    final_conv_kernel_size = experiment_params['final_conv_kernel_size']
    smooth_mode = experiment_params["smooth_mode"]
    smooth_kernel_size = experiment_params["smooth_kernel_size"]
    post_smooth_sigma = experiment_params["post_smooth_sigma"]

    structures_dirpath = args["structures_dirpath"]
    vector_dirpath = args["vector_dirpath"]

    batch_size = args["batch_size"]
    loss_factor_inside = args["loss_factor_inside"]
    loss_factor_mass = args["loss_factor_mass"]
    loss_factor_div = args["loss_factor_div"]
    loss_factor_tortuosity = args["loss_factor_tortuosity"]
    loss_factor_periodic = args["loss_factor_periodic"]
    margin_periodic = experiment_params["margin_periodic"]
    loss_factor_margin = args["loss_factor_margin"]
    loss_factor_hessian = args["loss_factor_hessian"]
    output_dir = args["output_dir"]
    run_on_cpu = args["run_on_cpu"]

    if torch.cuda.is_available():
        device_string = "cuda:0"
    if run_on_cpu or torch.cuda.is_available() == False:
        device_string = "cpu"
    device = torch.device(device_string)

    print(f"Will run on device: {device}")

    structures_files = os.listdir(structures_dirpath)
    velocities_files = os.listdir(vector_dirpath)
    structures_files = [f.replace(".gif", "").replace(".npy", "") for f in structures_files]
    velocities_files = [f.replace(".gif.vel.npz", "") for f in velocities_files]
    # systems for which we have both structure file and velocity file
    filelist = list(set(structures_files).intersection(velocities_files))

    #filelist = filelist[:20]

    # sort filelist as os.listdir returns them in random order
    filelist = sorted(filelist)
    random.shuffle(filelist)
    print(f"Total number of files: {len(filelist)}")

    # load model
    model = Unet(backbone_name=backbone_name, classes=2, smooth_mode=smooth_mode, smooth_kernel_size=smooth_kernel_size,
                 final_conv_kernel_size=final_conv_kernel_size).to(device)

    if device_string == "cpu":
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_path, weights_only=True))


    periodic_wrap = PeriodicWrap(margin=margin_periodic)
    valtest_augmentations = [periodic_wrap]

    dataset = PorousDataset(filelist=filelist, structures_dirpath=structures_dirpath, velocity_dirpath=vector_dirpath, augmentations=valtest_augmentations)
    print('Dataset size: ', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)

    dict_to_json(os.path.join(output_dir, 'arguments.json'), experiment_params)
    dict_to_json(os.path.join(output_dir, 'arguments_inference.json'), args)

    evaluation_start = time.time()
    evaluate(
        dataloader=dataloader,
        model=model,
        device=device,
        output_path=os.path.join(output_dir, "evaluation.csv"),
        loss_factor_mass=loss_factor_mass,
        loss_factor_inside=loss_factor_inside,
        loss_factor_div=loss_factor_div,
        loss_factor_tortosity=loss_factor_tortuosity,
        loss_factor_periodic=loss_factor_periodic,
        loss_factor_margin=loss_factor_margin,
        prediction_filename=os.path.join(output_dir, 'predictions_evaluation.pickle'),
        crop_to_256=True,
        post_smooth_sigma=0.0,
        loss_factor_hessian=loss_factor_hessian,
        float_type='float32'
    )
    evaluation_end = time.time()
    print(f'Inference time: {evaluation_end-evaluation_start}')




def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path", type=str, required=True, help="Path to experiment"
    )
    parser.add_argument(
        "--structures_dirpath", type=str, required=True, help="Where structure files are kept in .gif format"
    )
    parser.add_argument("--vector_dirpath", type=str, required=True, help="Where velocity files are kept")
    parser.add_argument(
        "--output_dir", type=str, default="", help="Path to output directory where experiment direcory will be created"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument(
        "--loss_factor_inside", type=float, default=5.0, help="Penalty factor for prediction inside material"
    )
    parser.add_argument("--loss_factor_mass", type=float, default=0.0, help="Penalty for non-presentev mass")
    parser.add_argument("--loss_factor_div", type=float, default=1.0, help="Penalty for non-zero diveregnce")
    parser.add_argument("--loss_factor_tortuosity", type=float, default=1.0, help="Penalty for tortuosity")
    parser.add_argument("--loss_factor_periodic", type=float, default=1.0, help="Penalty for non periodicity")
    parser.add_argument("--margin_periodic", type=int, default=0, help="Margin for periodic system")
    parser.add_argument("--loss_factor_margin", type=float, default=0.0, help="Penalty for non-periodic conditions when margin_periodic is used")
    parser.add_argument("--loss_factor_hessian", type=float, default=0.0, help="Penalty for large curvature")
    parser.add_argument('--run_on_cpu', action='store_true', help="Force running on cpu")

    return vars(parser.parse_args())


def main():
    args = parse_command_line_args()
    if args['margin_periodic'] == 0:
        args['loss_factor_margin'] = 0
    run_inference(args)



if __name__ == "__main__":
    main()
