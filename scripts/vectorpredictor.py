import argparse
import os
import random
import numpy as np
import torch


import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from srcvector.data import PorousDataset, RandomFlip, RandomRoll, PeriodicWrap
from srcvector.model import train_model
from srcvector.plots import plot_sample_predictions, vtk_sample_predictions
from srcvector.utils import experiment_name_with_timestamp, create_experiment_dir, dict_to_json
from srcvector.eval import evaluate
from backbonedunet.backboned_unet import Unet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BACKBONES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg16",
    "vgg19",
    "inception_v3",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]


def run_experiment(args):
    backbone_name = args["backbone_name"]
    structures_dirpath = args["structures_dirpath"]
    vector_dirpath = args["vector_dirpath"]
    n_limit = args["n_limit"]
    batch_size = args["batch_size"]
    lr = args["lr"]
    max_epochs = args["max_epochs"]
    loss_factor_inside = args["loss_factor_inside"]
    loss_factor_mass = args["loss_factor_mass"]
    loss_factor_div = args["loss_factor_div"]
    loss_factor_tortuosity = args["loss_factor_tortuosity"]
    loss_factor_periodic = args["loss_factor_periodic"]
    minimal_porosity = args["minimal_porosity"]
    margin_periodic = args["margin_periodic"]
    loss_factor_margin = args["loss_factor_margin"]
    loss_factor_hessian = args["loss_factor_hessian"]
    p_flip = args["p_flip"]
    roll_ratio_x = args["roll_ratio_x"]
    roll_ratio_y = args["roll_ratio_y"]
    smooth_mode = args["smooth_mode"]
    smooth_kernel_size = args["smooth_kernel_size"]
    post_smooth_sigma = args["post_smooth_sigma"]
    final_conv_kernel_size = args["final_conv_kernel_size"]
    max_predictions_written = args["max_predictions_written"]
    gradient_clip = args["gradient_clip"]
    weigths = args["weights"]

    if max_predictions_written < 0:
        max_predictions_written = np.Inf

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Will run on device: {device}")
    experiment_path = create_experiment_dir(args["output_dir"], args["experiment_name"])
    dict_to_json(os.path.join(experiment_path, "arguments.json"), args)

    structures_files = os.listdir(structures_dirpath)
    velocities_files = os.listdir(vector_dirpath)
    structures_files = [f.replace(".gif", "").replace(".npy", "") for f in structures_files]
    velocities_files = [f.replace(".gif.vel.npz", "") for f in velocities_files]
    # systems for which we have both structure file and velocity file
    filelist = list(set(structures_files).intersection(velocities_files))
    # filter structures with too low porosity
    filelist = [filename for filename in filelist if float(filename.split("trivialporosity=")[1]) > minimal_porosity]
    # sort filelist as os.listdir returns them in random order
    filelist = sorted(filelist)
    random.shuffle(filelist)
    if n_limit > 0:
        filelist = filelist[:n_limit]
    print(f"Total number of files: {len(filelist)}")

    # train, val, test, split
    test_size = 0.15
    val_size = 0.15
    filelist_trainval, filelist_test = train_test_split(filelist, test_size=test_size)
    filelist_train, filelist_val = train_test_split(
        filelist_trainval, test_size=val_size / (1 - test_size)
    )
    print(len(filelist_train), len(filelist_val), len(filelist_test))

    random_flip = RandomFlip(p_flip=p_flip)
    random_roll = RandomRoll(ratio_x=roll_ratio_x, ratio_y=roll_ratio_y)
    # we don't use PeriodicWrap
    periodic_wrap = PeriodicWrap(margin=margin_periodic)
    train_augmentations = [random_flip, random_roll, periodic_wrap]
    valtest_augmentations = [periodic_wrap]
    datasets = {
        "train": PorousDataset(
            filelist=filelist_train, structures_dirpath=structures_dirpath, velocity_dirpath=vector_dirpath, augmentations=train_augmentations,
        ),
        "val": PorousDataset(
            filelist=filelist_val, structures_dirpath=structures_dirpath, velocity_dirpath=vector_dirpath, augmentations=valtest_augmentations,
        ),
        "test": PorousDataset(
            filelist=filelist_test, structures_dirpath=structures_dirpath, velocity_dirpath=vector_dirpath, augmentations=valtest_augmentations,
        ),
    }
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val", "test"]}
    print(">> Dataloaders sizes")
    print(dataset_sizes)
    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0),
    }

    # start training
    model = Unet(backbone_name=backbone_name, classes=2, smooth_mode=smooth_mode, smooth_kernel_size=smooth_kernel_size,
                 final_conv_kernel_size=final_conv_kernel_size, weigths=weigths).to(device)
    if weigths not in ['random', 'pretrained']:
        print(f">> Loading weights from file: {weigths}")
        model.load_state_dict(torch.load(weigths, weights_only=True))
        print(f">> Weights loaded")

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    #optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.6)

    best_model, best_loss, history = train_model(
        model,
        dataloaders,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=max_epochs,
        experiment_path=experiment_path,
        device=device,
        loss_factor_inside=loss_factor_inside,
        loss_factor_mass=loss_factor_mass,
        loss_factor_div=loss_factor_div,
        loss_factor_tortuosity=loss_factor_tortuosity,
        loss_factor_periodic=loss_factor_periodic,
        loss_factor_margin=loss_factor_margin,
        loss_factor_hessian=loss_factor_hessian,
        gradient_clip=gradient_clip,
    )

    evaluate(
        dataloader=dataloaders["train"],
        model=best_model,
        device=device,
        output_path=os.path.join(experiment_path, "eval_train.csv"),
        loss_factor_mass=loss_factor_mass,
        loss_factor_inside=loss_factor_inside,
        loss_factor_div=loss_factor_div,
        loss_factor_tortosity=loss_factor_tortuosity,
        loss_factor_periodic=loss_factor_periodic,
        loss_factor_margin=loss_factor_margin,
        post_smooth_sigma=0.0,
        loss_factor_hessian=loss_factor_hessian,
        max_predictions_written=0,
    )
    evaluate(
        dataloader=dataloaders["val"],
        model=best_model,
        device=device,
        output_path=os.path.join(experiment_path, "eval_val.csv"),
        loss_factor_mass=loss_factor_mass,
        loss_factor_inside=loss_factor_inside,
        loss_factor_div=loss_factor_div,
        loss_factor_tortosity=loss_factor_tortuosity,
        loss_factor_periodic=loss_factor_periodic,
        loss_factor_margin=loss_factor_margin,
        post_smooth_sigma=0.0,
        loss_factor_hessian=loss_factor_hessian,
        max_predictions_written=0,
    )
    evaluate(
        dataloader=dataloaders["test"],
        model=best_model,
        device=device,
        output_path=os.path.join(experiment_path, "eval_test.csv"),
        loss_factor_mass=loss_factor_mass,
        loss_factor_inside=loss_factor_inside,
        loss_factor_div=loss_factor_div,
        loss_factor_tortosity=loss_factor_tortuosity,
        loss_factor_periodic=loss_factor_periodic,
        loss_factor_margin=loss_factor_margin,
        prediction_filename=os.path.join(experiment_path, 'predictions_test.pickle'),
        post_smooth_sigma=0.0,
        loss_factor_hessian=loss_factor_hessian,
        max_predictions_written=0,
    )
    evaluate(
        dataloader=dataloaders["test"],
        model=best_model,
        device=device,
        output_path=os.path.join(experiment_path, "eval_test.csv"),
        loss_factor_mass=loss_factor_mass,
        loss_factor_inside=loss_factor_inside,
        loss_factor_div=loss_factor_div,
        loss_factor_tortosity=loss_factor_tortuosity,
        loss_factor_periodic=loss_factor_periodic,
        loss_factor_margin=loss_factor_margin,
        prediction_filename=os.path.join(experiment_path, 'predictions_test.pickle'),
        crop_to_256=True,
        post_smooth_sigma=0.0,
        loss_factor_hessian=loss_factor_hessian,
        max_predictions_written=max_predictions_written,
    )
    # evaluate(
    #     dataloader=dataloaders["test"],
    #     model=best_model,
    #     device=device,
    #     output_path=os.path.join(experiment_path, "eval_test_smooth.csv"),
    #     loss_factor_mass=loss_factor_mass,
    #     loss_factor_inside=loss_factor_inside,
    #     loss_factor_div=loss_factor_div,
    #     loss_factor_tortosity=loss_factor_tortuosity,
    #     loss_factor_periodic=loss_factor_periodic,
    #     loss_factor_margin=loss_factor_margin,
    #     prediction_filename=os.path.join(experiment_path, 'predictions_test_smooth.pickle'),
    #     crop_to_256=True,
    #     post_smooth_sigma=post_smooth_sigma,
    #     loss_factor_hessian=loss_factor_hessian,
    # )
    # plot_sample_predictions(
    #     model=best_model,
    #     dataloader=dataloaders["train"],
    #     experiment_path=experiment_path,
    #     dataloader_name="train",
    #     device=device,
    # )
    # plot_sample_predictions(
    #     model=best_model,
    #     dataloader=dataloaders["test"],
    #     experiment_path=experiment_path,
    #     dataloader_name="test",
    #     device=device,
    # )
    plot_sample_predictions(
        model=best_model,
        dataloader=dataloaders["test"],
        experiment_path=experiment_path,
        dataloader_name="test",
        device=device,
        crop_to_256=True
    )
    vtk_sample_predictions(
        model=best_model,
        dataloader=dataloaders["test"],
        experiment_path=experiment_path,
        dataloader_name="test",
        device=device,
        crop_to_256=True,
        post_smooth_sigma=post_smooth_sigma,
    )




def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the training experiment")
    parser.add_argument(
        "--backbone_name", choices=BACKBONES, default="resnet18", help="Name of the training experiment"
    )
    parser.add_argument(
        "--structures_dirpath", type=str, required=True, help="Where structure files are kept in .gif format"
    )
    parser.add_argument("--vector_dirpath", type=str, required=True, help="Where velocity files are kept")
    parser.add_argument(
        "--output_dir", type=str, default="", help="Path to output directory where experiment direcory will be created"
    )
    parser.add_argument("--minimal_porosity", type=float, default=0.0)
    parser.add_argument("--n_limit", type=int, default=0, help="Limit number of structures to run")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gradient_clip", type=float, default=2.0, help="Gradient clip")
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
    parser.add_argument("--p_flip", type=float, default=0.5, help="Proba of vertical flip")
    parser.add_argument("--roll_ratio_x", type=float, default=0.0, help="Horizonalt roll ratio")
    parser.add_argument("--roll_ratio_y", type=float, default=0.0, help="Vertical roll ratio")
    parser.add_argument("--smooth_mode", choices=['none', 'gaussian_fixed', 'gaussian'], default='none')
    parser.add_argument("--smooth_kernel_size", type=int, default=3)
    parser.add_argument("--post_smooth_sigma", type=float, default=0.0)
    parser.add_argument("--final_conv_kernel_size", type=int, default=3)
    parser.add_argument("--max_predictions_written", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--weights", type=str, default="random")
    return vars(parser.parse_args())

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_command_line_args()
    args["experiment_name"] = experiment_name_with_timestamp(args["name"])
    set_seeds(seed=args['seed'])
    if args['margin_periodic'] == 0:
        args['loss_factor_margin'] = 0
    run_experiment(args)



if __name__ == "__main__":
    main()
