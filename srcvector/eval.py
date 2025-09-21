import torch
import pandas as pd
from srcvector.model import calc_loss, calc_prediction_info
from srcvector.utils import tensor_smooth_gaussian, dict_to_json
import numpy as np
from collections import defaultdict
import pickle


def evaluate(dataloader, model, device, output_path, loss_factor_mass=1.0, loss_factor_inside=1.0, loss_factor_div=1.0, loss_factor_tortosity=1.0, loss_factor_periodic=1.0, loss_factor_margin=0.0, loss_factor_hessian=0.0, prediction_filename=None, crop_to_256=False, post_smooth_sigma=0.0,
             max_predictions_written=-1, float_type='float16'):
    model.eval()
    evaluation_dict = { }
    if crop_to_256:
        output_path += '_crop256'
    all_metrics = defaultdict(float)
    all_predictions = {}
    with torch.set_grad_enabled(False):
        for inputs, target, filenames in dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            prediction = model(inputs)
            if crop_to_256:
                inputs = inputs[:, :, :256, :256]
                target = target[:, :, :256, :256]
                prediction = prediction[:, :, :256, :256]
            prediction = tensor_smooth_gaussian(prediction, sigma=post_smooth_sigma)
            _ = calc_loss(inputs, prediction, target, all_metrics, factor_mass=loss_factor_mass,
                             factor_inside=loss_factor_inside, factor_div=loss_factor_div,
                             factor_tortuosity=loss_factor_tortosity, calculate_all_metrics=True,
                          factor_periodic=loss_factor_periodic, factor_margin=loss_factor_margin,
                          factor_hessian=loss_factor_hessian,
                          model=model)
            if prediction_filename and (len(all_predictions) < max_predictions_written or max_predictions_written <= 0):
                batch_predictions = {filename: pred.cpu().numpy().astype(float_type)
                                     for filename, pred in zip(filenames, prediction)}
                all_predictions = all_predictions | batch_predictions

            evaluation_dict = evaluation_dict | calc_prediction_info(inputs=inputs, preds=prediction, targets=target, filenames=filenames)

    df = pd.DataFrame.from_dict(evaluation_dict, orient='index').reset_index().rename({'index': 'filename'}, axis=1)
    df.to_csv(output_path, index=False)

    samples_n = all_metrics['n']
    all_metrics = {x: y / samples_n for x, y in all_metrics.items()}
    all_metrics['n'] = samples_n

    # save basic metrics as json
    dict_to_json(f'{output_path}_metrics.json', all_metrics)

    # save predictions if required
    if prediction_filename:
        with open(prediction_filename, 'wb') as handle:
            pickle.dump(all_predictions, handle)

    model.train()
