import os
import argparse

import torch
import scipy.stats
import numpy as np

from datetime import datetime

from muscleabm.emulators import ModelManager, get_experiment_model_folderpath

torch.set_default_dtype(torch.float64)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix_experiment_dirpath", required=True,
        help="Matrix gridsearch experiment containing the op65 GP model and its global eigenparameter "
             "estimation, e.g. model_experiments/2026-06-03-matrix_shape."
    )
    parser.add_argument(
        "--mcmc_dirpath", required=True,
        help="Folder containing the WT and RD MCMC chains and likelihoods, e.g. "
             "model_experiments/2026-05-31-collisions_shape/mcmc_results."
    )
    return parser.parse_args()


def regularise_transforms(log_transforms, log_scale_factors, loss_histories):
    # Ensure we can make appropriate comparisons:
    unit_transforms = log_transforms / np.linalg.norm(log_transforms, axis=1, keepdims=True)
    mean_loss = np.mean(loss_histories[:, 4096:], axis=1)

    # Make base transform have the best plotting order:
    base_transform = np.copy(unit_transforms[np.argmin(mean_loss), :, :])
    base_transform = base_transform[:, [2, 0, 1]]
    base_transform[:, 1] *= -1
    base_transform[:, 2] *= -1

    # Set up collation:
    regularised_transforms = []
    regularised_scale_factors = []

    # Iterate through rest of dataset:
    for run_index in range(unit_transforms.shape[0]):
        run_transform = unit_transforms[run_index, :, :]
        cosine_matrix = base_transform.T @ run_transform

        arranged_transform = []
        arranged_sf = []
        for t_index in range(run_transform.shape[1]):
            choice_index = np.nanargmax(np.abs(cosine_matrix[t_index, :]))
            choice_sign = np.sign(cosine_matrix[t_index, choice_index])
            arranged_transform.append(run_transform[:, choice_index] * choice_sign)
            arranged_sf.append(log_scale_factors[run_index, choice_index])
            cosine_matrix[:, choice_index] = np.nan


        arranged_transform = np.stack(arranged_transform, axis=0)
        arranged_sf = np.stack(arranged_sf)
        assert(~np.any(np.isnan(arranged_transform)))
        regularised_transforms.append(arranged_transform)
        regularised_scale_factors.append(arranged_sf)

    return np.stack(regularised_transforms, axis=0), np.stack(regularised_scale_factors, axis=0)


def get_optimal_intervention(weights, model_manager, ctl_mle, rd_mle, gee_transform, ctl_gee_centres):
    # Get initial transform (converting directly to control distribution):
    inital_transform = (np.mean(ctl_mle, axis=0) / np.mean(rd_mle, axis=0))[:11]
    inital_transform = np.log(inital_transform)
    intervention_transform = torch.from_numpy(np.copy(inital_transform))
    intervention_transform.requires_grad_()

    # Get predictor for order parameter values:
    model_manager.likelihood.train()
    model_manager.model.train()

    ctl_op_preds = model_manager.likelihood(model_manager.model(torch.from_numpy(ctl_mle)))
    ctl_target_mean = torch.mean(ctl_op_preds.mean)
    ctl_target_mean = ctl_target_mean.detach()

    opt = torch.optim.Adam([intervention_transform], lr=0.003)

    # Instantiate relative weightings of different losses:
    alpha, beta, gamma = weights

    # Optimise intervention:
    loss_history = []
    intervention_history = []
    for step_index in range(1024):
        # Calculate adjusted RD:
        adjusted_rd = torch.from_numpy(rd_mle) * torch.exp(
            torch.cat([intervention_transform, torch.zeros(3)])
        )

        # Calculate complexity of transform:
        abs_intervention = torch.abs(intervention_transform)
        norm_intervention = abs_intervention / torch.sum(abs_intervention)
        ent_intervention = torch.exp(torch.sum(-norm_intervention * torch.log(norm_intervention)))

        # Calculate similarity to control space:
        adj_gee = torch.log(adjusted_rd) @ torch.from_numpy(gee_transform)
        mean_distance_components = torch.from_numpy(ctl_gee_centres) - torch.mean(adj_gee, dim=0)
        gee_distance = torch.sqrt(torch.sum(mean_distance_components ** 2))

        # Calculate similarity in order parameter:
        adj_predictions = model_manager.likelihood(model_manager.model(adjusted_rd))
        op_distance = torch.abs(ctl_target_mean - torch.mean(adj_predictions.mean))

        # Calculate magnitude of change:
        norm_transform = torch.linalg.norm(intervention_transform)

        # Update history:
        loss_history.append([
            ent_intervention.item(),
            gee_distance.item(),
            op_distance.item(),
            norm_transform.item()
        ])
        intervention_history.append(
            intervention_transform.detach().numpy().copy()
        )

        # Set up baseline to normalise losses:
        if step_index == 0:
            ent_baseline = ent_intervention.item()
            gee_baseline = gee_distance.item()
            op_baseline = op_distance.item()
            norm_baseline = norm_transform.item()

        # Combine losses:
        total_loss = \
            (ent_intervention / ent_baseline) \
            + (alpha * (gee_distance / gee_baseline)) \
            + (beta * (op_distance / op_baseline)) \
            + (gamma * (norm_transform / norm_baseline))

        # Run backprop and take update step:
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if (step_index + 1) % 32 == 0:
            print(step_index + 1)

    # Package outputs:
    loss_history = np.stack(loss_history, axis=0)
    baselines = np.array([ent_baseline, gee_baseline, op_baseline, norm_baseline])
    return intervention_transform.detach().numpy().copy(), loss_history, baselines


def main():
    args = parse_arguments()

    # Instantiate and load mean S_65 model:
    inducing_points = np.zeros((32, 14))
    op_model_manager = ModelManager(inducing_points, 0.003)
    op_model_manager.load(get_experiment_model_folderpath(args.matrix_experiment_dirpath, "op65"))

    # Set up MLE source:
    THIN_FACTOR = 64

    # Load MLE data:
    ctl_chain = np.load(os.path.join(args.mcmc_dirpath, "wt_mcmc_chain.npy"))
    rd_chain = np.load(os.path.join(args.mcmc_dirpath, "rd_mcmc_chain.npy"))
    ctl_likelihoods = np.load(os.path.join(args.mcmc_dirpath, "wt_mcmc_likelihoods.npy"))
    rd_likelihoods = np.load(os.path.join(args.mcmc_dirpath, "rd_mcmc_likelihoods.npy"))

    ctl_mle_idx = np.argsort(ctl_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]
    rd_mle_idx = np.argsort(rd_likelihoods[::THIN_FACTOR, :, 0].flatten())[-1024:]

    ctl_mle = ctl_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[ctl_mle_idx, :]
    rd_mle = rd_chain[::THIN_FACTOR, :, 0, :].reshape(-1, 11)[rd_mle_idx, :]

    ctl_mle = np.concatenate([ctl_mle, np.ones((ctl_mle.shape[0], 3)) * 0.5], axis=1)
    rd_mle =  np.concatenate([rd_mle, np.ones((rd_mle.shape[0], 3)) * 0.5], axis=1)

    # Load GEE transforms:
    log_transforms = np.load(os.path.join(
        args.matrix_experiment_dirpath,
        "gaussian_process_models", "op65",
        "global_eigenparameter_estimation", "log_transforms.npy"
    ))
    loss_histories = np.load(os.path.join(
        args.matrix_experiment_dirpath,
        "gaussian_process_models", "op65",
        "global_eigenparameter_estimation", "loss_histories.npy"
    ))
    log_scale_factors = np.load(os.path.join(
        args.matrix_experiment_dirpath,
        "gaussian_process_models", "op65",
        "global_eigenparameter_estimation", "scale_factors.npy"
    ))
    log_scale_factors = np.squeeze(log_scale_factors)
    regularised_transforms, regularised_scale_factors = regularise_transforms(
        log_transforms, log_scale_factors, loss_histories
    )

    # Get optimal transform:
    mean_loss = np.mean(loss_histories[:, 4096:], axis=1)
    gee_transform = np.copy(regularised_transforms[np.argmin(mean_loss), :, :]).T
    gee_transform /= np.linalg.norm(gee_transform, axis=0, keepdims=True)

    # Get distribution of CTL MLE in control space:
    ctl_gee = np.log(ctl_mle) @ gee_transform

    # Get mode of final GEE dimension (it is bimodal):
    dimension_range = [ctl_gee[:, 2].min(), ctl_gee[:, 2].max()]
    search_range = np.linspace(*dimension_range, 10000)
    densities = scipy.stats.gaussian_kde(ctl_gee[:, 2])(search_range)
    dimension_mode = search_range[np.argmax(densities)]
    ctl_gee_centres = np.array([*np.mean(ctl_gee[:, [0, 1]], axis=0),  dimension_mode])

    # Get optimal intervention, ignoring control space:
    BASE_WEIGHTS = [0, 0.0125, 0.15]
    base_intervention, base_loss_history, base_baselines = get_optimal_intervention(
        BASE_WEIGHTS, op_model_manager, ctl_mle, rd_mle,
        gee_transform, ctl_gee_centres
    )

    # Get optimal intervention, incorporating control space:
    GEE_WEIGHTS = [0.2, 0.025, 0.3]
    gee_intervention, gee_loss_history, gee_baselines = get_optimal_intervention(
        GEE_WEIGHTS, op_model_manager, ctl_mle, rd_mle,
        gee_transform, ctl_gee_centres
    )

    # Set up save directory:
    time_string = datetime.now().strftime("%Y-%m-%d-")
    save_dirpath = os.path.join("model_experiments", f"{time_string}intervention-experiment")
    if not os.path.exists(save_dirpath):
        os.mkdir(save_dirpath)

    np.savez(
        os.path.join(save_dirpath, "base_intervention.npz"),
        intervention=base_intervention,
        loss_history=base_loss_history,
        baselines=base_baselines
    )

    np.savez(
        os.path.join(save_dirpath, "gee_intervention.npz"),
        intervention=gee_intervention,
        loss_history=gee_loss_history,
        baselines=gee_baselines
    )


if __name__ == "__main__":
    main()
