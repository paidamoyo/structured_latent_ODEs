import os
import torch
import torch.nn as nn
import numpy as np
import logging
import logging.config
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from munch import munchify
from torch.utils.data import DataLoader

from utils.utils import set_seed
from data.proc.config_proc import load_config
from models.mechanistic_proc import MechanisticModel
from models.mechanistic_proc_Gauss import MechanisticModelGauss
from utils.proc_dataset import build_datasets

from utils.plotting import individual_proc, visualize_latent
from munch import munchify


# https://tech.dsmenders.com/tips-and-tricks-pyro-tutorials-1-6-0-documentation/


def batch_to_device(d, device):
    d["dev_1hot"] = d["dev_1hot"].to(device)
    d["aR"] = d["dev_1hot"][:, :3]
    d["aS"] = d["dev_1hot"][:, 3:]
    d["inputs"] = d["inputs"].to(device)
    d["C12"] = torch.unsqueeze(d["inputs"][:, 0], 1)
    d["C6"] = torch.unsqueeze(d["inputs"][:, 1], 1)
    d["observations"] = d["observations"].to(device)
    return munchify(d)


def compute_accuracy(pred, emp):
    accurate_preds = 0
    size = pred.size(0)
    # import ipdb;
    # ipdb.set_trace()
    for pred_i, act_i in zip(pred, emp):
        if pred_i == act_i:
            accurate_preds += 1
    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / size
    return accuracy


def compute_mse(pred, emp):
    mse_func = nn.MSELoss()
    return mse_func(pred, emp)


def input_pred_stats(
    data_loader,
    input_pred_fn,
    recon_fun,
    device,
    epoch,
    is_plot,
    times,
    is_post,
    losses,
    is_test=False,
):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    aR_predictions, aS_predictions = torch.zeros(0), torch.zeros(0)
    aR_empirical, aS_empirical = torch.zeros(0), torch.zeros(0)

    C12_predictions, C6_predictions = torch.zeros(0), torch.zeros(0)
    C12_empirical, C6_empirical = torch.zeros(0), torch.zeros(0)

    observations, treatments, devices = torch.zeros(0), torch.zeros(0), torch.zeros(0)
    mu_25, mu_50, mu_75, z, solution_xt = (
        torch.zeros(0),
        torch.zeros(0),
        torch.zeros(0),
        torch.zeros(0),
        torch.zeros(0),
    )
    aR, aS, C12, C6 = torch.zeros(0), torch.zeros(0), torch.zeros(0), torch.zeros(0)

    total_l1, size = 0, 0
    num_losses = len(losses)
    total_elbo = [0.0] * num_losses
    # use the appropriate data loader
    has_input_predictor = config.model in ["Mechanistic", "MechanisticGauss"]
    for batch in data_loader:
        # use classification function to compute all predictions for each batch
        batch = batch_to_device(batch, device=device)

        predictions = input_pred_fn(observations=batch.observations)
        observations = torch.cat((observations, batch.observations), dim=0)
        treatments = torch.cat((treatments, batch.inputs), dim=0)
        devices = torch.cat((devices, batch.dev_1hot), dim=0)
        aR = torch.cat((aR, batch.aR), dim=0)
        aS = torch.cat((aS, batch.aS), dim=0)
        C12 = torch.cat((C12, batch.C12), dim=0)
        C6 = torch.cat((C6, batch.C6), dim=0)

        for loss_id in range(num_losses):
            new_loss = losses[loss_id].evaluate_loss(
                observations=batch.observations,
                aR=batch.aR,
                aS=batch.aS,
                C12=batch.C12,
                C6=batch.C6,
            )
            total_elbo[loss_id] += new_loss / batch.observations.shape[0]

        results = recon_fun(
            observations=batch.observations,
            aR=batch.aR,
            aS=batch.aS,
            C12=batch.C12,
            C6=batch.C6,
            is_post=is_post,
        )
        mu_25 = torch.cat((mu_25, results["mu_25"]), dim=0)
        mu_50 = torch.cat((mu_50, results["mu_50"]), dim=0)
        mu_75 = torch.cat((mu_75, results["mu_75"]), dim=0)
        solution_xt = torch.cat((solution_xt, results["solution_xt"]), dim=0)
        z = torch.cat((z, results["z"]), dim=0)
        l1 = results["l1"]
        total_l1 += l1
        size += len(batch.observations)

        def get_idx(arr):
            _, idx = torch.topk(arr, 1)
            return idx

        aR_empirical = torch.cat((aR_empirical, get_idx(batch.aR)), dim=0)
        aS_empirical = torch.cat((aS_empirical, get_idx(batch.aS)), dim=0)
        C12_empirical = torch.cat((C12_empirical, batch.C12), dim=0)
        C6_empirical = torch.cat((C6_empirical, batch.C6), dim=0)

        if has_input_predictor:
            aR_predictions = torch.cat(
                (aR_predictions, get_idx(predictions["aR"])), dim=0
            )
            aS_predictions = torch.cat(
                (aS_predictions, get_idx(predictions["aS"])), dim=0
            )
            C12_predictions = torch.cat((C12_predictions, predictions["C12"]), dim=0)
            C6_predictions = torch.cat((C6_predictions, predictions["C6"]), dim=0)

    # compute the number of accurate predictions
    if has_input_predictor:
        aR_accuracy = compute_accuracy(pred=aR_predictions, emp=aR_empirical)
        aS_accuracy = compute_accuracy(pred=aS_predictions, emp=aS_empirical)
        C12_mse = compute_mse(pred=C12_predictions, emp=C12_empirical)
        C6_mse = compute_mse(pred=C6_predictions, emp=C6_empirical)
    else:
        aR_accuracy = np.nan
        aS_accuracy = np.nan
        C12_mse = np.nan
        C6_mse = np.nan
    if epoch % 100 == 0:
        data_print = "aR_empirical: {} aS_empirical: {} ".format(
            np.unique(aR_empirical, return_counts=True),
            np.unique(aS_empirical, return_counts=True),
        )
        logging.debug(data_print)
        print(data_print)

    if is_plot:
        results = {"mu_75": mu_75, "mu_50": mu_50, "mu_25": mu_25}
        individual_proc(
            observations=observations,
            results=munchify(results),
            epoch=epoch,
            treatments=treatments,
            devices=devices,
            config=config,
            times=times,
            is_post=is_post,
            is_test=is_test,
            solution_xt=solution_xt,
            z=z,
        )
    if is_test:
        multiple_samples(
            C12=C12,
            C6=C6,
            aR=aR,
            aS=aS,
            is_post=is_post,
            observations=observations,
            recon_fun=recon_fun,
        )

    return {
        "aR": aR_accuracy,
        "aS": aS_accuracy,
        "l1": total_l1 / size,
        "C12": C12_mse,
        "C6": C6_mse,
        "z": z,
        "elbo": total_elbo,
    }


def multiple_samples(C12, C6, aR, aS, is_post, observations, recon_fun):
    mu_25, mu_50, mu_75 = torch.zeros(0), torch.zeros(0), torch.zeros(0)

    num_samples = config.num_samples
    for i in range(num_samples):
        results = recon_fun(
            observations=observations, aR=aR, aS=aS, C12=C12, C6=C6, is_post=is_post
        )
        mu_25 = torch.cat((mu_25, torch.unsqueeze(results["mu_25"], 3)), dim=3)
        mu_50 = torch.cat((mu_50, torch.unsqueeze(results["mu_50"], 3)), dim=3)
        mu_75 = torch.cat((mu_75, torch.unsqueeze(results["mu_75"], 3)), dim=3)
    mu_50 = mu_50.detach().cpu().numpy()
    mu_75 = mu_75.detach().cpu().numpy()
    mu_25 = mu_25.detach().cpu().numpy()
    print("multiple samples: ", mu_75.shape)
    tag = "post_sample" if is_post else "prior_sample"
    np.save(file="results_{}/mu_50_{}".format(config.model, tag), arr=mu_50)
    np.save(file="results_{}/mu_75_{}".format(config.model, tag), arr=mu_75)
    np.save(file="results_{}/mu_25_{}".format(config.model, tag), arr=mu_25)


def run_batch(batch, losses):
    num_losses = len(losses)
    epoch_losses = [0.0] * num_losses

    for loss_id in range(num_losses):
        new_loss = losses[loss_id].step(
            observations=batch.observations,
            aR=batch.aR,
            aS=batch.aS,
            C12=batch.C12,
            C6=batch.C6,
        )
        epoch_losses[loss_id] += new_loss / batch.observations.shape[0]
    # see how long it took
    return epoch_losses


def train(config):
    # General settings
    print(config)
    logging.debug(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create train and test datasets:
    data = build_datasets(config)

    train_dataloader = DataLoader(
        dataset=data.train, batch_size=config.mini_batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=data.test, batch_size=config.mini_batch_size, shuffle=False
    )

    # Create Model
    if config.model == "Mechanistic":
        selected = MechanisticModel
    elif config.model == "MechanisticGauss":
        selected = MechanisticModelGauss
    else:
        raise ValueError("selected model is not implemented")
    var_model = selected(config=config, device=device, times=data.times).to(device)
    model_print = "Model: %s -  with %d parameters." % (
        config.model,
        sum(p.numel() for p in var_model.parameters()),
    )
    print(model_print)
    logging.debug(model_print)

    print(var_model)
    logging.debug(var_model)

    # Create optimizer
    adam_params = {"lr": config.learning_rate, "betas": (0.9, 0.999)}
    optimizer = Adam(adam_params)

    # Loss error on validation set (not test set!) for early stopping
    best_model = selected(config=config, device=device, times=data.times).to(device)
    best_val_loss = np.inf

    #  Setup Pyro model
    ELBO = Trace_ELBO
    elbo = ELBO(num_particles=config.num_particles)
    loss_basic = SVI(var_model.model, var_model.guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    if config.model in ["Mechanistic", "MechanisticGauss"]:
        # ELBO = JitTrace_ELBO if args.jit else Trace_ELBO
        ELBO = Trace_ELBO
        # elbo = ELBO(num_particles=args.num_particles, retain_graph=True)
        elbo = ELBO(num_particles=config.num_particles)
        loss_aux = SVI(var_model.model_meta, var_model.guide_meta, optimizer, loss=elbo)
        losses.append(loss_aux)

    print_losses = "Losses: {}".format(len(losses))
    print(print_losses)
    logging.debug(print_losses)

    # Run epochs
    for epoch in range(config.num_epochs + 1):
        epoch_loss_array = []
        for i_batch, mini_batch in enumerate(train_dataloader):
            mini_batch = batch_to_device(mini_batch, device=device)

            # Forward step
            average_loss = run_batch(batch=mini_batch, losses=losses)

            # Statistics
            epoch_loss_array.append(average_loss)

        # Calculate validation loss
        is_val_plot = epoch % config.plot_epoch == 0
        val_stats_post = input_pred_stats(
            data_loader=val_dataloader,
            input_pred_fn=var_model.pred_inputs,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=is_val_plot,
            times=data.times,
            is_post=True,
            losses=losses,
        )

        _ = input_pred_stats(
            data_loader=val_dataloader,
            input_pred_fn=var_model.pred_inputs,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=is_val_plot,
            times=data.times,
            is_post=False,
            losses=losses,
        )

        train_stats_post = input_pred_stats(
            data_loader=train_dataloader,
            input_pred_fn=var_model.pred_inputs,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=False,
            times=data.times,
            is_post=True,
            losses=losses,
        )

        train_stats_prior = input_pred_stats(
            data_loader=train_dataloader,
            input_pred_fn=var_model.pred_inputs,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=False,
            times=data.times,
            is_post=False,
            losses=losses,
        )

        if is_val_plot:
            visualize_latent(
                z_prior=train_stats_prior["z"],
                z_post=train_stats_post["z"],
                config=config,
                epoch=epoch,
            )

        # Mean train ELBO loss over all epoch
        epoch_mean_loss = np.mean(epoch_loss_array)

        str = ""
        val_elbo = np.sum(val_stats_post["elbo"])
        if val_elbo < best_val_loss and config.heldout is None:
            best_val_loss = val_elbo
            ## Save model and run hyper parameters
            best_epoch = epoch + 1
            print(f"update best epoch={best_epoch} val_loss: {best_val_loss}")
            best_model.load_state_dict(var_model.state_dict())
            str = "*"
        elif config.heldout is not None:
            best_epoch = epoch + 1
            print(f"update epoch={best_epoch} val_loss: {val_elbo}")
            best_model.load_state_dict(
                var_model.state_dict()
            )  # Updates every epoch (Assumes zero shot)

        summary_print = (
            "[Epoch %d/%d] loss= %.4f  aR_acc=(%.4f,%.4f)  aS_acc=(%.4f,%.4f) l1=(%.6f,%.6f) "
            "C12_mse=(%.4f,%.4f) C6_mse=(%.4f,%.4f) %s"
            % (
                epoch + 1,
                config.num_epochs,
                epoch_mean_loss,
                train_stats_post["aR"],
                val_stats_post["aR"],
                train_stats_post["aS"],
                val_stats_post["aS"],
                train_stats_post["l1"],
                val_stats_post["l1"],
                train_stats_post["C12"],
                val_stats_post["C12"],
                train_stats_post["C6"],
                val_stats_post["C6"],
                str,
            )
        )
        print(summary_print)
        logging.debug(summary_print)

    ## Evaluate on test
    test_stats_post = input_pred_stats(
        data_loader=val_dataloader,
        input_pred_fn=best_model.pred_inputs,
        device=device,
        recon_fun=best_model.recon,
        epoch=best_epoch,
        is_plot=True,
        times=data.times,
        is_post=True,
        is_test=True,
        losses=losses,
    )
    test_stats_prior = input_pred_stats(
        data_loader=val_dataloader,
        input_pred_fn=best_model.pred_inputs,
        device=device,
        recon_fun=best_model.recon,
        epoch=best_epoch,
        is_plot=True,
        times=data.times,
        is_post=False,
        is_test=True,
        losses=losses,
    )

    final_test = (
        "FINAL TEST: aR_acc=(%.4f,%.4f)  aS_acc=(%.4f,%.4f) C12_mse=(%.4f,%.4f) C6_mse=(%.4f,%.4f) l1=(%.6f,%.6f) "
        % (
            test_stats_post["aR"],
            test_stats_prior["aR"],
            test_stats_post["aS"],
            test_stats_prior["aS"],
            test_stats_post["C12"],
            test_stats_prior["C12"],
            test_stats_post["C6"],
            test_stats_prior["C6"],
            test_stats_post["l1"],
            test_stats_prior["l1"],
        )
    )
    print(final_test)
    logging.debug(final_test)
    print_elbo = "ELBO: post: {} prior: {}".format(
        test_stats_post["elbo"], test_stats_prior["elbo"]
    )
    print(print_elbo)
    logging.debug(print_elbo)


if __name__ == "__main__":
    config = load_config()
    set_seed(config.seed)
    results_path = "./results_{}".format(config.model)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    log_file = "results_{}/model.log".format(config.model)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
        }
    )
    logging.basicConfig(filename=log_file, filemode="w", level=logging.DEBUG)
    train(config)
