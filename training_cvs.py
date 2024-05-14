import os
import torch
import numpy as np
import logging
import logging.config
from munch import munchify

from data.cvs.config_cvs import load_config
from utils.ODE_dataset import create_transforms, ODEDataCSV
from utils.utils import set_seed
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from utils.plotting import individual_cvs, visualize_latent
from models.mechanistic_cvs import MechanisticModel
from models.mechanistic_cvs_Gauss import MechanisticModelGauss
from models.latent_ODE_cvs import LatentODEModel
from models.hierarchical_ODE_cvs import HierarchicalModel


def batch_to_device(d, device):
    iext = torch.unsqueeze(d["iext"], 1)
    d["iext"] = iext.to(device)

    rtpr = torch.unsqueeze(d["rtpr"], 1)
    d["rtpr"] = rtpr.to(device)

    observations = d["observations"].permute(0, 2, 1)  # swap to get obs * K * T
    d["observations"] = observations.to(device)
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
    iext_predictions, rtpr_predictions = torch.zeros(0), torch.zeros(0)
    iext_empirical, rtpr_empirical = torch.zeros(0), torch.zeros(0)
    observations = torch.zeros(0)
    mu_25, mu_50, mu_75, z, solution_xt = (
        torch.zeros(0),
        torch.zeros(0),
        torch.zeros(0),
        torch.zeros(0),
        torch.zeros(0),
    )

    total_l1 = 0
    num_losses = len(losses)
    total_elbo = [0.0] * num_losses
    size = 0
    # use the appropriate data loader
    has_classifier = config.model in ["Mechanistic", "GOKU", "MechanisticGauss"]
    for batch in data_loader:
        # use classification function to compute all predictions for each batch
        batch = batch_to_device(batch, device=device)
        observations = torch.cat((observations, batch.observations), dim=0)

        for loss_id in range(num_losses):
            new_loss = losses[loss_id].evaluate_loss(
                observations=batch.observations, iext=batch.iext, rtpr=batch.rtpr
            )
            total_elbo[loss_id] += new_loss / batch.observations.shape[0]

        results = recon_fun(
            observations=batch.observations,
            iext=batch.iext,
            rtpr=batch.rtpr,
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

        iext_empirical = torch.cat((iext_empirical, batch.iext), dim=0)
        rtpr_empirical = torch.cat((rtpr_empirical, batch.rtpr), dim=0)

        if has_classifier:
            predictions = input_pred_fn(observations=batch.observations)
            iext_predictions = torch.cat((iext_predictions, predictions["iext"]), dim=0)
            rtpr_predictions = torch.cat((rtpr_predictions, predictions["rtpr"]), dim=0)
            # compute the number of accurate predictions
    if has_classifier:
        iext_accuracy = compute_accuracy(pred=iext_predictions, emp=iext_empirical)
        rtpr_accuracy = compute_accuracy(pred=rtpr_predictions, emp=rtpr_empirical)
    else:
        iext_accuracy = np.nan
        rtpr_accuracy = np.nan
    if epoch % 100 == 0:
        data_print = "iext_empirical: {} rtpr_empirical: {} ".format(
            np.unique(iext_empirical, return_counts=True),
            np.unique(rtpr_empirical, return_counts=True),
        )
        logging.debug(data_print)
        print(data_print)

    if is_plot:
        results = {"mu_75": mu_75, "mu_50": mu_50, "mu_25": mu_25}
        individual_cvs(
            observations=observations,
            results=munchify(results),
            epoch=epoch,
            iext=iext_empirical,
            rtpr=rtpr_empirical,
            times=times,
            is_post=is_post,
            is_test=is_test,
            solution_xt=solution_xt,
            z=z,
            config=config,
        )
    return {
        "iext": iext_accuracy,
        "rtpr": rtpr_accuracy,
        "l1": total_l1 / size,
        "z": z,
        "elbo": torch.tensor(total_elbo),
    }


def run_batch(batch, losses):
    num_losses = len(losses)
    epoch_losses = [0.0] * num_losses

    for loss_id in range(num_losses):
        new_loss = losses[loss_id].step(
            observations=batch.observations, iext=batch.iext, rtpr=batch.rtpr
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
    data_transforms = create_transforms(config)
    ds_train = ODEDataCSV(
        data_dir=config.data_path,
        ds_type="train",
        seq_len=config.seq_len,
        random_start=False,  # Check if start at random place works
        transforms=data_transforms,
    )

    ds_val = ODEDataCSV(
        data_dir=config.data_path,
        ds_type="val",
        seq_len=config.seq_len,
        random_start=False,
        transforms=data_transforms,
    )

    ds_test = ODEDataCSV(
        data_dir=config.data_path,
        ds_type="test",
        seq_len=config.seq_len,
        random_start=False,
        transforms=data_transforms,
    )

    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=config.mini_batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        ds_val, batch_size=len(ds_val), shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=len(ds_test), shuffle=False
    )

    # Create Model
    times = torch.arange(
        0.0, end=config.seq_len * config.delta_t, step=config.delta_t, device=device
    )
    if config.model in ["GOKU", "Latent"]:
        selected = LatentODEModel
    elif config.model == "Hierarchical":
        selected = HierarchicalModel
    elif config.model == "Mechanistic":
        selected = MechanisticModel
    elif config.model == "MechanisticGauss":
        selected = MechanisticModelGauss
    else:
        raise ValueError("selected model is not implemented")
    var_model = selected(config=config, device=device, times=times).to(device)
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
    best_model = selected(config=config, device=device, times=times).to(device)
    best_val_acc = 0
    best_epoch = 0
    best_val_loss = np.inf

    #  Setup Pyro model
    ELBO = Trace_ELBO
    elbo = ELBO(num_particles=config.num_particles)
    loss_basic = SVI(var_model.model, var_model.guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    if config.model in ["Mechanistic", "GOKU", "MechanisticGauss"]:
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
        val_stats = input_pred_stats(
            data_loader=val_dataloader,
            input_pred_fn=var_model.classifier,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=is_val_plot,
            times=times,
            is_post=True,
            losses=losses,
        )

        _ = input_pred_stats(
            data_loader=val_dataloader,
            input_pred_fn=var_model.classifier,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=is_val_plot,
            times=times,
            is_post=False,
            losses=losses,
        )
        train_stats_post = input_pred_stats(
            data_loader=train_dataloader,
            input_pred_fn=var_model.classifier,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=False,
            times=times,
            is_post=True,
            losses=losses,
        )

        train_stats_prior = input_pred_stats(
            data_loader=train_dataloader,
            input_pred_fn=var_model.classifier,
            device=device,
            recon_fun=var_model.recon,
            epoch=epoch,
            is_plot=False,
            times=times,
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

        val_elbo = torch.sum(val_stats["elbo"]) * len(val_stats["elbo"])
        improved = ""
        if best_val_loss >= val_elbo:
            best_val_loss = val_elbo
            best_epoch = epoch
            best_model.load_state_dict(var_model.state_dict())
            improved = "*"

        # Mean train ELBO loss over all epoch
        epoch_mean_loss = np.mean(epoch_loss_array)

        summary_print = (
            "[Epoch %d/%d] loss= %.4f  iext_acc=(%.4f,%.4f)  rtpr_acc=(%.4f,%.4f) l1=(%.6f,%.6f), %s"
            % (
                epoch,
                config.num_epochs,
                epoch_mean_loss,
                train_stats_post["iext"],
                val_stats["iext"],
                train_stats_post["rtpr"],
                val_stats["rtpr"],
                train_stats_post["l1"],
                val_stats["l1"],
                improved,
            )
        )
        print(summary_print)
        logging.debug(summary_print)

    ## Evaluate on test
    test_stats_post = input_pred_stats(
        data_loader=test_dataloader,
        input_pred_fn=best_model.classifier,
        device=device,
        recon_fun=best_model.recon,
        epoch=best_epoch,
        is_plot=True,
        times=times,
        is_post=True,
        is_test=True,
        losses=losses,
    )
    test_stats_prior = input_pred_stats(
        data_loader=test_dataloader,
        input_pred_fn=best_model.classifier,
        device=device,
        recon_fun=best_model.recon,
        epoch=best_epoch,
        is_plot=True,
        times=times,
        is_post=False,
        is_test=True,
        losses=losses,
    )

    final_test = (
        "FINAL TEST: iext_acc=(%.4f,%.4f)  rtpr_acc=(%.4f,%.4f) l1=(%.6f,%.6f)"
        % (
            test_stats_post["iext"],
            test_stats_prior["iext"],
            test_stats_post["rtpr"],
            test_stats_prior["rtpr"],
            test_stats_post["l1"],
            test_stats_prior["l1"],
        )
    )
    print(final_test)
    logging.debug(final_test)
    print_elbo = "ELBO: best_epoch: {} post: {} prior: {}".format(
        best_epoch, test_stats_post["elbo"], test_stats_prior["elbo"]
    )
    print(print_elbo)
    logging.debug(print_elbo)


if __name__ == "__main__":
    config = load_config()

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
    set_seed(config.seed)
    train(config)
