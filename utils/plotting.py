import seaborn as sns
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

sns.set_style('white')
sns.set_context('paper')
sns.set()
from sklearn.manifold import TSNE

legend_size = 14


def individual_cvs(observations, results, iext, rtpr, times, epoch, is_post, is_test, solution_xt, z, config):
    # print("plotting individual")
    '''Multi-panel plot for each sample'''
    observations = observations.detach().cpu().numpy()
    iext = iext.detach().cpu().numpy()
    rtpr = rtpr.detach().cpu().numpy()
    solution_xt = solution_xt.detach().cpu().numpy()
    z = z.detach().cpu().numpy()

    # print(np.max(observations), np.min(observations))
    mesh = np.array(np.meshgrid(np.unique(iext), np.unique(rtpr)))
    combinations = mesh.T.reshape(-1, 2)
    # print(combinations)
    select_idx = np.array([])
    for c in combinations:
        sel_iext = c[0]
        sel_rtpr = c[1]
        idx = np.where((np.logical_and(iext == sel_iext, rtpr == sel_rtpr)))[0]
        n_plots = 3
        select_idx = np.append(select_idx, idx[0:n_plots])

    select_idx = select_idx.astype(int)

    plot_by_label(epoch=epoch, iext=iext, rtpr=rtpr, results=results, observations=observations, times=times,
                  is_post=is_post, select_idx=select_idx, is_test=is_test, solution_xt=solution_xt, z=z, config=config)
    return


def individual_challenge(observations, results, shedding, symptoms, times, epoch, is_post, is_test, solution_xt, z,
                         config):
    # print("plotting individual")
    '''Multi-panel plot for each sample'''
    observations = observations.detach().cpu().numpy()
    shedding = shedding.detach().cpu().numpy()
    symptoms = symptoms.detach().cpu().numpy()
    solution_xt = solution_xt.detach().cpu().numpy()
    z = z.detach().cpu().numpy()

    # print(np.max(observations), np.min(observations))
    mesh = np.array(np.meshgrid(np.unique(shedding), np.unique(symptoms)))
    combinations = mesh.T.reshape(-1, 2)
    # print(combinations)
    select_idx = np.array([])
    for c in combinations:
        sel_shedding = c[0]
        sel_symptoms = c[1]
        idx = np.where((np.logical_and(shedding == sel_shedding, symptoms == sel_symptoms)))[0]
        n_plots = 3
        select_idx = np.append(select_idx, idx[0:n_plots])

    select_idx = select_idx.astype(int)

    plot_by_label_challenge(epoch=epoch, shedding=shedding, symptoms=symptoms, results=results,
                            observations=observations, times=times, is_post=is_post, select_idx=select_idx,
                            is_test=is_test, solution_xt=solution_xt, z=z,
                            config=config)
    return


def plot_by_label(epoch, iext, results, observations, rtpr, select_idx, times, is_post, is_test, solution_xt, z,
                  config):
    mu_50 = results.mu_50.detach().cpu().numpy()
    mu_75 = results.mu_75.detach().cpu().numpy()
    mu_25 = results.mu_25.detach().cpu().numpy()

    columns = np.arange(observations.shape[1])
    rows = np.arange(len(select_idx))

    colors = ['tab:gray', 'r', 'y', 'c']
    fs = 14
    nplots = 10
    # rows = rows[0:nplots]
    # idx = idx[0:nplots]
    plt.clf()
    fig, axs = plt.subplots(len(rows), len(columns), sharex=True, sharey=True, figsize=(12, 20))
    mesh = np.array(np.meshgrid(rows, columns))
    combinations = mesh.T.reshape(-1, 2)
    for r, c in combinations:
        loc = select_idx[r]
        empirical = observations[loc, c, :]

        axs[r, c].plot(times, empirical, 'k.', markersize=2)
        axs[r, c].plot(times, mu_50[loc, c, :], '-', lw=2, alpha=0.75, color=colors[c])
        axs[r, c].plot(times, mu_75[loc, c, :], '-.', lw=2, alpha=0.75, color=colors[c])
        axs[r, c].plot(times, mu_25[loc, c, :], '-.', lw=2, alpha=0.75, color=colors[c])

        axs[r, c].set_xlim(0.0, max(times) + 0.01)
        axs[r, c].set_ylim(-0.01, 1.01)
        axs[r, c].set_xticks([0, 20, 40, 60, 80])
        axs[r, c].tick_params(axis='both', which='major', labelsize=fs)
    cols = ['Pa', 'Pv', 'fHR']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    rows = ["IR={},{}".format(int(row[0]), int(row[1])) for row in zip(iext[select_idx], rtpr[select_idx])]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, labelpad=25, fontsize=fs - 2)
    fig.text(0, 0.5, "Normalized output", ha="center", va="center", rotation=90, fontsize=fs)
    fig.text(0.6, 0, "Time (s)", ha="center", va="bottom", fontsize=fs)
    fig.tight_layout()
    name = 'test' if is_test else 'val'
    tag = 'post' if is_post else 'prior'
    if is_test:
        np.save(file="results_{}/observations".format(config.model), arr=observations)
        np.save(file="results_{}/iext".format(config.model), arr=iext)
        np.save(file="results_{}/rtpr".format(config.model), arr=rtpr)
        np.save(file="results_{}/times".format(config.model), arr=times)
        np.save(file="results_{}/mu_50_{}".format(config.model, tag), arr=mu_50)
        np.save(file="results_{}/mu_75_{}".format(config.model, tag), arr=mu_75)
        np.save(file="results_{}/mu_25_{}".format(config.model, tag), arr=mu_25)
        np.save(file="results_{}/solution_xt_{}".format(config.model, tag), arr=solution_xt)
        np.save(file="results_{}/z_{}".format(config.model, tag), arr=z)

    plt.savefig("results_{}/{}_{}_{}".format(config.model, name, epoch, tag))
    return


def plot_by_label_challenge(epoch, symptoms, results, observations, shedding, select_idx, times, is_post, is_test,
                            solution_xt, z, config):
    mu_50 = results.mu_50.detach().cpu().numpy()
    mu_75 = results.mu_75.detach().cpu().numpy()
    mu_25 = results.mu_25.detach().cpu().numpy()

    columns = np.arange(observations.shape[1])
    rows = np.arange(len(select_idx))

    colors = ['tab:gray', 'r', 'y', 'c']
    fs = 14
    nplots = 10
    # rows = rows[0:nplots]
    # idx = idx[0:nplots]
    plt.clf()
    fig, axs = plt.subplots(len(rows), len(columns), sharex=True, sharey=True, figsize=(12, 20))
    mesh = np.array(np.meshgrid(rows, columns))
    combinations = mesh.T.reshape(-1, 2)
    for r, c in combinations:
        loc = select_idx[r]
        empirical = observations[loc, c, :]

        axs[r, c].plot(times, empirical, 'k.', markersize=2)
        axs[r, c].plot(times, mu_50[loc, c, :], '-', lw=2, alpha=0.75, color=colors[c])
        axs[r, c].plot(times, mu_75[loc, c, :], '-.', lw=2, alpha=0.75, color=colors[c])
        axs[r, c].plot(times, mu_25[loc, c, :], '-.', lw=2, alpha=0.75, color=colors[c])

        axs[r, c].set_xlim(0.0, max(times) + 0.01)
        axs[r, c].set_ylim(-0.01, 1.01)
        axs[r, c].set_xticks(np.arange(start=0, stop=len(times), step=50))
        axs[r, c].tick_params(axis='both', which='major', labelsize=fs)
    cols = ['HR', 'TEMP', 'EDA', 'ACC']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    rows = ["SySh={},{}".format(int(row[0]), int(row[1])) for row in zip(symptoms[select_idx], shedding[select_idx])]
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, labelpad=25, fontsize=fs - 2)
    fig.text(0, 0.5, "Normalized output", ha="center", va="center", rotation=90, fontsize=fs)
    fig.text(0.6, 0, "Time (hrs)", ha="center", va="bottom", fontsize=fs)
    fig.tight_layout()
    name = 'test' if is_test else 'val'
    tag = 'post' if is_post else 'prior'
    if is_test:
        np.save(file="results_{}/observations".format(config.model), arr=observations)
        np.save(file="results_{}/symptoms".format(config.model), arr=symptoms)
        np.save(file="results_{}/shedding".format(config.model), arr=shedding)
        np.save(file="results_{}/times".format(config.model), arr=times)
        np.save(file="results_{}/mu_50_{}".format(config.model, tag), arr=mu_50)
        np.save(file="results_{}/mu_75_{}".format(config.model, tag), arr=mu_75)
        np.save(file="results_{}/mu_25_{}".format(config.model, tag), arr=mu_25)
        np.save(file="results_{}/solution_xt_{}".format(config.model, tag), arr=solution_xt)
        np.save(file="results_{}/z_{}".format(config.model, tag), arr=z)

    plt.savefig("results_{}/{}_{}_{}".format(config.model, name, epoch, tag))
    return


def gen_treatment_str(conditions, treatments, unit=None):
    vstr_list = []
    for k, v in zip(conditions, treatments):
        val = np.exp(v) - 1.0
        if (val > 0.0) & (val < 1.0):
            vstr = '%s = %1.1f' % (k, val)
        else:
            vstr = '%s = %1.0f' % (k, val)
        if unit is not None:
            vstr = '%s %s' % (vstr, unit)
        vstr_list.append(vstr)
    return '\n'.join(vstr_list)


def individual_proc(results, observations, treatments, devices, config, epoch, times, is_post, is_test, z, solution_xt):
    observations = observations.detach().cpu().numpy()
    treatments = treatments.detach().cpu().numpy()
    devices = devices.detach().cpu().numpy()

    solution_xt = solution_xt.detach().cpu().numpy()
    z = z.detach().cpu().numpy()
    mu_50 = results.mu_50.detach().cpu().numpy()
    mu_75 = results.mu_75.detach().cpu().numpy()
    mu_25 = results.mu_25.detach().cpu().numpy()

    name = 'test' if is_test else 'val'
    tag = 'post' if is_post else 'prior'

    if is_test:
        np.save(file="results_{}/observations".format(config.model), arr=observations)
        np.save(file="results_{}/treatments".format(config.model), arr=treatments)
        np.save(file="results_{}/devices".format(config.model), arr=devices)
        np.save(file="results_{}/times".format(config.model), arr=times)

        np.save(file="results_{}/mu_50_{}".format(config.model, tag), arr=mu_50)
        np.save(file="results_{}/mu_75_{}".format(config.model, tag), arr=mu_75)
        np.save(file="results_{}/mu_25_{}".format(config.model, tag), arr=mu_25)
        np.save(file="results_{}/solution_xt_{}".format(config.model, tag), arr=solution_xt)
        np.save(file="results_{}/z_{}".format(config.model, tag), arr=z)

    # print(maxs, nplots, observations.shape, treatments.shape)
    for device_id in np.unique(devices, axis=0):
        comp = np.equal(devices, device_id)
        sel_device = np.sum(comp, 1) == devices.shape[1]

        both_locs = []
        for col in range(2):  # treament and device_id
            sel_treat = treatments[:, col] > 0.0
            all_locs = np.where(sel_device & sel_treat)[0]
            indices = np.argsort(treatments[all_locs, col])
            both_locs.append(all_locs[indices])

        plot_by_device(both_locs=both_locs, config=config, device_id=device_id, epoch=epoch, observations=observations,
                       treatments=treatments, times=times, name=name, tag=tag, mu_25=mu_25, mu_50=mu_50, mu_75=mu_75)
    return


def plot_by_device(both_locs, config, times, device_id, epoch, observations,
                   treatments, name, tag, mu_25, mu_50, mu_75):
    nplots = observations.shape[1]
    colors = ['tab:gray', 'r', 'y', 'c']
    fs = 14
    maxs = np.max(observations, axis=(0, 2))
    ntreatments = max(map(len, both_locs))
    # print(ntreatments)

    f = plt.figure(figsize=(12, 1.5 * ntreatments))
    for col, locs in enumerate(both_locs):  # C6 vs C12
        left = 0.1 + col * 0.5
        bottom = 0.4 / ntreatments
        width = 0.33 / nplots
        dx = 0.38 / nplots
        dy = (1 - bottom) / ntreatments
        height = 0.8 * dy
        for i, loc in enumerate(locs[:ntreatments]):
            treatment_str = gen_treatment_str(config.data.conditions, treatments[loc], unit='nM')

            for idx, maxi in enumerate(maxs):  # signals
                ax = f.add_subplot(ntreatments, 2 * nplots,
                                   col * nplots + (ntreatments - i - 1) * 2 * nplots + idx + 1)
                ax.set_position([left + idx * dx, bottom + (ntreatments - i - 1) * dy, width, height])

                ax.plot(times, observations[loc, idx, :] / maxi, 'k.', markersize=2)
                ax.plot(times, mu_50[loc, idx, :] / maxi, '-', lw=2, alpha=0.75, color=colors[idx])
                ax.plot(times, mu_75[loc, idx, :] / maxi, '-.', lw=2, alpha=0.75, color=colors[idx])
                ax.plot(times, mu_25[loc, idx, :] / maxi, '-.', lw=2, alpha=0.75, color=colors[idx])
                ax.set_xlim(0.0, 17)
                ax.set_xticks([0, 5, 10, 15])
                ax.set_ylim(-0.2, 1.2)
                ax.tick_params(axis='both', which='major', labelsize=fs)

                if i == 0:
                    plt.title(config.data.signals[idx], fontsize=fs)
                if i < ntreatments - 1:
                    ax.set_xticklabels([])
                if idx == 0:
                    ax.set_ylabel(treatment_str, labelpad=25, fontsize=fs - 2)
                else:
                    ax.set_yticklabels([])

                sns.despine()

        # Add labels
        f.text(left - 0.35 * dx, 0.5, "Normalized output", ha="center", va="center", rotation=90, fontsize=fs)
        f.text(left + 2 * dx, 0, "Time (h)", ha="center", va="bottom", fontsize=fs)

    id_1 = np.argmax(device_id[0:3])
    id_2 = np.argmax(device_id[3:])

    plt.savefig("results_{}/{}_{}_id_{}_{}_{}".format(config.model, name, epoch, id_1, id_2, tag))
    return


def visualize_latent(z_prior, z_post, config, epoch):
    fig = plt.figure(figsize=(5, 4))
    z_prior = z_prior.detach().cpu().numpy()
    z_post = z_post.detach().cpu().numpy()
    z_post_prior = np.concatenate((z_post, z_prior), axis=0)
    perp = 10
    tsne = TSNE(random_state=config.seed, perplexity=perp, verbose=1, n_components=2, init='pca', n_iter=5000)
    tsne_z_post_prior = tsne.fit_transform(z_post_prior)
    model_colors = sns.color_palette("husl", 2)

    size_z = len(z_post)
    plt.scatter(tsne_z_post_prior[0:size_z, 0], tsne_z_post_prior[0:size_z, 1],
                edgecolors=model_colors[0], cmap='prism', c="w", marker="o", label='Z_post')
    plt.scatter(tsne_z_post_prior[size_z:, 0], tsne_z_post_prior[size_z:, 1],
                edgecolors=model_colors[1], cmap='prism', c="w", marker="o", label='Z_prior')
    plt.tight_layout()
    plt.legend(fontsize=legend_size)
    fig.savefig("results_{}/z_TSNE_{}".format(config.model, epoch))
