import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import DataFrame
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from config.DQNConfig import DQNConfig
from config.GlobalConfig import GlobalConfig
from config.network import Network


def get_visual(hetnet):
    # Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='MBS', markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='SBS', markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='UE', markerfacecolor='r', markersize=10)]

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.legend(handles=legend_elements, loc='best')

    plt.xlim(-1005, 1100)
    plt.ylim(-1005, 1005)
    plt.grid(linestyle='-', linewidth=1, zorder=0, color='#E5E5E5')

    for linha_ue in hetnet.network_element:
        ne_element = [ne_element for ne_element in linha_ue if ne_element.coverage_status is True]
        for ne in ne_element:
            p_ue = [ne.ue.point.x, ne.bs.point.x]
            p_bs = [ne.ue.point.y, ne.bs.point.y]
            plt.plot(p_ue, p_bs, color="black", linewidth=0.5, zorder=5)

    for ue in hetnet.ue_list:
        p = (ue.point.x, ue.point.y)
        ue_circle = plt.Circle(p, 8.5, color="red", zorder=10)
        ax.add_patch(ue_circle)
        if ue.evaluation is False:
            n_ue_circle = plt.Circle(p, 20.5, color="red", zorder=10, fill=False)
            ax.add_patch(n_ue_circle)

    for bs in hetnet.list_bs:
        p = (bs.point.x, bs.point.y)
        if bs.type == 'MBS':
            ue_circle = plt.Circle(p, 13.5, color="blue", zorder=10)
        else:
            ue_circle = plt.Circle(p, 13.5, color="green", zorder=10)
        ax.add_patch(ue_circle)

    plt.show()


def get_evaluation_evolution(data, filename, marker='', xlim=None, ylim=None, alpha=False):
    plt.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'

    for key in data:
        if alpha:
            plt.plot(data[key][0:100], marker, label=key, markersize=2.2)
        else:
            plt.plot(data[key][0:300], marker, label=key, markersize=2.2)

    plt.xlabel('Iterations')
    plt.ylabel('Objective Function')
    plt.grid(linestyle=':')

    if alpha:
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper left')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/iom/images/", filename), dpi=Network.DEFAULT.image_resolution)
    plt.close()


def get_visual_cluster(clustering, data):
    labels = clustering.labels_
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def get_visual_pareto(data):
    all_points = []
    for d_ in data:
        if d_.evaluation_f1 != 0 and d_.evaluation_f2 != 0:
            point = [(-1) * d_.evaluation_f1, d_.evaluation_f2]
            all_points.append(point)

    df = DataFrame(all_points, columns=['x', 'y'])

    # Make the plot with this subset
    plt.plot('x', 'y', data=df, linestyle='', marker='o')

    # titles
    plt.xlabel('Value of X')
    plt.ylabel('Value of Y')
    plt.title('Overplotting? Sample your data', loc='left')
    plt.show()


def get_violinchart(data, key):
    # sns.violinplot(x=data["Algorithm"], y=data["Mean Number of Clusters"], scale_hue=False)
    # sns.swarmplot(x="Algorithm", y="Mean Number of Clusters", data=data)
    # plt.grid(linestyle=':', zorder=1)
    # plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/dcm/images/", "dcm_violin_number_of_clusters_{}.eps".format(key)), dpi=Network.DEFAULT.image_resolution, bbox_inches='tight')

    # plt.figure()
    # sns.violinplot(x=data["Algorithm"], y=data["Mean Number of Samples per Clusters"])
    sns.swarmplot(x="Algorithm", y="Mean Number of Samples per Clusters", data=data)
    plt.grid(linestyle=':', zorder=1)
    plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/dcm/images/", "dcm_violin_number_of_samples_{}.eps".format(key)), dpi=Network.DEFAULT.image_resolution, bbox_inches='tight')


def get_scatterplot(data, key):
    # sns.lmplot(x='index', y="Mean Number of Clusters", data=data.reset_index(), fit_reg=False, hue='Algorithm', legend=False, markers=["o", "x", "1"], palette=dict(DBSCAN='#1f77b4', KMeans='#e1802c', Birch='#3a913a'))
    sns.lmplot(x='index', y="Mean Number of Clusters", data=data.reset_index(), fit_reg=False, hue='Algorithm', legend=False, markers=["o", "x"], palette=dict(KMeans='#e1802c', Birch='#3a913a'))
    plt.grid(linestyle=':', zorder=1)
    plt.legend(loc='upper right')
    plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/dcm/images/", "dcm_scatter_number_of_clusters_{}.eps".format(key)), dpi=Network.DEFAULT.image_resolution, bbox_inches='tight')

    plt.figure()
    # sns.lmplot(x='index', y="Mean Number of Samples per Clusters", data=data.reset_index(), fit_reg=False, hue='Algorithm', legend=False, markers=["o", "x", "1"], palette=dict(DBSCAN='#1f77b4', KMeans='#e1802c', Birch='#3a913a'))
    sns.lmplot(x='index', y="Mean Number of Samples per Clusters", data=data.reset_index(), fit_reg=False, hue='Algorithm', legend=False, markers=["o", "x"], palette=dict(KMeans='#e1802c', Birch='#3a913a'))
    plt.grid(linestyle=':', zorder=1)
    plt.legend(loc='upper right')
    plt.savefig(
        '{}{}'.format("/Users/hugo/Desktop/PyCRE/dcm/images/", "dcm_scatter_number_of_samples_{}.eps".format(key)),
        dpi=Network.DEFAULT.image_resolution, bbox_inches='tight')


def get_barchart(cluster_data, path, z=1.96):
    # Bar position
    bar_position_left = [0, 3.0, 6.0]
    bar_position_center = [0.6, 3.6, 6.6]
    bar_position_right = [1.2, 4.2, 7.2]

    for key in cluster_data:
        data = cluster_data[key]

        clusters_bar = []
        samples_bar = []
        outliers_bar = []

        error_cluster = []
        error_samples = []
        error_outliers = []

        for d_ in data:
            # Get Mean and Standard Deviation
            mean = d_.satisfaction_growth(axis=0)
            std = d_.std(axis=0)

            # Append data
            clusters_bar.append(mean['Number of Clusters'])
            samples_bar.append(mean['Number of Samples per Clusters'])
            outliers_bar.append(mean['Number of Outliers'])

            # Compute SE and CI
            # Number of Clusters
            se_clusters = std['Number of Clusters'] / np.sqrt(len(d_))
            lcb_clusters = mean['Number of Clusters'] - z * se_clusters
            ucb_clusters = mean['Number of Clusters'] + z * se_clusters
            error_cluster.append([lcb_clusters, ucb_clusters])

            # Number of Samples per Clusters
            se_samples = std['Number of Samples per Clusters'] / np.sqrt(len(d_))
            lcb_samples = mean['Number of Samples per Clusters'] - z * se_samples
            ucb_samples = mean['Number of Samples per Clusters'] + z * se_samples
            error_samples.append([lcb_samples, ucb_samples])

            # Number of Outliers
            se_outliers = std['Number of Outliers'] / np.sqrt(len(d_))
            lcb_outliers = mean['Number of Outliers'] - z * se_outliers
            ucb_outliers = mean['Number of Outliers'] + z * se_outliers
            error_outliers.append([lcb_outliers, ucb_outliers])

        error_cluster = np.array(error_cluster).T.tolist()
        error_samples = np.array(error_samples).T.tolist()
        error_outliers = np.array(error_outliers).T.tolist()

        if key is 'DBSCAN':
            position = bar_position_left
            color = '#1f77b4'
        elif key is 'KMeans':
            position = bar_position_center
            color = '#e1802c'
        else:
            position = bar_position_right
            color = '#3a913a'

        # Clusters
        plt.figure(1)
        plt.bar(position, clusters_bar, yerr=error_cluster, capsize=7, width=0.5, zorder=10, label=key, color=color)

        plt.figure(2)
        plt.bar(position, samples_bar, yerr=error_samples, capsize=7, width=0.5, zorder=10, label=key, color=color)

        plt.figure(3)
        plt.bar(position, outliers_bar, yerr=error_outliers, capsize=7, width=0.5, zorder=10, label=key, color=color)

    for id_ in range(1, 4):
        if id_ is 1:
            ylabel = 'Mean Number of Clusters'
            filename = 'mean_number_of_clusters'
        elif id_ is 2:
            ylabel = 'Mean Number of Samples per Clusters'
            filename = 'mean_number_of_samples_per_clusters'
        else:
            ylabel = 'Mean Number of Outliers'
            filename = 'mean_number_of_outliers'

        plt.figure(id_)
        plt.xticks(bar_position_center, ('300', '600', '900'))
        plt.xlabel('Mean User Density [UE/km2]')
        plt.ylabel(ylabel)
        plt.grid(linestyle=':', zorder=1)
        plt.legend(loc='best')
        plt.savefig('{}{}'.format(path, "{}.eps".format(filename)), dpi=Network.DEFAULT.image_resolution,
                    bbox_inches='tight')


def get_bar_chart(mean_n_bs, mean_load_per_weight, path, filename):
    bar_width = 0.4

    m_n_bs = [mean_n_bs[0], mean_n_bs[3], mean_n_bs[8]]
    m_l_bs = [mean_load_per_weight[0], mean_load_per_weight[3], mean_load_per_weight[8]]

    r1 = np.arange(len(m_n_bs))
    r2 = [x + bar_width + 0.02 for x in r1]
    r3 = (r1 + r2) / 2

    plt.bar(r1, m_n_bs, width=bar_width, label="Percentage of Slices by Number of BSs", zorder=10)
    plt.bar(r2, m_l_bs, width=bar_width, label="Slice Load Average", zorder=10)

    plt.xlabel('Pareto Weight')
    plt.ylabel('%')
    plt.xticks(r3, ['0.9', '0.5', '0.1'])

    # Create legend & Show graphic
    plt.grid(linestyle=':', zorder=1)
    plt.legend()
    plt.savefig('{}{}'.format(path, "{}.eps".format(filename)), dpi=Network.DEFAULT.image_resolution,
                bbox_inches='tight')


def get_pareto_frontier(gwo_evaluation_f1, gwo_evaluation_f2, mogwo_evaluation, path, filename):
    plt.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'

    plt.plot(gwo_evaluation_f1, gwo_evaluation_f2, zorder=10, label="GWO")
    plt.plot('x', 'y', data=mogwo_evaluation, linestyle='', marker='o', color="red", zorder=10, label="MOGWO")

    plt.grid(linestyle=':', zorder=1)
    plt.xlabel("Objective Function $f_1$")
    plt.ylabel("Objective Function $f_2$")
    plt.legend()
    plt.savefig('{}{}'.format(path, "{}.eps".format(filename)), dpi=Network.DEFAULT.image_resolution)


def get_mean_evaluation_cluster(data, filename, marker=''):
    for key in data:
        plt.plot(data[key][0:100], marker, label=key, markersize=2.2)

    plt.hlines(75.0, -5.0, 105, linestyles='dashed', colors="r", label="Threshold")

    plt.xlabel('Episodes')
    plt.ylabel('Mean Satisfaction per Cluster (%)')
    plt.xlim(-5.0, 105.0)
    plt.ylim(40.0, 102.0)
    plt.grid(linestyle=':')

    plt.legend(loc='upper left')

    plt.savefig('{}{}'.format("/Users/hugo/Desktop/PyCRE/rlm/images/", filename), dpi=Network.DEFAULT.image_resolution)
    plt.close()


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def get_net_arch_episode_lengths_curve(data, path, image_id):
    for key in data:
        plt.plot(data[key], label=key)

    plt.xlabel('Episodes')
    plt.ylabel('Smoothing Training Steps')
    plt.xlim(0, 1005)
    plt.grid(linestyle=':')
    plt.legend(loc='best')
    plt.legend()
    plt.savefig(os.path.join(path, "{}_net_arch_episode_lengths.eps".format(image_id)), dpi=GlobalConfig.DEFAULT.image_resolution, bbox_inches='tight')


def get_net_arch_episode_rewards_curve(data, path, image_id):
    for key in data:
        plt.plot(data[key], label=key)

    plt.xlabel('Episodes')
    plt.ylabel('Smoothing of Obtained Rewards')
    plt.xlim(0, 1005)
    plt.grid(linestyle=':')
    plt.legend(loc='best')
    plt.legend()
    plt.savefig(os.path.join(path, "{}_net_arch_episode_rewards.eps".format(image_id)), dpi=GlobalConfig.DEFAULT.image_resolution, bbox_inches='tight')


def get_learning_rate_episode_lengths_curve(data, path, image_id):
    plt.rc('text', usetex=True)
    plt.rcParams["savefig.pad_inches"] = 0.05
    for key in data:
        plt.plot(data[key], label="$\delta={}$".format(key))

    plt.xlabel('Episodes')
    plt.ylabel('Smoothing Training Steps')
    plt.xlim(0, 1005)
    plt.grid(linestyle=':')
    plt.legend(loc='best')
    plt.legend()
    plt.savefig(os.path.join(path, "{}_learning_rate_episode_lengths.eps".format(image_id)), dpi=GlobalConfig.DEFAULT.image_resolution, bbox_inches='tight')


def get_learning_rate_episode_rewards_curve(data, path, image_id):
    plt.rc('text', usetex=True)
    plt.rcParams["savefig.pad_inches"] = 0.05
    for key in data:
        plt.plot(data[key], label="$\delta={}$".format(key))

    plt.xlabel('Episodes')
    plt.ylabel('Smoothing of Obtained Rewards')
    plt.xlim(0, 1005)
    plt.grid(linestyle=':')
    plt.legend(loc='best')
    plt.legend()
    plt.savefig(os.path.join(path, "{}_learning_rate_episode_rewards.eps".format(image_id)), dpi=GlobalConfig.DEFAULT.image_resolution, bbox_inches='tight')


def get_training_steps_curve(data, path, xlabel="Episodes", ylabel="Smoothing Training Steps", xlim=(0, 1005), pad_inches=None):
    plt.rc('text', usetex=True)
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    if pad_inches is not None:
        plt.rcParams["savefig.pad_inches"] = pad_inches

    for key in data:
        plt.plot(data[key], label=key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.grid(linestyle=':')
    plt.legend(loc='best')
    plt.legend()
    plt.savefig(path, dpi=GlobalConfig.DEFAULT.image_resolution, bbox_inches='tight')
