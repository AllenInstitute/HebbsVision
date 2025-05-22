import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy import stats
import seaborn as sns
import networkx as nx
import pickle
import itertools
from dotmotif import Motif, GrandIsoExecutor
from scipy.stats import kruskal, f_oneway, levene, ranksums, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests

def plot_shared_vs_disjoint(shared_values, disjoint_values, title, y_lab, save_fig : bool, folder : str):
    # Plot the distributions for shared and disjoint groups
    _, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Violin plot
    sns.violinplot(data=[shared_values, disjoint_values],
                   inner='box',
                   linewidth=1.5,
                   palette=[(.4, .6, .8, .5), 'grey'],
                   ax=ax,
                   #bw_adjust=0.5,
                   cut=0.1)
    
    # Swarm plot overlay
    sns.swarmplot(data=[shared_values, disjoint_values],
                  color='white',
                  edgecolor='grey',
                  size=3,
                  ax=ax)
    
    # Calculate and add the median line
    medians = [np.median(np.array(shared_values)), np.median(np.array(disjoint_values))]
    for i, median in enumerate(medians):
        ax.plot([i - 0.2, i + 0.2], [median, median], color='orange', linestyle='-', linewidth=2, label="Median" if i == 0 else "")

    # Set x-axis labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Shared Assembly Membership", "Disjoint Assembly Membership"], size=12)
    ax.set_title(f"{title}", size=16)
    ax.set_ylabel(y_lab, size=12)

    # Save figure if requested
    if save_fig:
        fig_title = f"{folder}_Shared_vs_Disjoint_Boxplot.png"
        plt.savefig(fig_title, dpi=1200)


    plt.tight_layout()
    plt.show()

def chi_squared_analysis(data):
    """
    Perform an overall chi-squared test of independence on a contingency table, visualize observed and expected values,
    and if significant, perform pairwise chi-squared tests between groups with Benjamini-Hochberg correction.

    Parameters:
    data (pd.DataFrame): A DataFrame representing the contingency table.

    Returns:
    pd.DataFrame: Pairwise chi-squared test results with adjusted p-values if the overall test is significant.
    """
    # Overall chi-squared test
    chi2, p, dof, expected = stats.chi2_contingency(data)
    expected_df = pd.DataFrame(expected, index=data.index, columns=data.columns)

    # Plot observed and expected values side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Observed Values")
    sns.heatmap(expected_df, annot=True, fmt=".2f", cmap="Reds", ax=axes[1])
    axes[1].set_title("Expected Values")
    plt.tight_layout()
    plt.show()

    # Display overall chi-squared test results
    print(f"Chi-squared Statistic: {chi2:.2f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4f}")

def construct_contingency_table(data_dict, groups):
    # Generate lists for connected and not connected counts
    connected_counts = [sum(1 for _, val in data_dict[group].items() if val == 1) for group in groups]
    not_connected_counts = [sum(1 for _, val in data_dict[group].items() if val == 0) for group in groups]
    
    # Create the DataFrame
    return pd.DataFrame({
        'Connected': connected_counts,
        'Not Connected': not_connected_counts
    }, index=[group.capitalize() for group in groups])


def ranksum_signedrank_two_group_comparison(comparison_dict, aggregation_method="by connection", directionality=None, data_type="binary", 
                            paired=False, non_zero=False, save_fig=False):
    """
    Compares 'shared' and 'disjoint' groups based on connection type and data type.
    Uses a one-sided Wilcoxon rank-sum test and performs a Wilcoxon signed-rank test if paired=True.

    Parameters:
    - comparison_dict (dict): Dictionary with 'shared' and 'disjoint' data.
    - aggregation_method (str): Type of connection ('connection' for pairwise, 'cell' for inbound/outbound by cell).
    - directionality (str): Direction of connectivity for 'cell' type ('inbound' or 'outbound').
    - data_type (str): Data type ('binary' for connectivity, 'summed_psd' for nonzero PSD).
    - paired (bool): If True, performs an additional Wilcoxon signed-rank test on paired data.
    - non_zero (bool): If True, filters out zero entries for summed PSD.
    - save_fig (bool): If True, saves the figure as a PNG file.
    """

    # Set title and labels based on connection_type and data_type
    if aggregation_method == "connection":  # Pairwise connections
        if data_type == "binary":
            title = "Pairwise Binary Connectivity By Connection Type"
            y_lab = "Binary Connections"
            folder = "pairwise_binary_connectivity"
        elif data_type == "summed_psd":
            if non_zero == True:
                title = "Pairwise Nonzero Summed PSD By Connection Type"
                y_lab = "Nonzero Summed PSD (nm$^3$)"
                folder = "pairwise_nonzero_summed_psd"
            else:
                title = "Pairwise Summed PSD By Connection Type"
                y_lab = "Summed PSD (nm$^3$)"
                folder = "pairwise_summed_psd"
        else:
            raise ValueError("Invalid data_type for pairwise connection.")
    
    elif aggregation_method == "cell":  # By cell with inbound/outbound directionality
        if directionality not in ["inbound", "outbound"]:
            raise ValueError("For 'cell' connection_type, directionality must be 'inbound' or 'outbound'.")
        
        if data_type == "binary":
            title = f"Probability of {directionality.capitalize()} Connection by Connection Type"
            y_lab = f"Probability of {directionality.capitalize()} Connection"
            folder = f"{directionality}_connection_probability"
        elif data_type == "summed_psd":
            title = f"Average Nonzero {directionality.capitalize()} PSD by Connection Type"
            y_lab = f"Average Nonzero {directionality.capitalize()} PSD (nm$^3$)"
            folder = f"{directionality}_average_nonzero_psd"
        else:
            raise ValueError("Invalid data_type for inbound/outbound connection.")
    else:
        raise ValueError("Invalid connection_type. Must be 'connection' or 'cell'.")

    shared_values = np.array(list(comparison_dict["shared"].values()))
    disjoint_values = np.array(list(comparison_dict["disjoint"].values()))
    # Filter out zeros if non_zero is specified for summed_psd
    if non_zero and data_type == "summed_psd":
        shared_values = shared_values[shared_values != 0]
        disjoint_values = disjoint_values[disjoint_values != 0]

    # Perform the Wilcoxon rank-sum test (one-sided, shared > disjoint)
    rank_sum_stat, rank_sum_p = stats.ranksums(shared_values, disjoint_values, alternative='greater')

    print(f"Wilcoxon Rank-Sum Test (unpaired, shared > disjoint):\nStatistic: {rank_sum_stat:.6f}, P-value: {rank_sum_p:.6f}")
    title = f'{title}\nWilcoxon Rank-Sum P-value: {rank_sum_p:.4f}'

    # If paired=True, also perform a Wilcoxon signed-rank test on paired observations
    if paired:
        shared_keys = set(comparison_dict.get('shared', {}).keys())
        disjoint_keys = set(comparison_dict.get('disjoint', {}).keys())
        common_keys = shared_keys & disjoint_keys

        if common_keys:
            # Extract paired data for common keys
            shared_paired = np.array([comparison_dict["shared"][key] for key in common_keys])
            disjoint_paired = np.array([comparison_dict["disjoint"][key] for key in common_keys])

            # Perform Wilcoxon signed-rank test on paired data
            signed_rank_stat, signed_rank_p = stats.wilcoxon(shared_paired, disjoint_paired, alternative='greater')

            print(f"Wilcoxon Signed-Rank Test (paired, shared > disjoint):\nStatistic: {signed_rank_stat:.6f}, P-value: {signed_rank_p:.6f}")
            title = f'{title}, Wilcoxon Signed-Rank P-value: {signed_rank_p:.4f}'
        else:
            print("No common observations found for paired analysis.")

    plot_shared_vs_disjoint(shared_values, disjoint_values, title, y_lab, save_fig, folder)

def ttest_two_group_comparison(comparison_dict, aggregation_method="by connection", directionality=None, 
                               data_type="binary", paired=False, non_zero=False, save_fig=False):
    """
    Compares 'shared' and 'disjoint' groups based on connection type and data type.
    Uses a one-sided independent t-test and performs a paired t-test if paired=True, on log-transformed values.

    Parameters:
    - comparison_dict (dict): Dictionary with 'shared' and 'disjoint' data.
    - aggregation_method (str): Type of connection ('connection' for pairwise, 'cell' for inbound/outbound by cell).
    - directionality (str): Direction of connectivity for 'cell' type ('inbound' or 'outbound').
    - data_type (str): Data type ('binary' for connectivity, 'summed_psd' for nonzero PSD).
    - paired (bool): If True, performs an additional paired t-test on paired data.
    - non_zero (bool): If True, filters out zero entries for summed PSD.
    - save_fig (bool): If True, saves the figure as a PNG file.
    """

    # Set title and labels based on aggregation_method and data_type
    if aggregation_method == "connection":  # Pairwise connections
        if data_type == "binary":
            title = "Pairwise Binary Connectivity By Connection Type"
            y_lab = "Binary Connections"
            folder = "pairwise_binary_connectivity"
        elif data_type == "summed_psd":
            if non_zero:
                title = "Pairwise Nonzero Summed PSD By Connection Type"
                y_lab = "Nonzero Summed PSD (nm$^3$)"
                folder = "pairwise_nonzero_summed_psd"
            else:
                title = "Pairwise Summed PSD By Connection Type"
                y_lab = "Summed PSD (nm$^3$)"
                folder = "pairwise_summed_psd"
        else:
            raise ValueError("Invalid data_type for pairwise connection.")
    
    elif aggregation_method == "cell":  # By cell with inbound/outbound directionality
        if directionality not in ["inbound", "outbound"]:
            raise ValueError("For 'cell' aggregation method, directionality must be 'inbound' or 'outbound'.")
        
        if data_type == "binary":
            title = f"Probability of {directionality.capitalize()} Connection by Connection Type"
            y_lab = f"Probability of {directionality.capitalize()} Connection"
            folder = f"{directionality}_connection_probability"
        elif data_type == "summed_psd":
            title = f"Average Nonzero {directionality.capitalize()} PSD by Connection Type"
            y_lab = f"Average Nonzero {directionality.capitalize()} PSD (nm$^3$)"
            folder = f"{directionality}_average_nonzero_psd"
        else:
            raise ValueError("Invalid data_type for inbound/outbound connection.")
    else:
        raise ValueError("Invalid aggregation_method. Must be 'connection' or 'cell'.")

    shared_values = np.array(list(comparison_dict["shared"].values()))
    disjoint_values = np.array(list(comparison_dict["disjoint"].values()))

    # Filter out zeros if non_zero is specified for summed_psd and log-transform the values
    if non_zero and data_type == "summed_psd":
        shared_values = shared_values[shared_values != 0]
        disjoint_values = disjoint_values[disjoint_values != 0]

    # Apply log transformation
    shared_values = np.log(shared_values)
    disjoint_values = np.log(disjoint_values)

    # Perform the independent t-test (one-sided, shared > disjoint)
    t_stat, t_p_value = stats.ttest_ind(shared_values, disjoint_values, alternative='greater', equal_var=False)

    print(f"Independent t-test (unpaired, shared > disjoint):\nStatistic: {t_stat:.6f}, P-value: {t_p_value:.6f}")
    title = f'{title}\n Independent t-test P-value: {t_p_value:.4f}'


    # If paired=True, also perform a paired t-test on paired observations
    if paired:
        shared_keys = set(comparison_dict.get('shared', {}).keys())
        disjoint_keys = set(comparison_dict.get('disjoint', {}).keys())
        common_keys = shared_keys & disjoint_keys

        if common_keys:
            # Extract paired data for common keys
            shared_paired = np.array([comparison_dict["shared"][key] for key in common_keys])
            disjoint_paired = np.array([comparison_dict["disjoint"][key] for key in common_keys])

            # Apply log transformation to paired data
            shared_paired = np.log(shared_paired)
            disjoint_paired = np.log(disjoint_paired)

            # Perform paired t-test
            t_stat_paired, t_p_value_paired = stats.ttest_rel(shared_paired, disjoint_paired, alternative='greater')

            print(f"Paired t-test (paired, shared > disjoint):\nStatistic: {t_stat_paired:.6f}, P-value: {t_p_value_paired:.6f}")
            title = f'{title}, Paired t-test P-value: {t_p_value_paired:.4f}'

        else:
            print("No common observations found for paired analysis.")

    plot_shared_vs_disjoint(shared_values, disjoint_values, title, y_lab, save_fig, folder)