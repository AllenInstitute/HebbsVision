# %% [markdown]
import sys
print(sys.executable)
print(sys.path)



# # Figure 4 Master Freeze Document
# 
# This is the master document for Figure 4, which includes all code that will be frozen before receiving the final set of co-registrated cells. All frozen code is tasked to directly test a hypothesis that was made in Hebb's "The Organization of Behavior".
# 
# The methodology for all written code is provided in the *Methods* section of our paper. While the code includes the ability to test between other sets, the main sets of comparison will be coregistered cells which have 'shared' assembly membership to those who have 'disjoint' membership. Futher clarification is found in the *Methods* section. 
# 
# To analyze for probability of connections and strength of connections, we have specified these tests:
# 
# 1. **Monosynaptic Pairs** - 
#     1. Chi-squared test to binary connectivity
#     2. Wilcoxon rank-sum test to summed Post Synaptic Density (PSD)
# 2. **Per-cell Outbound and Inbound**
#     1. Wilcoxon rank-sum test to probability of connection
# 3. **Per-cell Nonzero Outbound and Inbound**
#     1. Wilcoxon signed rank test to summed PSD volumes
#     2. Wilcoxon rank-sum to summed PSD volumes
# 4. **Centrality Measurements**
#     1. Wilcoxon rank-sum to Out-Degree Centrality
#     2. Wilcoxon rank-sum to In-Degree Centrality
#     3. Wilcoxon rank-sum to Betweenness Centrality
#     4. Wilcoxon rank-sum to Closeness Centrality
# 5. Repeat 1-3 for **Multisynaptic (3-Neuron) Chains**
# 6. Repeat 1-3 for **Multisynaptic Chains with a middle interneuron**
# 
# We additionally perform a **Tail Analysis**, where we perform a **Chi-Squared Test of Goodness-of-Fit** for differences in proportion of connection type comparing all to "tail" connections. 
# 
# Any other analysis that will be explored later are presented in the other Figure 5 Master document.

# %%
# importing packages
import matplotlib.pyplot as plt
import ptitprince as pt
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
from scipy.stats import kruskal, f_oneway, levene, ranksums, ttest_ind, wilcoxon, norm, chi2_contingency, chisquare
from statsmodels.stats.multitest import multipletests
from sklearn import mixture
from scipy.interpolate import interp1d
from tabulate import tabulate
from statannotations.Annotator import Annotator

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10,10)
sns.set_theme(style="whitegrid")
random.seed(747)

# Import Stefan's Library for Data Management of V1DD
from lsmm_data import LSMMData
# import lsmm_data.LSMMData

run_descriptors = []
scans_to_merge = ['1_2_4_742', '1_3_4_742', '1_4_4_742']
for scan_session_affinity_filestring in scans_to_merge:
    with open(f'./FigureCode/Figure4/pyr_cells_rectangular_connectome_{scan_session_affinity_filestring}.json') as f:
        pyr_cells_rect_lsmm_json_input = json.load(f)
        run_descriptors.append(pyr_cells_rect_lsmm_json_input['run_descriptor'])
    with open(f'./FigureCode/Figure4/pyr_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        pyr_cells_rect_lsmm_json_input = json.load(f)
        run_descriptors.append(pyr_cells_rect_lsmm_json_input['run_descriptor'])
    with open(f'./FigureCode/Figure4/all_cells_rectangular_connectome_{scan_session_affinity_filestring}.json') as f:
        pyr_cells_rect_lsmm_json_input = json.load(f)
        run_descriptors.append(pyr_cells_rect_lsmm_json_input['run_descriptor'])
    with open(f'./FigureCode/Figure4/all_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        pyr_cells_rect_lsmm_json_input = json.load(f)
        run_descriptors.append(pyr_cells_rect_lsmm_json_input['run_descriptor'])
    

# Change to 2 for final run in merged
# for scan_session_affinity_filestring in ['1_2_4_742', '1_3_4_742', '1_4_4_742']:
    # scan_session_affinity_filestring = '1_2_4_742'  # Edit this for different versions
    # scan_session_affinity_filestring = '1_3_4_742'  # Edit this for different versions
    # scan_session_affinity_filestring = '1_3_4_974'  # Edit this for different versions
    # scan_session_affinity_filestring = '1_3_4_1196'  # Edit this for different versions
    # scan_session_affinity_filestring = '1_4_4_742'  # Edit this for different versions

    # %%
    # Set-Wise Comparison Functions: Determining the intersection of assembly assignment of two pyramidal cells 
    # These comparison functions map to C in the statistical methods section.
def shared(pre, post, A):
    try:
        return not A[pre].isdisjoint(A[post])
    except KeyError:
        return False

def disjoint(pre, post, A):
    try:
        return A[pre].isdisjoint(A[post])
    except KeyError:
        return False

def shared_no_a(pre, post, A):
    return (pre in no_A) and (post in no_A) # type: ignore

def no_a_a(pre, post, A):
    return (pre in no_A) and (post not in no_A) # type: ignore

def a_no_a(pre, post, A):
    return (pre not in no_A) and (post in no_A) # type: ignore

def no_a_to_any(pre, _, A):
    return (pre in no_A) # type: ignore

def a_to_any(pre, _, A):
    return (pre not in no_A) # type: ignore

comparison_functions = [shared, disjoint] #, shared_no_a, no_a_a, a_no_a, no_a_to_any, a_to_any]
groups = ['shared', 'disjoint']

# %%
merge_string = ""
for filestring in scans_to_merge:
    merge_string += filestring + "_"

# %% [markdown]
# ## Monosynaptic Analysis on Pyramidal Cell Rectangular Connectome

# %%
def save_figure(figure_name):
    plt.savefig(
        f"./draft_figures/{figure_name}_merged_{merge_string}.png",
        dpi=500,
        bbox_inches="tight")

def save_values(values_name, first_values, second_values):
    with open(f"./values/{values_name}_merged_{merge_string}.pkl", "wb") as f:
        pickle.dump((first_values, second_values), f)
    
def plot_shared_vs_disjoint(shared_values, disjoint_values, title, y_lab, p_val, save=False, figure_name=None):
    """
    Plots a raincloud plot for two connection type groups, with sample sizes in the y-axis labels.

    Parameters:
        shared_values (list or array): Data for shared assembly group.
        disjoint_values (list or array): Data for disjoint assembly group.
        title (str): Title of the plot.
        y_lab (str): Label for the x-axis.
        p_val (float): P-value for significance annotation.
        save_fig (bool): Whether to save the figure.
        folder (str): Folder to save the figure if save_fig is True.
    """
    # Calculate sample sizes
    n_shared = len(shared_values)
    n_disjoint = len(disjoint_values)

    y_labels = [f"Shared\n(n={n_shared})", f"Disjoint\n(n={n_disjoint})"]

    # Data frame for easier plotting
    data = pd.DataFrame({
        "Values": np.concatenate([shared_values, disjoint_values]),
        "Group": [y_labels[0]] * n_shared + [y_labels[1]] * n_disjoint
    })

    # Set up the plot
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    # Create the raincloud plot
    ax = pt.RainCloud(
        y="Values",
        x="Group",
        data=data,  
        palette=[(.4, .6, .8, .5), 'grey'],
        width_viol=0.3,  
        alpha=0.8,  
        move=0.25,
        point_size = 6,  
        orient="v" 
    )

    # Set markings for significance
    pairs = [(y_labels[0], y_labels[1])]
    annot = Annotator(ax, 
                    pairs,
                    data=data,
                    x="Group",
                    y="Values",
                    order=y_labels # Force the order
                    )
    annot.set_pvalues([p_val])
    annot.configure(text_format="star", loc="inside", fontsize=30)
    annot.annotate()

    # Add plot title and labels
    plt.title(title, size=30)
    plt.xlabel("Connection Type", size=26)
    plt.ylabel(y_lab, size=26)
    plt.xticks(fontsize = 26)
    plt.yticks(fontsize = 26)

    if save == True:
        save_figure(figure_name)
        save_values(figure_name, shared_values, disjoint_values)
    
    plt.tight_layout()
    #plt.show()
    plt.close()

def plot_shared_vs_disjoint_with_side_plot(shared_values, disjoint_values, title, 
                                        y_lab, p_val, for_chains = True,
                                        save=False, figure_name=None
):
    """
    Plots a raincloud plot comparing connection types, 
    plus a smaller side subplot summarizing mean ± SEM for each group.

    Parameters:
        shared_values (list or array): Data for shared assembly group.
        disjoint_values (list or array): Data for disjoint assembly group.
        title (str): Title of the plot.
        y_lab (str): Label for the x-axis.
        p_val (float): P-value for significance annotation.
        save_fig (bool): Whether to save the figure.
        folder (str): Folder to save the figure if save_fig is True.
    """

    # Calculate sample sizes
    n_shared = len(shared_values)
    n_disjoint = len(disjoint_values)

    y_labels = [f"Shared\n(n={n_shared})", f"Disjoint\n(n={n_disjoint})"]

    # Build a frame for easier plotting
    data = pd.DataFrame({
        "Values": np.concatenate([shared_values, disjoint_values]),
        "Group": [y_labels[0]] * n_shared + [y_labels[1]] * n_disjoint
    })

    # Compute the statistics for the side plot
    # (Assuming values > 0 for simplicity; modify if needed.)
    shared_log = np.log10(shared_values)
    disjoint_log = np.log10(disjoint_values)

    mean_shared_log = np.mean(shared_log)
    mean_disjoint_log = np.mean(disjoint_log)
    sem_shared_log = stats.sem(shared_log, ddof=1) if n_shared > 1 else 0
    sem_disjoint_log = stats.sem(disjoint_log, ddof=1) if n_disjoint > 1 else 0

    # Set up a figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    # Allocate 2 columns with a narrower column on the right
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 1], wspace=0.3)
    
    # Set up styling
    ax_main = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])
    sns.set_theme(style="whitegrid")

    # --- Main plot (original RainCloud) ---
    pt.RainCloud(
        y="Values",
        x="Group",
        data=data,
        palette=[(.4, .6, .8, .5), 'grey'],
        width_viol=0.3,
        alpha=0.8,
        move=0.25,
        point_size=6,
        orient="v",
        ax=ax_main
    )

    # Annotate significance
    pairs = [(y_labels[0], y_labels[1])]
    annot = Annotator(ax_main, 
                    pairs,
                    data=data,
                    x="Group",
                    y="Values",
                    order=y_labels # Force the order
                    )
    annot.set_pvalues([p_val])
    annot.configure(text_format="star", loc="inside", fontsize=28)
    annot.annotate()

    # Axis title and labels
    ax_main.set_title(title, size=24)
    ax_main.set_xlabel("Connection Type", size=24)
    ax_main.set_ylabel(y_lab, size=24)
    ax_main.tick_params(labelsize=24)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax_main.yaxis.get_offset_text().set_fontsize(24)


    # --- Side plot (Mean ± SEM of log(data))---
    # Currently place two horizontal lines and use fill_between for each ± sem region.

    x_vals = [1, 2]  # x positions for shared and disjoint
    mean_logs = [mean_shared_log, mean_disjoint_log]
    sem_logs = [sem_shared_log, sem_disjoint_log]
    colors = [(0.4, 0.6, 0.8, 0.8), 'grey']

    for i, x in enumerate(x_vals):
        m_log = mean_logs[i]
        s_log = sem_logs[i]
        c = colors[i]

        # Horizontal line for mean
        ax_side.hlines(
            y = m_log, 
            xmin = x - 0.15, 
            xmax = x + 0.15, 
            color = c, 
            linewidth = 3
        )
        # Shaded area for ± SEM
        ax_side.fill_betweenx(
            y = [m_log - s_log, m_log + s_log],
            x1 = x - 0.15,
            x2 = x + 0.15,
            color = c,
            alpha = 0.4
        )

    # Tidy up side axis
    ax_side.set_title("Mean ± SEM", size=24)
    ax_side.set_xlim(0.5, 2.5)  
    ax_side.set_xticks(x_vals)
    ax_side.set_xticklabels(["Shared", "Disjoint"], fontsize=24)
    ax_side.tick_params(axis='y', labelsize=24)
    # Show that y-values are on a log base 10 scale
    if for_chains:
        ax_side.set_ylabel(r"$\log_{10}$(Synaptic Weight Products) $(\mathrm{\mu m^6})$", size=24)
    else:
        ax_side.set_ylabel(r"$\log_{10}$(Synaptic Weight) $(\mathrm{\mu m^3})$", size=24)

    if save and figure_name is not None:
        save_figure(figure_name)
        save_values(figure_name, shared_values, disjoint_values)

    plt.tight_layout()
    #plt.show()
    plt.close()

def chi_squared_analysis(data, save=False, figure_name=None):
    """
    Perform an overall chi-squared test of independence on a contingency table and display
    observed and expected values as pretty tables with test results.

    Parameters:
    data (pd.DataFrame): A DataFrame representing the contingency table.

    Returns:
    None: Prints the tables and results directly.
    """
    # Perform chi-squared test
    chi2, p, dof, expected = chi2_contingency(data)
    print(data)
    expected_df = pd.DataFrame(expected, index=data.index, columns=data.columns)

    # Create pretty tables
    observed_table = tabulate(
        [[row] + list(data.loc[row]) for row in data.index],
        headers=["Connection Type"] + list(data.columns),
        tablefmt="pretty"
    )
    expected_table = tabulate(
        [[row] + [f"{val:.2f}" for val in expected_df.loc[row]] for row in expected_df.index],
        headers=["Connection Type"] + list(expected_df.columns),
        tablefmt="pretty"
    )

    # Print the results
    print("Observed Contingency Table:")
    print(observed_table, "\n")
    print("Expected Contingency Table:")
    print(expected_table, "\n")
    print("Chi-squared Test Results:")
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4g}")

    # Plot the heatmap with updated annotation and tick font sizes
    plt.figure(figsize=(6, 3))
    sns.set_theme(style="whitegrid")

    # Create a custom uniform heatmap
    ax = sns.heatmap(
        data,
        annot=True,               # Add annotations for the counts
        fmt="d",                  # Integer format for annotations
        cmap=sns.color_palette(["lightgrey"], as_cmap=True),  # All cells the same light gray color
        cbar=False,               # Remove the color bar
        annot_kws={"fontsize": 22},  # Set font size for annotations
        linewidths=2,             # Add grid lines
        linecolor='black'         # Grid line color
    )

    # Add title and labels
    plt.title(f"Probability of Connection\nChi-squared P-value: {p:.2g}", size=24)
    plt.xlabel("Connection Status", size=22)
    plt.ylabel("Connection Type", size=22)
    plt.xticks(fontsize=22)
    ax.set_yticklabels(data.index, rotation=90, va='center', fontsize=22)


    if save==True:
        save_figure(figure_name)
        save_values(figure_name, data, None)

    plt.tight_layout()
    #plt.show()

def chi_squared_analysis_v2(data, save=False, figure_name=None):
    """
    Perform an overall chi-squared test of independence on a contingency table and display
    observed and expected values as pretty tables with test results. This version plots a 
    heatmap of the *cell-wise chi-square contributions* (rather than the raw counts), 
    to visualize which cells contribute most to the chi-square statistic.

    Parameters:
    data (pd.DataFrame): A DataFrame representing the contingency table constructed from the
                        `construct_contingency_table` function.
    save (bool): Whether to save the resulting plot.
    figure_name (str or None): The filename to use if saving the plot.

    Returns:
    None: Prints the tables and results directly.
    """
    # Perform chi-squared test
    chi2, p, dof, expected = chi2_contingency(data)
    expected_df = pd.DataFrame(expected, index=data.index, columns=data.columns)

    # Create pretty tables
    observed_table = tabulate(
        [[row] + list(data.loc[row]) for row in data.index],
        headers=["Connection Type"] + list(data.columns),
        tablefmt="pretty"
    )
    expected_table = tabulate(
        [[row] + [f"{val:.2f}" for val in expected_df.loc[row]] for row in expected_df.index],
        headers=["Connection Type"] + list(expected_df.columns),
        tablefmt="pretty"
    )

    # Print the results
    print("Observed Contingency Table:")
    print(observed_table, "\n")
    print("Expected Contingency Table:")
    print(expected_table, "\n")
    print("Chi-squared Test Results:")
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p:.4g}")

    # # Calculate the cell-wise contributions
    # contributions = (data - expected_df) ** 2 / expected_df
    # contributions = contributions.fillna(0)  # Replace NaN with 0 for cells with no expected count

    # Calculate directional cell-wise contributions using Pearson residuals
    residuals = (data - expected_df) / np.sqrt(expected_df)
    residuals = residuals.fillna(0)  # Replace NaN values if any expected count is zero

    # Plot the heatmap with updated annotation and tick font sizes
    plt.figure(figsize=(6, 3))
    sns.set_theme(style="whitegrid")

    # Create a custom uniform heatmap
    ax = sns.heatmap(
        residuals,
        annot=True,               # Add annotations for the counts
        fmt=".2f",                  # Integer format for annotations
        cmap=sns.color_palette(["lightgrey"], as_cmap=True),  # All cells the same light gray color
        cbar=False,               # Remove the color bar
        annot_kws={"fontsize": 24},  # Set font size for annotations
        linewidths=2,             # Add grid lines
        linecolor='black'         # Grid line color
    )

    # Add title and labels
    plt.title(f"Chi-Square Pearson Residuals\nP-value: {p:.2g}",size=24)
    plt.xlabel("Connection Status", size=24)
    plt.ylabel("Connection Type", size=24)
    plt.xticks(fontsize=22)
    ax.set_yticklabels(data.index, rotation=90, va='center', fontsize=22)


    if save==True:
        save_figure(figure_name)
        save_values(figure_name, data, None)

    plt.tight_layout()
    #plt.show()

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
                            paired=False, non_zero=False, chain_test = False, chain_description = "Excitatory", save=True, figure_name=None):
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
    - chain_test (bool): If True, the test is considering chains.
    - chain_description (str): Type of intermediate cell in chain ('excitatory' or 'inhibitory')
    """

    # Set title and labels based on connection_type and data_type
    if aggregation_method == "connection":  # Pairwise connections
        if data_type == "binary":
            title = "Binary Connectivity"
            y_lab = "Binary Connections"
            folder = "pairwise_binary_connectivity"
        elif data_type == "summed_psd":
            if non_zero == True:
                title = "Synaptic Weight"
                y_lab = "Nonzero Synaptic Weight (\u03bcm$^3$)"
                # title = "Nonzero Summed PSD"
                # y_lab = "Nonzero Summed PSD (\u03bcm$^3$)"
                folder = "pairwise_nonzero_summed_psd"
            else:
                title = "Synaptic Weight"
                y_lab = "Synaptic Weight (\u03bcm$^3$)"
                # title = "Summed PSD"
                # y_lab = "Summed PSD (\u03bcm$^3$)"
                folder = "pairwise_summed_psd"
        else:
            raise ValueError("Invalid data_type for pairwise connection.")

    elif aggregation_method == "cell":  # By cell with inbound/outbound directionality
        if directionality not in ["inbound", "outbound"]:
            raise ValueError("For 'cell' connection_type, directionality must be 'inbound' or 'outbound'.")
        
        if data_type == "binary":
            title = f"Probability of {directionality.capitalize()} Connection by Cell"
            y_lab = f"Probability of {directionality.capitalize()} Connection"
            folder = f"{directionality}_connection_probability"
        elif data_type == "summed_psd":
            title = f"Average Nonzero {directionality.capitalize()} PSD by Cell"
            y_lab = f"Average Nonzero {directionality.capitalize()} PSD (\u03bcm$^3$)"
            folder = f"{directionality}_average_nonzero_psd"
        else:
            raise ValueError("Invalid data_type for inbound/outbound connection.")
    else:
        raise ValueError("Invalid connection_type. Must be 'connection' or 'cell'.")

    if chain_test:
        if directionality == None:
            title = f"{chain_description} Chain Weight Proucts"
            y_lab = "Synaptic Weight Products"
        elif data_type == 'binary':
            title = f"Probability of {directionality.capitalize()} Chain Connection by Cell"
            y_lab = f"Probability of {directionality.capitalize()} Chain Connection"
        elif data_type == 'summed_psd':
            title = f"Average Nonzero {directionality.capitalize()} PSD Chain Product by Cell"
            y_lab = "Synaptic Weight Products"
        
    # Accept both dict-of-key->value (e.g., {(j,i):val}) and sequence/array-like (e.g., lists of values)
    def _to_array_like(x):
        # leave dicts intact for paired/keyed analysis but return array of values for tests
        if isinstance(x, dict):
            return np.array(list(x.values())), True  # True indicates original was dict-like
        elif isinstance(x, (np.ndarray, list, tuple, pd.Series)):
            return np.array(x), False
        else:
            try:
                return np.array(list(x)), False
            except Exception:
                raise TypeError("Unsupported data type for group values. Expect dict or sequence.")

    if 'shared' not in comparison_dict or 'disjoint' not in comparison_dict:
        raise KeyError("comparison_dict must contain 'shared' and 'disjoint' keys.")

    shared_vals_raw = comparison_dict['shared']
    disjoint_vals_raw = comparison_dict['disjoint']

    shared_values, shared_was_dict = _to_array_like(shared_vals_raw)
    disjoint_values, disjoint_was_dict = _to_array_like(disjoint_vals_raw)

    # Filter out zeros if non_zero is specified for summed_psd
    if non_zero and data_type == "summed_psd":
        shared_values = shared_values[shared_values != 0]
        disjoint_values = disjoint_values[disjoint_values != 0]

    # Perform the Wilcoxon rank-sum test (one-sided)
    if chain_description == 'Inhibitory':
        rank_sum_stat, rank_sum_p = stats.ranksums(shared_values, disjoint_values, alternative='less')
        print(f"Wilcoxon Rank-Sum Test (unpaired, shared < disjoint):\nStatistic: {rank_sum_stat:.4g}, P-value: {rank_sum_p:.4g}")
    else:
        rank_sum_stat, rank_sum_p = stats.ranksums(shared_values, disjoint_values, alternative='greater')
        print(f"Wilcoxon Rank-Sum Test (unpaired, shared > disjoint):\nStatistic: {rank_sum_stat:.4g}, P-value: {rank_sum_p:.4g}")

    title = f'{title}\nRank-Sum P-value: {rank_sum_p:.2g}'

    # If paired=True, attempt a Wilcoxon signed-rank test on paired observations
    if paired:
        # If both original inputs were dicts, use common keys for pairing
        if shared_was_dict and disjoint_was_dict:
            shared_keys = set(shared_vals_raw.keys())
            disjoint_keys = set(disjoint_vals_raw.keys())
            common_keys = shared_keys & disjoint_keys

            if common_keys:
                shared_paired = np.array([shared_vals_raw[key] for key in common_keys])
                disjoint_paired = np.array([disjoint_vals_raw[key] for key in common_keys])
            else:
                print("No common observations found for paired analysis.")
                shared_paired = disjoint_paired = None
        else:
            # For sequence inputs, require equal length vectors and assume positional pairing
            if len(shared_values) == len(disjoint_values):
                shared_paired = shared_values
                disjoint_paired = disjoint_values
            else:
                print("Paired analysis requested but inputs are not dict-like and lengths differ; skipping paired test.")
                shared_paired = disjoint_paired = None

        if shared_paired is not None:
            if chain_description == 'Inhibitory':
                signed_rank_stat, signed_rank_p = stats.wilcoxon(shared_paired, disjoint_paired, alternative='less')
                print(f"Wilcoxon Signed-Rank Test (paired, shared < disjoint):\nStatistic: {signed_rank_stat:.4g}, P-value: {signed_rank_p:.4g}")
            else:
                signed_rank_stat, signed_rank_p = stats.wilcoxon(shared_paired, disjoint_paired, alternative='greater')
                print(f"Wilcoxon Signed-Rank Test (paired, shared > disjoint):\nStatistic: {signed_rank_stat:.4g}, P-value: {signed_rank_p:.4g}")

            title = f'{title}, Signed-Rank P-value: {signed_rank_p:.2g}'

    if len(shared_values) == 0 or len(disjoint_values) == 0:
        print("Warning: shared_values or disjoint_values is empty. Skipping plot.")
    else:
        plot_shared_vs_disjoint(shared_values, disjoint_values, title, y_lab, p_val=rank_sum_p, save=True, figure_name=figure_name)
        for_chains = True if chain_test else False
        plot_shared_vs_disjoint_with_side_plot(shared_values, disjoint_values, title, y_lab, p_val = rank_sum_p, save=True, for_chains = for_chains, figure_name=figure_name + "_with_side_plot")

# %% [markdown]
# ### Prepare Sets

# %%
# Save all produced sets
print("Loading Rectangular")
save_folder = 'master_freeze_produced_sets/monosynaptic_rectangular/rectangular_'
W_nonzero_pairwise = {}
B_pairwise = {}
W_nonzero_out = {}  
W_nonzero_in = {}
B_out = {}
B_in = {}
for run_descriptor in run_descriptors:
    try:
        with open(f"{save_folder}{run_descriptor}W_nonzero_pairwise.pkl", "rb") as f:
            W_nonzero_pairwise_temp = pickle.load(f)
            W_nonzero_pairwise.update(W_nonzero_pairwise_temp)

    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}B_pairwise.pkl", "rb") as f:
            B_pairwise_temp = pickle.load(f)
            B_pairwise.update(B_pairwise_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")        
    try:
        with open(f"{save_folder}{run_descriptor}W_nonzero_out.pkl", "rb") as f:
            W_nonzero_out_temp = pickle.load(f)
            W_nonzero_out.update(W_nonzero_out_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}W_nonzero_in.pkl", "rb") as f:
            W_nonzero_in_temp = pickle.load(f)
            W_nonzero_in.update(W_nonzero_in_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}B_out.pkl", "rb") as f:
            B_out_temp = pickle.load(f)
            B_out.update(B_out_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}B_in.pkl", "rb") as f:
            B_in_temp = pickle.load(f)
            B_in.update(B_in_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")

monosynaptic_pairwise_contingency_table = construct_contingency_table(B_pairwise, groups)

# %% [markdown]
# ### Report Results

# %%
print("Monosynaptic Pairwise Connections by Connection Type Contingency Table:")
chi_squared_analysis(monosynaptic_pairwise_contingency_table, save=True, figure_name='Prob_Conn_by_Conn_Type')
chi_squared_analysis_v2(monosynaptic_pairwise_contingency_table, save=True, figure_name='Prob_Conn_by_Conn_Type_v2')

# %%
ranksum_signedrank_two_group_comparison(W_nonzero_pairwise,
                                        aggregation_method='connection',
                                        data_type='summed_psd',
                                        non_zero=True,
                                        save=True,
                                        figure_name='Nonzero_PSD_by_Conn'
                                        )

# %%
ranksum_signedrank_two_group_comparison(B_out,
                                        aggregation_method='cell',
                                        directionality='outbound',
                                        data_type='binary',
                                        paired=True,
                                        save=True,
                                        figure_name='Prob_Outbound_Conn'
                                        )

# %%
ranksum_signedrank_two_group_comparison(W_nonzero_out,
                                        aggregation_method='cell',
                                        directionality='outbound',
                                        data_type='summed_psd',
                                        paired=True,
                                        non_zero=True,
                                        save=True,
                                        figure_name = 'Avg_Nonzero_Outbound_PSD'
                                        )

# %%
ranksum_signedrank_two_group_comparison(B_in,
                                        aggregation_method='cell',
                                        directionality='inbound',
                                        data_type='binary',
                                        paired=True,
                                        save=True,
                                        figure_name='Prob_Inbound_Conn'
                                        )

# %%
ranksum_signedrank_two_group_comparison(W_nonzero_in,
                                        aggregation_method='cell',
                                        directionality='inbound',
                                        data_type='summed_psd',
                                        paired=True,
                                        non_zero=True,
                                        save=True,
                                        figure_name='Avg_Nonzero_Inbound_PSD'
                                        )

# %% [markdown]
# ## Higher-Order Connectivity Analysis: Centrality

# %%
def produce_centrality_plot(input_centrality_dict: dict,
                                    just_pyramidal=False,
                                    outdegree=False,
                                    indegree=False, 
                                    closeness=False, 
                                    betweenness=False,
                                    save=False,
                                    figure_name=None):
    """
    Produces a raincloud plot for centrality metrics.

    Parameters:
        input_centrality_dict (dict): Dictionary containing centrality values.
        just_pyramidal (bool): Whether to filter to pyramidal cells only.
        outdegree (bool): Whether to use outdegree centrality.
        indegree (bool): Whether to use indegree centrality.
        closeness (bool): Whether to use closeness centrality.
        betweenness (bool): Whether to use betweenness centrality.

    Returns:
        None
    """
    if outdegree and indegree:
        raise ValueError("Must either be working with outdegree or indegree.")
    if closeness and betweenness:
        raise ValueError("Must either be working with closeness or betweenness.")
    if (outdegree or indegree) and (closeness or betweenness):
        raise ValueError("Must either be working with directionality (indegree/outdegree) or higher-order (betweenness/closeness).")

    suffix = "of Co-Registered Cells"

    # Based on the connectome flags, set the correct y_label and plot title
    if outdegree:
        centrality_desc = "Outdegree_Centrality"
        suffix = "Outdegree Centrality " + suffix
        y_lab = "Outdegree Centrality"
    elif indegree:
        centrality_desc = "Indegree_Centrality"
        suffix = "Indegree Centrality " + suffix
        y_lab = "Indegree Centrality"
    elif closeness: 
        centrality_desc = "Closeness_Centrality"
        suffix = "Closeness Centrality " + suffix
        y_lab = "Closeness Centrality"
    elif betweenness:
        centrality_desc = "Betweenness_Centrality"
        suffix = "Betweenness Centrality " + suffix
        y_lab = "Betweenness Centrality"
    else:
        raise ValueError("Must Specify Degree")

    centrality_dict = {}
    for key in input_centrality_dict.keys():
        centrality_dict[key] = np.array(input_centrality_dict[key])

    all_arr = [centrality_dict['All A'], centrality_dict['No A']]
    result = stats.ranksums(centrality_dict['All A'], centrality_dict['No A'], 'greater')
    print(f"Rank-Sum Test (unpaired, All A > No A):\nStatistic: {result.statistic:.4g}, P-value: {result.pvalue:.4g}")

    # Calculate sample sizes
    n_all_a = len(centrality_dict['All A'])
    n_no_a = len(centrality_dict['No A'])

    # Create a figure
    plt.figure(figsize=(12,10))
    sns.set_theme(style="whitegrid")

    # Prepare data for raincloud plot
    data = pd.DataFrame({
        "Values": np.concatenate(all_arr),
        "Group": [f"Assembly\n(n={n_all_a})"] * len(centrality_dict['All A']) + \
                [f"Non-Assembly\n(n={n_no_a})"] * len(centrality_dict['No A'])
    })

    # Create the raincloud plot
    ax = pt.RainCloud(
        y="Values",
        x="Group",
        data=data,
        palette=[(.4, .6, .8, .5), 'grey'],
        width_viol=0.3,  # Adjust violin width
        alpha=0.8,  # Transparency of the cloud
        move=0.25,  # Adjust position of violins
        point_size = 6,
        orient="v"  # Horizontal orientation
    )

    # Set markings for significance
    y_labels = [f"Assembly\n(n={n_all_a})", f"Non-Assembly\n(n={n_no_a})"]
    pairs = [(y_labels[0], y_labels[1])]
    annot = Annotator(ax, 
                    pairs,
                    data=data,
                    x="Group",
                    y="Values",
                    order=y_labels # Force the order
                    )
    annot.set_pvalues([result.pvalue])
    annot.configure(text_format="star", loc="inside", fontsize=32)
    annot.annotate()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(32)

    # Add a multiline title to include the p-value, add y_label
    title = f'{suffix}\nRank-Sum P-value: {result.pvalue:.2g}'
    plt.title(title, size=32)
    plt.ylabel(y_lab, size=32)
    plt.xticks(fontsize=32)  # Adjust size of xticks
    plt.yticks(fontsize=32)  # Adjust size of yticks
    plt.xlabel("Assigned Assembly Status", size=32)

    if save == True:
        save_figure(figure_name)

    plt.tight_layout()
    #plt.show()

# %% [markdown]
# ### All Cells Proofread Connectome

# %%
# Pull Data from LSMM Data
print("Opening All Cells Square")
save_folder = 'master_freeze_produced_sets/centrality/all_cell_connectome_'
indegree_centrality_by_grouped_membership = {}
outdegree_centrality_by_grouped_membership = {}
closeness_centrality_by_grouped_membership = {}  
betweenness_centrality_by_grouped_membership = {}
for run_descriptor in run_descriptors:
    try:
        with open(f"{save_folder}{run_descriptor}indegree_centrality.pkl", "rb") as f:
            indegree_centrality_by_grouped_membership_temp = pickle.load(f)
            indegree_centrality_by_grouped_membership.update(indegree_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}outdegree_centrality.pkl", "rb") as f:
            outdegree_centrality_by_grouped_membership_temp = pickle.load(f)
            outdegree_centrality_by_grouped_membership.update(outdegree_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")        
    try:
        with open(f"{save_folder}{run_descriptor}closeness_centrality.pkl", "rb") as f:
            closeness_centrality_by_grouped_membership_temp = pickle.load(f)
            closeness_centrality_by_grouped_membership.update(closeness_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}betweenness_centrality.pkl", "rb") as f:
            betweenness_centrality_by_grouped_membership_temp = pickle.load(f)
            betweenness_centrality_by_grouped_membership.update(betweenness_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")

# %%
produce_centrality_plot(outdegree_centrality_by_grouped_membership,
                        outdegree = True, save=True, figure_name='Outdegree_Centrality_All')

# %%
produce_centrality_plot(indegree_centrality_by_grouped_membership,
                        indegree = True, save=True, figure_name='Indegree_Centrality_All')

# %%
produce_centrality_plot(betweenness_centrality_by_grouped_membership,
                        betweenness = True, save=True, figure_name='Betweenness_Centrality_All')

# %%
produce_centrality_plot(closeness_centrality_by_grouped_membership,
                        closeness = True, save=True, figure_name='Closeness_Centrality_All')

# %% [markdown]
# ### Pyramidal Cells Proofread Connectome

# %%
# Save all produced sets
print("Opening Pyramidal Only Square")
save_folder = 'master_freeze_produced_sets/centrality/pyr_only_connectome_'
indegree_centrality_by_grouped_membership = {}
outdegree_centrality_by_grouped_membership = {}
closeness_centrality_by_grouped_membership = {}  
betweenness_centrality_by_grouped_membership = {}
for run_descriptor in run_descriptors:
    try:
        with open(f"{save_folder}{run_descriptor}indegree_centrality.pkl", "rb") as f:
            indegree_centrality_by_grouped_membership_temp = pickle.load(f)
            indegree_centrality_by_grouped_membership.update(indegree_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}outdegree_centrality.pkl", "rb") as f:
            outdegree_centrality_by_grouped_membership_temp = pickle.load(f)
            outdegree_centrality_by_grouped_membership.update(outdegree_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")        
    try:
        with open(f"{save_folder}{run_descriptor}closeness_centrality.pkl", "rb") as f:
            closeness_centrality_by_grouped_membership_temp = pickle.load(f)
            closeness_centrality_by_grouped_membership.update(closeness_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")
    try:
        with open(f"{save_folder}{run_descriptor}betweenness_centrality.pkl", "rb") as f:
            betweenness_centrality_by_grouped_membership_temp = pickle.load(f)
            betweenness_centrality_by_grouped_membership.update(betweenness_centrality_by_grouped_membership_temp)
    except Exception as e:
        print(f"Failed to load {run_descriptor}: {e}")

# %%
produce_centrality_plot(outdegree_centrality_by_grouped_membership,
                        outdegree = True,
                        just_pyramidal = True, 
                        save=True,
                        figure_name='Outdegree_Centrality_Pyr')

# %%
produce_centrality_plot(indegree_centrality_by_grouped_membership,
                        indegree = True,
                        just_pyramidal = True, 
                        save=True,
                        figure_name='Indegree_Centrality_Pyr')

# %%
produce_centrality_plot(betweenness_centrality_by_grouped_membership,
                        betweenness = True,
                        just_pyramidal = True, 
                        save=True,
                        figure_name='Betweenness_Centrality_Pyr')

# %%
produce_centrality_plot(closeness_centrality_by_grouped_membership,
                        closeness = True,
                        just_pyramidal = True, 
                        save=True,
                        figure_name='Closeness_Centrality_Pyr')

# %% [markdown]
# ## Higher-Order Conectivity Analysis: Chain Motifs

# %% [markdown]
# ### Prep Data

# %%

for output_string in ["SquareChain", "RectChain"]:
    # %%
    # Load all produced sets
    W_chain_nonzero_pairwise_excitatory = {}
    W_chain_nonzero_pairwise_inhibitory = {}
    B_chain_pairwise_excitatory = {}
    B_chain_pairwise_inhibitory = {}
    W_nonzero_chain_out_excitatory = {}
    W_nonzero_chain_out_inhibitory = {}
    W_nonzero_chain_in_excitatory = {}
    W_nonzero_chain_in_inhibitory = {}
    B_chain_out_excitatory = {}
    B_chain_out_inhibitory = {}
    B_chain_in_excitatory = {}
    B_chain_in_inhibitory = {}
    save_folder = 'master_freeze_produced_sets/chain_connections/'
    for run_descriptor in run_descriptors:
        try:
            with open(f"{save_folder}{run_descriptor}W_chain_nonzero_pairwise_excitatory.pkl", "rb") as f:
                W_chain_nonzero_pairwise_excitatory_temp = pickle.load(f)
                W_chain_nonzero_pairwise_excitatory.update(W_chain_nonzero_pairwise_excitatory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}W_chain_nonzero_pairwise_inhibitory.pkl", "rb") as f:
                W_chain_nonzero_pairwise_inhibitory_temp = pickle.load(f)
                W_chain_nonzero_pairwise_inhibitory.update(W_chain_nonzero_pairwise_inhibitory_temp)
        except:
            continue        
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_pairwise_excitatory.pkl", "rb") as f:
                B_chain_pairwise_excitatory_temp = pickle.load(f)
                B_chain_pairwise_excitatory.update(B_chain_pairwise_excitatory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_pairwise_inhibitory.pkl", "rb") as f:
                B_chain_pairwise_inhibitory_temp = pickle.load(f)
                B_chain_pairwise_inhibitory.update(B_chain_pairwise_inhibitory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}W_nonzero_chain_out_excitatory.pkl", "rb") as f:
                W_nonzero_chain_out_excitatory_temp = pickle.load(f)
                W_nonzero_chain_out_excitatory.update(W_nonzero_chain_out_excitatory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}W_nonzero_chain_out_inhibitory.pkl", "rb") as f:
                W_nonzero_chain_out_inhibitory_temp = pickle.load(f)
                W_nonzero_chain_out_inhibitory.update(W_nonzero_chain_out_inhibitory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}W_nonzero_chain_in_excitatory.pkl", "rb") as f:
                W_nonzero_chain_in_excitatory_temp = pickle.load(f)
                W_nonzero_chain_in_excitatory.update(W_nonzero_chain_in_excitatory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}W_nonzero_chain_in_inhibitory.pkl", "rb") as f:
                W_nonzero_chain_in_inhibitory_temp = pickle.load(f)
                W_nonzero_chain_in_inhibitory.update(W_nonzero_chain_in_inhibitory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_out_excitatory.pkl", "rb") as f:
                B_chain_out_excitatory_temp = pickle.load(f)
                B_chain_out_excitatory.update(B_chain_out_excitatory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_out_inhibitory.pkl", "rb") as f:
                B_chain_out_inhibitory_temp = pickle.load(f)
                B_chain_out_inhibitory.update(B_chain_out_inhibitory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_in_excitatory.pkl", "rb") as f:
                B_chain_in_excitatory_temp = pickle.load(f)
                B_chain_in_excitatory.update(B_chain_in_excitatory_temp)
        except:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_in_inhibitory.pkl", "rb") as f:
                B_chain_in_inhibitory_temp = pickle.load(f)
                B_chain_in_inhibitory.update(B_chain_in_inhibitory_temp)
        except:
            continue

    # %% [markdown]
    # ### Plot Results

    # %%
    excitatory_contingency_table = construct_contingency_table(B_chain_pairwise_excitatory, groups)
    inhibitory_contingency_table = construct_contingency_table(B_chain_pairwise_inhibitory, groups)

    print("Excitatory Chain Contingency Table:")
    chi_squared_analysis(excitatory_contingency_table, save=True, figure_name='Prob_Conn_by_Conn_Type_E_Chains')
    chi_squared_analysis_v2(excitatory_contingency_table, save=True, figure_name='Prob_Conn_by_Conn_Type_E_Chains_v2')

    print("\nInhibitory Chain Contingency Table:")
    chi_squared_analysis(inhibitory_contingency_table, save=True, figure_name='Prob_Conn_by_Conn_Type_I_Chains')
    chi_squared_analysis_v2(inhibitory_contingency_table, save=True, figure_name='Prob_Conn_by_Conn_Type_I_Chains_v2')

    # %%
    ranksum_signedrank_two_group_comparison(W_nonzero_chain_in_inhibitory,
                                            aggregation_method='cell',
                                            directionality='inbound',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Inhibitory",
                                            save=True,
                                            figure_name=f'Avg_Nonzero_Inbound_PSD_I_Chain{output_string}'
                                            )

    # %%
    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_excitatory,
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "Excitatory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_E_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_inhibitory,
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "Inhibitory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_I_Chain{output_string}'
                                            )

    # %%
    ranksum_signedrank_two_group_comparison(B_chain_out_excitatory,
                                            aggregation_method='cell',
                                            directionality='outbound',
                                            data_type='binary',
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Excitatory",
                                            save=True,
                                            figure_name=f'Prob_Outbound_Conn_E_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(B_chain_out_inhibitory,
                                            aggregation_method='cell',
                                            directionality='outbound',
                                            data_type='binary',
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Inhibitory",
                                            save=True,
                                            figure_name=f'Prob_Outbound_Conn_I_Chain{output_string}'
                                            )

    # %%
    ranksum_signedrank_two_group_comparison(W_nonzero_chain_out_excitatory,
                                            aggregation_method='cell',
                                            directionality='outbound',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Excitatory",
                                            save=True,
                                            figure_name=f'Avg_Nonzero_Outbound_PSD_E_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_nonzero_chain_out_inhibitory,
                                            aggregation_method='cell',
                                            directionality='outbound',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Inhibitory",
                                            save=True,
                                            figure_name=f'Avg_Nonzero_Outbound_PSD_I_Chain{output_string}'
                                            )

    # %%
    ranksum_signedrank_two_group_comparison(B_chain_in_excitatory,
                                            aggregation_method='cell',
                                            directionality='inbound',
                                            data_type='binary',
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Excitatory",
                                            save=True,
                                            figure_name=f'Prob_Inbound_Conn_E_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(B_chain_in_inhibitory,
                                            aggregation_method='cell',
                                            directionality='inbound',
                                            data_type='binary',
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Inhibitory",
                                            save=True,
                                            figure_name=f'Prob_Inbound_Conn_I_Chain{output_string}'
                                            )

    # %%
    ranksum_signedrank_two_group_comparison(W_nonzero_chain_in_excitatory,
                                            aggregation_method='cell',
                                            directionality='inbound',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Excitatory",
                                            save=True,
                                            figure_name=f'Avg_Nonzero_Inbound_PSD_E_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_nonzero_chain_in_inhibitory,
                                            aggregation_method='cell',
                                            directionality='inbound',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            paired=True,
                                            chain_test=True,
                                            chain_description= "Inhibitory",
                                            save=True,
                                            figure_name=f'Avg_Nonzero_Inbound_PSD_I_Chain{output_string}'
                                            )

    # ______   __   __           _____    _____    _        _        _____   __   __  ______    _____ 
    # | ___ \  \ \ / /          /  __ \  |  ___|  | |      | |      |_   _|  \ \ / /  |  _  \  |  ___|
    # | |_/ /   \ V /           | /  \/  | |__    | |      | |        | |     \ V /   | | | |  | |__  
    # | ___ \    \ /            | |      |  __|   | |      | |        | |      \ /    | |/ /   |  __| 
    # | |_/ /    | |            | \__/\  | |___   | |____  | |____    | |      | |    | |      |___ 
    # \____/     \_/             \____/  \____/   \_____/  \_____/    \_/      \_/    |_|       \____/

    # Load all produced sets by cell type
    print("Opening Chain by Cell Type")
    save_folder = 'master_freeze_produced_sets/chain_connections/'
    W_chain_nonzero_pairwise_by_type = {}
    W_chain_nonzero_pairwise_by_type["PTC"] = {'shared':{}, 'disjoint':{}}
    W_chain_nonzero_pairwise_by_type["DTC"] = {'shared':{}, 'disjoint':{}}
    W_chain_nonzero_pairwise_by_type["ITC"] = {'shared':{}, 'disjoint':{}}
    W_chain_nonzero_pairwise_by_type["STC"] = {'shared':{}, 'disjoint':{}}
    W_chain_nonzero_pairwise_by_type["INH"] = {'shared':{}, 'disjoint':{}}
    W_chain_nonzero_pairwise_by_type["PYR"] = {'shared':{}, 'disjoint':{}}
    B_chain_pairwise_by_type = {}
    B_chain_pairwise_by_type["PTC"] = {'shared':{}, 'disjoint':{}}
    B_chain_pairwise_by_type["DTC"] = {'shared':{}, 'disjoint':{}}
    B_chain_pairwise_by_type["ITC"] = {'shared':{}, 'disjoint':{}}
    B_chain_pairwise_by_type["STC"] = {'shared':{}, 'disjoint':{}}
    B_chain_pairwise_by_type["INH"] = {'shared':{}, 'disjoint':{}}
    B_chain_pairwise_by_type["PYR"] = {'shared':{}, 'disjoint':{}}

    for run_descriptor in run_descriptors:
        try:
            with open(f"{save_folder}{run_descriptor}W_chain_nonzero_pairwise_by_type.pkl", "rb") as f:
                W_chain_nonzero_pairwise_by_type_temp = pickle.load(f)
                for cell_type in W_chain_nonzero_pairwise_by_type_temp:
                    for comparison in ['shared', 'disjoint']:
                        if comparison not in W_chain_nonzero_pairwise_by_type[cell_type]:
                            W_chain_nonzero_pairwise_by_type[cell_type][comparison] = W_chain_nonzero_pairwise_by_type_temp[cell_type][comparison]
                        else:
                            W_chain_nonzero_pairwise_by_type[cell_type][comparison].update(W_chain_nonzero_pairwise_by_type_temp[cell_type][comparison])
        except FileNotFoundError:
            continue
        try:
            with open(f"{save_folder}{run_descriptor}B_chain_pairwise_by_type.pkl", "rb") as f:
                B_chain_pairwise_by_type_temp = pickle.load(f)
                for cell_type in B_chain_pairwise_by_type_temp:
                    for comparison in ['shared', 'disjoint']:
                        if comparison not in B_chain_pairwise_by_type[cell_type]:
                            B_chain_pairwise_by_type[cell_type][comparison] = B_chain_pairwise_by_type_temp[cell_type][comparison]
                        else:
                            B_chain_pairwise_by_type[cell_type][comparison].update(B_chain_pairwise_by_type_temp[cell_type][comparison])
        except FileNotFoundError:
            continue

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_by_type['PTC'],
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "ProxTC Inhibitory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_PTC_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_by_type['DTC'],
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "DistTC Inhibitory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_DTC_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_by_type['ITC'],
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "InhTC Inhibitory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_ITC_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_by_type['STC'],
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "SparTC Inhibitory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_STC_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_by_type['INH'],
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "Lumped Inhibitory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_LumpedInh_Chain{output_string}'
                                            )

    ranksum_signedrank_two_group_comparison(W_chain_nonzero_pairwise_by_type['PYR'],
                                            aggregation_method='connection',
                                            data_type='summed_psd',
                                            non_zero=True,
                                            chain_test=True,
                                            chain_description= "Excitatory",
                                            save=True,
                                            figure_name=f'Nonzero_PSD_by_Conn_Pyr_Chain{output_string}'
                                            )
