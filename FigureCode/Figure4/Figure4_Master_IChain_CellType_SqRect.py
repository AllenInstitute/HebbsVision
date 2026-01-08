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

# Change to 2 for final run in merged
for scan_session_affinity_filestring in ['1_2_4_742', '1_3_4_742', '1_4_4_742']:
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
    with open(f'./FigureCode/Figure4/pyr_cells_rectangular_connectome_{scan_session_affinity_filestring}.json') as f:
        pyr_cells_rect_lsmm_json_input = json.load(f)
    pyr_cells_rect_v1dd_data = LSMMData.LSMMData(pyr_cells_rect_lsmm_json_input)

    data_a = pyr_cells_rect_v1dd_data.data
    params_a = pyr_cells_rect_v1dd_data.params
    dirs_a = pyr_cells_rect_v1dd_data.dirs
    mappings_a = pyr_cells_rect_v1dd_data.mappings

    print("Coregistration")
    print(data_a.keys())

    print("Assemblies")


    # %% [markdown]
    # ## Monosynaptic Analysis on Pyramidal Cell Rectangular Connectome

    # %%
    def save_figure(figure_name):
        plt.savefig(
            f"./draft_figures/{figure_name}_{params_a['run_descriptor']}.png",
            dpi=500,
            bbox_inches="tight")

    def save_values(values_name, first_values, second_values):
        with open(f"./values/{values_name}_{params_a['run_descriptor']}.pkl", "wb") as f:
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
    ### Pull necessary data from V1DD using LSMMData Manager
    cell_table = data_a['structural']['pre_cell'].copy()
    cell_table['connectome_index'] = cell_table.index
    post_cell_table = data_a['structural']['post_cell'].copy()
    post_cell_table['connectome_index'] = post_cell_table.index
    synapse_table = data_a['structural']['synapse']

    # Establish seperate sets for the pre and post synaptic partnes
    # This is necessary as the set of connectome index of pre-synaptic cells do not 
    # match the post-synaptic cells due to allowing unproofread post-synaptic targets.
    assemblies = set(mappings_a['connectome_indexes_by_assembly'].keys())
    assemblies = assemblies - set('No A')
    # print(mappings_a['connectome_indexes_by_assembly'].keys())
    individual_assembly_indexes = [mappings_a['connectome_indexes_by_assembly'][i] for i in list(assemblies)]
    individual_post_assembly_indexes = [mappings_a['post_connectome_indexes_by_assembly'][i] for i in list(assemblies)]

    coregistered_post_cell_indexes = mappings_a['assemblies_by_post_connectome_index'].keys()
    coregistered_cell_indexes = mappings_a['assemblies_by_connectome_index'].keys()

    no_a_cell_indexes = mappings_a['connectome_indexes_by_assembly']['No A']
    no_a_post_cell_indexes = mappings_a['post_connectome_indexes_by_assembly']['No A']

    pooled_assembly_indexes = list(set(coregistered_cell_indexes) - set(no_a_cell_indexes))
    pooled_assembly_post_indexes = list(set(coregistered_post_cell_indexes) - set(no_a_post_cell_indexes))
    # Each cell has a distinct root id, so it is unnecessary to establish different sets
    assembly_to_root_ids = mappings_a['pt_root_ids_by_assembly']
    assembly_root_ids_set = set(mappings_a['assemblies_by_pt_root_id'].keys())

    # Filter synapses_table to only synapses between two assembly cells (including No A)
    synapses_df = synapse_table[synapse_table['pre_pt_root_id'].isin(assembly_root_ids_set)]
    synapses_df = synapses_df[synapses_df['post_pt_root_id'].isin(assembly_root_ids_set)]
    synapses_df['size'] = synapses_df['size'] * (9 * 9 * 45) / (10**9) # Voxels -> Cubic micrometers

    # Filter cell tables to only assembly cells
    cell_table = cell_table[cell_table['pt_root_id'].isin(assembly_root_ids_set)]
    post_cell_table = post_cell_table[post_cell_table['pt_root_id'].isin(assembly_root_ids_set)]

    # Finalized set of Root IDs, which are 
    pre_root_ids = set(cell_table['pt_root_id'].values)
    post_root_ids = set(post_cell_table['pt_root_id'].values)
    all_root_ids = pre_root_ids | post_root_ids

    # %%
    ### Prep the sets for Analysis, following our description in the Methods section
    # Collect our connectomes of pre and post synaptic sets based on the root_ids of the neurons
    w = {}
    s = {}
    b = {}
    for pre in pre_root_ids:
        for post in post_root_ids:
            if pre != post:
                w[(pre, post)] = 0
                s[(pre, post)] = 0
                b[(pre, post)] = 0

    for i, row in synapses_df.iterrows():
        pre = row['pre_pt_root_id']
        post = row['post_pt_root_id']
        if (pre, post) in w:
            w[(pre, post)] += row['size']
            s[(pre, post)] += 1
            b[(pre, post)] = 1
        # else:
        #     w[(pre, post)] = row['size']
        #     s[(pre, post)] = 1
        

    # Split out assemblies and no_a
    assembly_names = set(assembly_to_root_ids.keys()) - set(['No A'])
    A_invert = {assembly: set(assembly_to_root_ids[assembly]) for assembly in assembly_names}
    no_A = set(assembly_to_root_ids['No A'])
    all_coregistered_root_ids = mappings_a['assemblies_by_pt_root_id'].keys()
    assembly_root_ids_excluding_no_A = set(all_coregistered_root_ids) - no_A
    A = {pt_root_id: set(mappings_a['assemblies_by_pt_root_id'][pt_root_id]) for pt_root_id in all_root_ids if 'No A' not in mappings_a['assemblies_by_pt_root_id'][pt_root_id]}

    # %%
    W_nonzero_pairwise = {}
    B_pairwise = {}
    for connection_type in comparison_functions:
        W_nonzero_pairwise[connection_type.__name__] = {}
        B_pairwise[connection_type.__name__] = {}
        for (j, i) in w.keys():
            if connection_type(j, i, A):
                B_pairwise[connection_type.__name__][(j, i)] = 1 if w[(j, i)] > 0 else 0
                if w[(j, i)] > 0:
                    W_nonzero_pairwise[connection_type.__name__][(j, i)] = w[(j, i)]

    W_nonzero_out = {}
    for connection_type in comparison_functions:
        W_nonzero_out[connection_type.__name__] = {}
        for j in pre_root_ids:
            if len([i for i in post_root_ids if i != j and connection_type(j, i, A) and w[(j, i)] > 0]) > 0:
                W_nonzero_out[connection_type.__name__][j] = sum([w[(j, i)] for i in post_root_ids if connection_type(j, i, A) and j != i]) / len([i for i in post_root_ids if i != j and connection_type(j, i, A) and w[(j, i)] > 0])

    W_nonzero_in = {}
    for connection_type in comparison_functions:
        W_nonzero_in[connection_type.__name__] = {}
        for i in post_root_ids:
            if len([j for j in pre_root_ids if connection_type(j, i, A)]) > 0 and len([j for j in pre_root_ids if j != i and connection_type(j, i, A) and w[(j,i)] > 0]):
                W_nonzero_in[connection_type.__name__][i] = sum([w[(j, i)] for j in pre_root_ids if connection_type(j, i, A) and i != j]) / len([j for j in pre_root_ids if j != i and connection_type(j, i, A) and w[(j,i)] > 0])

    B_out = {}
    for connection_type in comparison_functions:
        B_out[connection_type.__name__] = {}
        for j in pre_root_ids:
            if len([i for i in post_root_ids if connection_type(j, i, A) and j != i]) > 0:
                B_out[connection_type.__name__][j] = sum([b[(j, i)] for i in post_root_ids if connection_type(j, i, A) and j != i]) / len([i for i in post_root_ids if connection_type(j, i, A) and j != i])

    B_in = {}
    for connection_type in comparison_functions:
        B_in[connection_type.__name__] = {}
        for i in post_root_ids:
            if len([j for j in pre_root_ids if connection_type(j, i, A) and i != j]) > 0:
                B_in[connection_type.__name__][i] = sum([b[(j, i)] for j in pre_root_ids if connection_type(j, i, A) and i != j]) / len([j for j in pre_root_ids if connection_type(j, i, A) and j != i])
    # create contingency table for monosynaptic connections count by connection type
    monosynaptic_pairwise_contingency_table = construct_contingency_table(B_pairwise, groups)

    # %%
    # Save all produced sets
    print("Saving Rectangular")
    save_folder = 'master_freeze_produced_sets/monosynaptic_rectangular/rectangular_'
    with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_pairwise.pkl", "wb") as f:
        pickle.dump(W_nonzero_pairwise, f)
    with open(f"{save_folder}{params_a['run_descriptor']}B_pairwise.pkl", "wb") as f:
        pickle.dump(B_pairwise, f)
    with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_out.pkl", "wb") as f:
        pickle.dump(W_nonzero_out, f)
    with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_in.pkl", "wb") as f:
        pickle.dump(W_nonzero_in, f)
    with open(f"{save_folder}{params_a['run_descriptor']}B_out.pkl", "wb") as f:
        pickle.dump(W_nonzero_out, f)
    with open(f"{save_folder}{params_a['run_descriptor']}B_in.pkl", "wb") as f:
        pickle.dump(W_nonzero_in, f)

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
    with open(f'FigureCode/Figure4/all_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        all_cells_square_v1dd_datalsmm_json_input = json.load(f)
    all_cells_square_v1dd_data = LSMMData.LSMMData(all_cells_square_v1dd_datalsmm_json_input)

    data_a = all_cells_square_v1dd_data.data
    params_a = all_cells_square_v1dd_data.params
    dirs_a = all_cells_square_v1dd_data.dirs
    mappings_a = all_cells_square_v1dd_data.mappings

    # %%
    # Calculate Centrality Measurements
    binary_connectome = data_a['structural']['binary_connectome']
    all_to_all_graph = nx.from_numpy_array(binary_connectome, create_using=nx.DiGraph)

    indegree_centrality = nx.in_degree_centrality(all_to_all_graph)
    outdegree_centrality = nx.out_degree_centrality(all_to_all_graph)
    closeness_centrality = nx.closeness_centrality(all_to_all_graph, wf_improved = True)
    betweenness_centrality = nx.betweenness_centrality(all_to_all_graph, normalized= True)

    # %%
    connectome_index_by_assemblies = mappings_a['connectome_indexes_by_assembly']
    assembly_connectome_indexes = np.unique(np.concatenate([val for key, val in connectome_index_by_assemblies.items() if key != 'No A']))
    no_assembly_connectome_indexes = np.array(list(connectome_index_by_assemblies['No A']))

    # Produce Grouped Counts for Inbound and Outbound Connections
    indegree_centrality_by_grouped_membership = {'No A': [], 'All A': []}
    outdegree_centrality_by_grouped_membership = {'No A': [], 'All A': []}
    closeness_centrality_by_grouped_membership = {'No A': [], 'All A': []}
    betweenness_centrality_by_grouped_membership = {'No A': [], 'All A': []}

    # Add to dictionaries to plot
    for assembly_cell_idx in assembly_connectome_indexes:
        indegree_centrality_by_grouped_membership['All A'].append(indegree_centrality[assembly_cell_idx])
        outdegree_centrality_by_grouped_membership['All A'].append(outdegree_centrality[assembly_cell_idx])
        closeness_centrality_by_grouped_membership['All A'].append(closeness_centrality[assembly_cell_idx])
        betweenness_centrality_by_grouped_membership['All A'].append(betweenness_centrality[assembly_cell_idx])
    for no_assembly_cell_idx in no_assembly_connectome_indexes:
        indegree_centrality_by_grouped_membership['No A'].append(indegree_centrality[no_assembly_cell_idx])
        outdegree_centrality_by_grouped_membership['No A'].append(outdegree_centrality[no_assembly_cell_idx])
        closeness_centrality_by_grouped_membership['No A'].append(closeness_centrality[no_assembly_cell_idx])
        betweenness_centrality_by_grouped_membership['No A'].append(betweenness_centrality[no_assembly_cell_idx])

    # %%
    # Save all produced sets
    save_folder = 'master_freeze_produced_sets/centrality/all_cell_connectome_'
    with open(f"{save_folder}{params_a['run_descriptor']}indegree_centrality.pkl", "wb") as f:
        pickle.dump(indegree_centrality_by_grouped_membership, f)
    with open(f"{save_folder}{params_a['run_descriptor']}outdegree_centrality.pkl", "wb") as f:
        pickle.dump(outdegree_centrality_by_grouped_membership, f)
    with open(f"{save_folder}{params_a['run_descriptor']}closeness_centrality.pkl", "wb") as f:
        pickle.dump(closeness_centrality_by_grouped_membership, f)
    with open(f"{save_folder}{params_a['run_descriptor']}betweenness_centrality.pkl", "wb") as f:
        pickle.dump(betweenness_centrality_by_grouped_membership, f)

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
    # Pull Data from LSMM Data
    print("Opening Pyr Cells Square")
    with open(f'FigureCode/Figure4/pyr_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        lsmm_json_input = json.load(f)
    v1dd_data = LSMMData.LSMMData(lsmm_json_input)

    data_a = v1dd_data.data
    params_a = v1dd_data.params
    dirs_a = v1dd_data.dirs
    mappings_a = v1dd_data.mappings

    # %%
    # Calculate Centrality Measurements
    binary_connectome = data_a['structural']['binary_connectome']
    all_to_all_graph = nx.from_numpy_array(binary_connectome, create_using=nx.DiGraph)

    indegree_centrality = nx.in_degree_centrality(all_to_all_graph)
    outdegree_centrality = nx.out_degree_centrality(all_to_all_graph)
    closeness_centrality = nx.closeness_centrality(all_to_all_graph, wf_improved = True)
    betweenness_centrality = nx.betweenness_centrality(all_to_all_graph, normalized= True)

    # %%
    connectome_index_by_assemblies = mappings_a['connectome_indexes_by_assembly']
    assembly_connectome_indexes = np.unique(np.concatenate([val for key, val in connectome_index_by_assemblies.items() if key != 'No A']))
    no_assembly_connectome_indexes = np.array(list(connectome_index_by_assemblies['No A']))

    # Produce Grouped Counts for Inbound and Outbound Connections
    indegree_centrality_by_grouped_membership = {'No A': [], 'All A': []}
    outdegree_centrality_by_grouped_membership = {'No A': [], 'All A': []}
    closeness_centrality_by_grouped_membership = {'No A': [], 'All A': []}
    betweenness_centrality_by_grouped_membership = {'No A': [], 'All A': []}

    # Add to dictionaries to plot
    for assembly_cell_idx in assembly_connectome_indexes:
        indegree_centrality_by_grouped_membership['All A'].append(indegree_centrality[assembly_cell_idx])
        outdegree_centrality_by_grouped_membership['All A'].append(outdegree_centrality[assembly_cell_idx])
        closeness_centrality_by_grouped_membership['All A'].append(closeness_centrality[assembly_cell_idx])
        betweenness_centrality_by_grouped_membership['All A'].append(betweenness_centrality[assembly_cell_idx])
    for no_assembly_cell_idx in no_assembly_connectome_indexes:
        indegree_centrality_by_grouped_membership['No A'].append(indegree_centrality[no_assembly_cell_idx])
        outdegree_centrality_by_grouped_membership['No A'].append(outdegree_centrality[no_assembly_cell_idx])
        closeness_centrality_by_grouped_membership['No A'].append(closeness_centrality[no_assembly_cell_idx])
        betweenness_centrality_by_grouped_membership['No A'].append(betweenness_centrality[no_assembly_cell_idx])

    # %%
    # Save all produced sets
    save_folder = 'master_freeze_produced_sets/centrality/pyr_only_connectome_'
    with open(f"{save_folder}{params_a['run_descriptor']}indegree_centrality.pkl", "wb") as f:
        pickle.dump(indegree_centrality_by_grouped_membership, f)
    with open(f"{save_folder}{params_a['run_descriptor']}outdegree_centrality.pkl", "wb") as f:
        pickle.dump(outdegree_centrality_by_grouped_membership, f)
    with open(f"{save_folder}{params_a['run_descriptor']}closeness_centrality.pkl", "wb") as f:
        pickle.dump(closeness_centrality_by_grouped_membership, f)
    with open(f"{save_folder}{params_a['run_descriptor']}betweenness_centrality.pkl", "wb") as f:
        pickle.dump(betweenness_centrality_by_grouped_membership, f)

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

    loaded_jsons = []
    with open(f'FigureCode/Figure4/all_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        loaded_jsons.append(json.load(f))

    with open(f'FigureCode/Figure4/all_cells_rectangular_connectome_{scan_session_affinity_filestring}.json') as f:
        loaded_jsons.append(json.load(f))

    for to_proofread in [False, True]:
        if to_proofread:
            output_string = "SquareChain"
            loaded_json = loaded_jsons[0]
            # loaded_json['proofread_to_proofread'] = True

        else:
            output_string = "RectChain"
            loaded_json = loaded_jsons[1]   
            # loaded_json['proofread_to_proofread'] = False

        print(f"Opening {output_string} Data")
        my_data = LSMMData.LSMMData(loaded_json)

        data_a = my_data.data
        params_a = my_data.params
        dirs_a = my_data.dirs
        mappings_a = my_data.mappings

        # Make a graph
        pre_cell_table = data_a['structural']['pre_cell'].copy()
        post_cell_table = data_a['structural']['post_cell'].copy()
        pre_cell_table['connectome_index'] = pre_cell_table.index
        post_cell_table['connectome_index'] = post_cell_table.index
        synapse_table = data_a['structural']['synapse']
        # adjacency_matrix = square_data['structural']['binary_connectome']
        temp_graph = nx.DiGraph()

        # Reset all variables at the start of each iteration
            # Filter cell tables to only assembly cells
        assembly_cell_table = cell_table[cell_table['pt_root_id'].isin(assembly_root_ids_set)]
        assembly_post_cell_table = post_cell_table[post_cell_table['pt_root_id'].isin(assembly_root_ids_set)]

        # Finalized set of Root IDs, which are 
        pre_root_ids = set(assembly_cell_table['pt_root_id'].values)
        post_root_ids = set(assembly_post_cell_table['pt_root_id'].values)
        all_root_ids = pre_root_ids | post_root_ids
        two_chain_results_array = np.array([])
        W_chain_excitatory = {}
        W_chain_inhibitory = {}
        B_chain_excitatory = {}
        B_chain_inhibitory = {}
        A = {pt_root_id: set(mappings_a['assemblies_by_pt_root_id'][pt_root_id]) for pt_root_id in all_root_ids if 'No A' not in mappings_a['assemblies_by_pt_root_id'][pt_root_id]}

        print(f"{len(pre_cell_table)}, {len(post_cell_table)} cells")
        
        # summed_size_connectome = data_a['structural']['summed_size_connectome']
        pre_index_to_root_id = [mappings_a['connectome_index_to_root_id'][i] for i in range(len(mappings_a['connectome_index_to_root_id']))]
        post_index_to_root_id = [mappings_a['post_connectome_index_to_root_id'][i] for i in range(len(mappings_a['post_connectome_index_to_root_id']))]

        summed_size_connectome = data_a['structural']['summed_size_connectome']
        summed_size_connectome_df = pd.DataFrame(
            summed_size_connectome,
            index=pre_index_to_root_id,
            columns=post_index_to_root_id
        )

        # Add edges to the graph
        rows, cols = data_a['structural']['binary_connectome'].shape
        for i in range(rows):
            for j in range(cols):
                if data_a['structural']['binary_connectome'][i, j] != 0:  # 0 means no edge
                    temp_graph.add_edge(pre_cell_table.iloc[i]['pt_root_id'], post_cell_table.iloc[j]['pt_root_id'], weight=data_a['structural']['summed_size_connectome'][i, j])

        # Motif Analysis with DotMotif: 2 Chain, All Pyr
        executor = GrandIsoExecutor(graph=temp_graph)
        chain_defs = Motif("""
                        A -> B
                        B -> C
                    """)

        chain_results = executor.find(chain_defs)

        two_chain_results_array = np.array([list(c.values()) for c in tqdm(chain_results)])

        ### Pool necessary Data
        chain_count_string_array = ['pyr_cell_2chain']
        individual_assembly_indexes = [mappings_a['connectome_indexes_by_assembly'][f'{i}'] for i in mappings_a['connectome_indexes_by_assembly'].keys() if i[0] == 'A']
        individual_post_assembly_indexes = [mappings_a['post_connectome_indexes_by_assembly'][f'{i}'] for i in mappings_a['post_connectome_indexes_by_assembly'].keys() if i[0] == 'A']

        coregistered_post_cell_indexes = mappings_a['assemblies_by_post_connectome_index'].keys()
        coregistered_cell_indexes = mappings_a['assemblies_by_connectome_index'].keys()
        no_a_cell_indexes = mappings_a['connectome_indexes_by_assembly']['No A']
        no_a_post_cell_indexes = mappings_a['post_connectome_indexes_by_assembly']['No A']
        pooled_assembly_indexes = list(set(coregistered_cell_indexes) - set(no_a_cell_indexes))
        pooled_assembly_post_indexes = list(set(coregistered_post_cell_indexes) - set(no_a_post_cell_indexes))

        assembly_to_root_ids = mappings_a['pt_root_ids_by_assembly']
        assembly_root_ids_set = set(mappings_a['assemblies_by_pt_root_id'].keys())

        # Filter synapses_table to only synapses between two assembly cells (including No A)
        synapses_df = synapse_table[synapse_table['pre_pt_root_id'].isin(assembly_root_ids_set)]
        synapses_df = synapses_df[synapses_df['post_pt_root_id'].isin(assembly_root_ids_set)]

        # Filter cell tables to only assembly cells (including 'No A')
        assembly_cell_table = pre_cell_table[pre_cell_table['pt_root_id'].isin(assembly_root_ids_set)]
        assembly_post_cell_table = post_cell_table[post_cell_table['pt_root_id'].isin(assembly_root_ids_set)]

        assembly_pre_root_ids = set(assembly_cell_table['pt_root_id'].values)
        assembly_post_root_ids = set(assembly_post_cell_table['pt_root_id'].values)
        all_root_ids = assembly_pre_root_ids | assembly_post_root_ids
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Assembly Cells Count: {len(all_root_ids)}")
        print(f"Two Chain Results Count: {len(two_chain_results_array)}")
        print(f"Assembly Pre Cells Count: {len(assembly_pre_root_ids)}")
        print(f"Assembly Post Cells Count: {len(assembly_post_root_ids)}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # additional_post_cells = post_root_ids - pre_root_ids
        # for i in additional_post_cells:
        #     chains_for_i = two_chain_results_array[(two_chain_results_array[:, 2] == i)]
        #     valid_chains = []
        #     for chain in chains_for_i:
        #         j, k, i_chain = chain
        #         for connection_type in comparison_functions:
        #             if connection_type(j, i, A):
        #                 valid_chains.append(chain)
        #                 # print(f"Post-synaptic cell {i} has {len(valid_chains)} valid inbound chains with {connection_type.__name__} cells.")
        
        # for i in additional_post_cells:
        #     chains_for_i = two_chain_results_array[(two_chain_results_array[:, 2] == i)]
        #     for chain in chains_for_i:
        #         j, k, i_chain = chain
        #         for connection_type in comparison_functions:
        #             if connection_type(j, i, A):
        #                 # print(f"Chain {j} -> {k} -> {i} satisfies the connection type condition.")
        #                 # print(f"Assembly membership of {j}: {A.get(j, set())}")
        #                 # print(f"Assembly membership of {i}: {A.get(i, set())}")

        # # Filter chains terminating on additional post-synaptic cells
        # additional_chains = two_chain_results_array[
        #     np.isin(two_chain_results_array[:, 2], list(additional_post_cells))
        # ]

        # print(f"Total chains terminating on additional post cells: {len(additional_chains)}")

        # # Collect all unique assembly memberships for origin and terminus cells (j, i)
        # all_origin_assembly_memberships = set()
        # all_terminus_assembly_memberships = set()
        # for chain in additional_chains:
        #     j, k, i = chain
        #     memberships = mappings_a['assemblies_by_pt_root_id'].get(j, set())
        #     all_origin_assembly_memberships.update(memberships)
        #     memberships2 = mappings_a['assemblies_by_pt_root_id'].get(i, set())
        #     all_terminus_assembly_memberships.update(memberships2)

        # # Print the set of all unique assembly memberships
        # print("Unique assembly memberships for origin cells in chains terminating on additional post cells:")
        # print(all_origin_assembly_memberships)
        # print("Unique assembly memberships for terminus cells in chains terminating on additional post cells:")
        # print(all_terminus_assembly_memberships)

        # Collect all unique assembly memberships for terminus cells (j)
        
        # for chain in additional_chains:
        #     j, k, i = chain
            

        # # Print the set of all unique assembly memberships
        # print("Unique assembly memberships for origin cells in chains terminating on additional post cells:")
        # print(all_origin_assembly_memberships)

        # Pause while awaiting user input
        # input(f"\nFinished with {scan_session_affinity_filestring}: {output_string}\nPress Enter to continue...")
 
        # %%
        # Initialize dictionaries to store weights and binary connectivity
        W_chain_excitatory = {}
        W_chain_inhibitory = {}
        B_chain_excitatory = {}
        B_chain_inhibitory = {}

        # Define all potential (pre-cell, post-cell) pairs with excitatory and inhibitory chain types
        for j in all_root_ids:
            for i in all_root_ids:
                if j != i:  # Exclude autapses
                    W_chain_excitatory[(j, i)] = 0
                    W_chain_inhibitory[(j, i)] = 0
                    B_chain_excitatory[(j, i)] = 0
                    B_chain_inhibitory[(j, i)] = 0

        pt_root_id_to_classification = post_cell_table.set_index('pt_root_id')['classification_system'].to_dict()
        print("PT Root ID to Classification Mapping:")
        print(pt_root_id_to_classification)
        print(post_cell_table.set_index('pt_root_id')['classification_system'])


        # Process each row in `two_chain_results_array` to populate weights and binary connectivity
        for _, row in tqdm(enumerate(two_chain_results_array)):
            pre_cell, mid_cell, post_cell = row  # j: pre-cell, k: middle cell, i: post-cell
            if pre_cell in all_root_ids and post_cell in all_root_ids:
            # Determine chain type (excitatory if middle cell is in excitatory set, else inhibitory)
                if pt_root_id_to_classification[mid_cell] == 'inhibitory':
                    W_chain = W_chain_inhibitory
                    B_chain = B_chain_inhibitory
                elif pt_root_id_to_classification[mid_cell] == 'excitatory':
                    W_chain = W_chain_excitatory
                    B_chain = B_chain_excitatory
            # Get synapse weights for connections j -> k and k -> i
                w_jk = summed_size_connectome_df.loc[pre_cell, mid_cell] * (9 * 9 * 45) / (10**9) # cubic micrometers
                w_ki = summed_size_connectome_df.loc[mid_cell, post_cell] * (9 * 9 * 45) / (10**9) # cubic micrometers
            
            # Updates weights and binary connectivity
                W_chain[(pre_cell, post_cell)] += (w_jk * w_ki)
                B_chain[(pre_cell, post_cell)] = 1

        # %%
        ## PARIWISE NONZERO PSD and CONNECTION PROBABILITY BY CONNECTION TYPE
        # Aggregate nonzero pairs and calculate connectivity probabilities by connection type
        W_chain_nonzero_pairwise_excitatory = {}
        W_chain_nonzero_pairwise_inhibitory = {}
        B_chain_pairwise_excitatory = {}
        B_chain_pairwise_inhibitory = {}

        for cond_function in comparison_functions:
            # Initialize dictionaries per connection type
            W_chain_nonzero_pairwise_excitatory[cond_function.__name__] = {}
            W_chain_nonzero_pairwise_inhibitory[cond_function.__name__] = {}
            B_chain_pairwise_excitatory[cond_function.__name__] = {}
            B_chain_pairwise_inhibitory[cond_function.__name__] = {}

            # Process all (j, i) pairs from excitatory/inhibitory dictionaries
            for (j, i) in W_chain_excitatory.keys():
                if cond_function(j, i, A):
                    # Set binary connectivity for each connection type
                    B_chain_pairwise_excitatory[cond_function.__name__][(j, i)] = 1 if W_chain_excitatory[(j, i)] > 0 else 0
                    B_chain_pairwise_inhibitory[cond_function.__name__][(j, i)] = 1 if W_chain_inhibitory[(j, i)] > 0 else 0
                    
                    # Store only nonzero weights
                    if W_chain_excitatory[(j, i)] > 0:
                        W_chain_nonzero_pairwise_excitatory[cond_function.__name__][(j, i)] = W_chain_excitatory[(j, i)]
                    if W_chain_inhibitory[(j, i)] > 0:
                        W_chain_nonzero_pairwise_inhibitory[cond_function.__name__][(j, i)] = W_chain_inhibitory[(j, i)]


        ## ADD IN NORMALIZATION BY # POTENTIAL CONNECTIONS
        ## Inbound, Outbound Collections
        from collections import Counter

        classification_map = cell_table.set_index('connectome_index')['classification_system'].to_dict() # Map to tell us if its Excitatory or Inhibitory 
        backup_assembly_pre_root_ids = assembly_pre_root_ids
        backup_assembly_post_root_ids = assembly_post_root_ids
        classification_counts = Counter(classification_map.values())

        num_excitatory = classification_counts.get('excitatory', 0)
        num_inhibitory = classification_counts.get('inhibitory', 0)
        # Aggregate chain weights and connectivity for outbound and inbound paths
        W_nonzero_chain_out_excitatory = {}
        W_nonzero_chain_out_inhibitory = {}
        W_nonzero_chain_in_excitatory = {}
        W_nonzero_chain_in_inhibitory = {}
        B_chain_out_excitatory = {}
        B_chain_out_inhibitory = {}
        B_chain_in_excitatory = {}
        B_chain_in_inhibitory = {}

        # Iterate through connection types in C (e.g., shared, disjoint)
        for connection_type in comparison_functions:
            # Initialize per connection type dictionaries
            W_nonzero_chain_out_excitatory[connection_type.__name__] = {}
            W_nonzero_chain_out_inhibitory[connection_type.__name__] = {}
            W_nonzero_chain_in_excitatory[connection_type.__name__] = {}
            W_nonzero_chain_in_inhibitory[connection_type.__name__] = {}
            B_chain_out_excitatory[connection_type.__name__] = {}
            B_chain_out_inhibitory[connection_type.__name__] = {}
            B_chain_in_excitatory[connection_type.__name__] = {}
            B_chain_in_inhibitory[connection_type.__name__] = {}
            # Outbound analysis (from pre-cell j to post-cell i through middle cell k)
            print("Beginning Outbound Analysis")
            for j in all_root_ids:
                potential_partners_out = {i for i in all_root_ids if i != j and connection_type(j, i, A)}
                if potential_partners_out:  # Only proceed if valid post-cell partners exist
                    realized_chains_count_excitatory = 0
                    realized_chains_count_inhibitory = 0
                    normalized_realized_chains_count_excitatory = 0
                    normalized_realized_chains_count_inhibitory = 0
                    sum_weights_excitatory = 0
                    sum_weights_inhibitory = 0
                    
                    potential_inhibitory_chains = num_inhibitory * len(potential_partners_out)
                    for i in potential_partners_out:
                        if i in assembly_pre_root_ids:
                            potential_excitatory_chains = (num_excitatory - 2) * (len(potential_partners_out)) # maybe middle cell is in set of coregistered that satisfies condition
                        else:
                            potential_excitatory_chains = (num_excitatory - 1) * (len(potential_partners_out)) # maybe middle cell is in set of coregistered that satisfies condition
                        temporary_chain_results = two_chain_results_array[(two_chain_results_array[:, 0] == j) & (two_chain_results_array[:, 2] == i)]
                        for chain in temporary_chain_results:
                            j, k, i = chain
                            w_jk = summed_size_connectome_df.loc[j, k]
                            w_ki = summed_size_connectome_df.loc[k, i]
                            # Check chain type and accumulate only when both segments have valid weights
                            if pt_root_id_to_classification.get(k) == 'excitatory':
                                # print(f"Outbound: Processing excitatory chain: {j} -> {k} -> {i} with weights {w_jk}, {w_ki}")
                                sum_weights_excitatory += w_jk * w_ki
                                realized_chains_count_excitatory += 1
                                normalized_realized_chains_count_excitatory += 1 / potential_excitatory_chains
                            elif pt_root_id_to_classification.get(k) == 'inhibitory':
                                # print(f"Outbound: Processing inhibitory chain: {j} -> {k} -> {i} with weights {w_jk}, {w_ki}")
                                sum_weights_inhibitory += w_jk * w_ki
                                realized_chains_count_inhibitory += 1
                                # normalized_realized_chains_count_inhibitory += 1 / potential_inhibitory_chains
                        # Normalize weights by realized chain count only if nonzero
                        if realized_chains_count_excitatory > 0:
                            W_nonzero_chain_out_excitatory[connection_type.__name__][j] = sum_weights_excitatory / realized_chains_count_excitatory
                        if realized_chains_count_inhibitory > 0:
                            W_nonzero_chain_out_inhibitory[connection_type.__name__][j] = sum_weights_inhibitory / realized_chains_count_inhibitory

                    # Calculate binary connectivity only if there are potential partners
                    B_chain_out_excitatory[connection_type.__name__][j] = normalized_realized_chains_count_excitatory
                    B_chain_out_inhibitory[connection_type.__name__][j] = realized_chains_count_inhibitory / potential_inhibitory_chains if potential_inhibitory_chains > 0 else 0 # We can normalize here, since number of potential inhibitory partners is constant over i
        
            # Inbound analysis (make sure we get all post-cells?)
            print("Beginning Inbound Analysis")
            for i in all_root_ids:
                # if i in list(all_root_ids)[:5]:
                    # print(f"Processing post-cell {i} for inbound chains.")
                potential_partners_in = [j for j in all_root_ids if i != j and connection_type(j, i, A)]
                # print (f"Found {len(potential_partners_in)} potential pre-cell partners for post-cell {i}.")
                if len(potential_partners_in) > 0:  # Only proceed if valid post-cell partners exist
                    # print(f"\tPassed if statement...")
                    realized_chains_count_excitatory = 0
                    realized_chains_count_inhibitory = 0
                    normalized_realized_chains_count_excitatory = 0
                    normalized_realized_chains_count_inhibitory = 0
                    sum_weights_excitatory = 0
                    sum_weights_inhibitory = 0

                    potential_inhibitory_chains = num_inhibitory * len(potential_partners_in)
                    for j in potential_partners_in:
                        # if j == potential_partners_in[0]:
                            # print(f"\tProcessing pre-cell {j} for post-cell {i}.")
                        if i in assembly_pre_root_ids: # i is post, but if it overlaps with pre, we have a different normalization value.
                            potential_excitatory_chains = (num_excitatory - 2) * (len(potential_partners_in)) # maybe middle cell is in set of coregistered that satisfies condition
                        else:
                            potential_excitatory_chains = (num_excitatory - 1) * (len(potential_partners_in)) # maybe middle cell is in set of coregistered that satisfies condition
                        temporary_chain_results = two_chain_results_array[(two_chain_results_array[:, 0] == j) & (two_chain_results_array[:, 2] == i)]
                        # print("Chains...")
                        for chain in tqdm(temporary_chain_results):
                            j, k, i = chain
                            w_jk = summed_size_connectome_df.loc[j, k]
                            w_ki = summed_size_connectome_df.loc[k, i]
                            # Check chain type and accumulate only when both segments have valid weights
                            # print(pt_root_id_to_classification.get(k))
                            if pt_root_id_to_classification.get(k) == 'excitatory':
                                # print(f"Inbound: Processing excitatory chain: {j} -> {k} -> {i} with weights {w_jk}, {w_ki}")
                                sum_weights_excitatory += w_jk * w_ki
                                realized_chains_count_excitatory += 1
                                normalized_realized_chains_count_excitatory += 1 / potential_excitatory_chains
                            elif pt_root_id_to_classification.get(k) == 'inhibitory':
                                # print(f"Inbound: Processing inhibitory chain: {j} -> {k} -> {i} with weights {w_jk}, {w_ki}")
                                sum_weights_inhibitory += w_jk * w_ki
                                realized_chains_count_inhibitory += 1
                                # normalized_realized_chains_count_inhibitory += 1 / potential_inhibitory_chains

                    # Normalize weights by realized chain count only if nonzero
                    if realized_chains_count_excitatory > 0:
                        W_nonzero_chain_in_excitatory[connection_type.__name__][i] = sum_weights_excitatory / realized_chains_count_excitatory
                    if realized_chains_count_inhibitory > 0:
                        W_nonzero_chain_in_inhibitory[connection_type.__name__][i] = sum_weights_inhibitory / realized_chains_count_inhibitory

                    # Calculate binary connectivity only if there are potential partners
                    B_chain_in_excitatory[connection_type.__name__][i] = normalized_realized_chains_count_excitatory
                    B_chain_in_inhibitory[connection_type.__name__][i] = realized_chains_count_inhibitory / potential_inhibitory_chains if potential_inhibitory_chains > 0 else 0 # We can normalize here, since number of potential inhibitory partners is constant over i

        # %%
        # Save all produced sets
        save_folder = 'master_freeze_produced_sets/chain_connections/'
        with open(f"{save_folder}{params_a['run_descriptor']}W_chain_nonzero_pairwise_excitatory.pkl", "wb") as f:
            pickle.dump(W_chain_nonzero_pairwise_excitatory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}W_chain_nonzero_pairwise_inhibitory.pkl", "wb") as f:
            pickle.dump(W_chain_nonzero_pairwise_inhibitory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}B_chain_pairwise_excitatory.pkl", "wb") as f:
            pickle.dump(B_chain_pairwise_excitatory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}B_chain_pairwise_inhibitory.pkl", "wb") as f:
            pickle.dump(B_chain_pairwise_inhibitory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_chain_out_excitatory.pkl", "wb") as f:
            pickle.dump(W_nonzero_chain_out_excitatory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_chain_out_inhibitory.pkl", "wb") as f:
            pickle.dump(W_nonzero_chain_out_inhibitory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_chain_in_excitatory.pkl", "wb") as f:
            pickle.dump(W_nonzero_chain_in_excitatory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}W_nonzero_chain_in_inhibitory.pkl", "wb") as f:
            pickle.dump(W_nonzero_chain_in_inhibitory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}B_chain_out_excitatory.pkl", "wb") as f:
            pickle.dump(B_chain_out_excitatory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}B_chain_out_inhibitory.pkl", "wb") as f:
            pickle.dump(B_chain_out_inhibitory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}B_chain_in_excitatory.pkl", "wb") as f:
            pickle.dump(B_chain_in_excitatory, f)
        with open(f"{save_folder}{params_a['run_descriptor']}B_chain_in_inhibitory.pkl", "wb") as f:
            pickle.dump(B_chain_in_inhibitory, f)

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


        A = {pt_root_id: set(mappings_a['assemblies_by_pt_root_id'][pt_root_id]) for pt_root_id in all_root_ids if 'No A' not in mappings_a['assemblies_by_pt_root_id'][pt_root_id]}

        # Initialize DTC dictionaries to store weights and binary connectivity
        psd_volumes_PTC = {}
        psd_volumes_DTC = {}
        psd_volumes_ITC = {}
        psd_volumes_STC = {}
        psd_volumes_INH = {}
        psd_volumes_PYR = {}

        # Define all potential (pre-cell, post-cell) pairs with excitatory and inhibitory chain types bridging them
        for j in assembly_pre_root_ids:
            for i in assembly_post_root_ids:
                if j != i:  # Exclude autapses
                    W_chain_excitatory[(j, i)] = 0
                    W_chain_inhibitory[(j, i)] = 0
                    B_chain_excitatory[(j, i)] = 0
                    B_chain_inhibitory[(j, i)] = 0

        pt_root_id_to_cell_type = post_cell_table.set_index('pt_root_id')['cell_type'].to_dict()

        count_DTC = 0
        count_PTC = 0
        count_ITC = 0
        count_STC = 0
        count_PYR = 0

        for connection_type in comparison_functions:
            psd_volumes_DTC[connection_type.__name__] = []
            psd_volumes_PTC[connection_type.__name__] = []
            psd_volumes_ITC[connection_type.__name__] = []
            psd_volumes_STC[connection_type.__name__] = []
            psd_volumes_INH[connection_type.__name__] = []
            psd_volumes_PYR[connection_type.__name__] = []
            

        # Per-type unique (pre,post) pairwise summed weights and binary indicators
        _type_keys = ['PTC', 'DTC', 'ITC', 'STC', 'PYR', 'INH']
        W_chain_by_type = {k: {} for k in _type_keys}
        B_chain_by_type = {k: {} for k in _type_keys}
        for connection_type in comparison_functions:
            for cell_type in _type_keys:
                W_chain_by_type[cell_type][connection_type.__name__] = {}
                B_chain_by_type[cell_type][connection_type.__name__] = {}

        category = None
        # Process each row in `two_chain_results_array` to populate weights and binary connectivity
        for _, row in tqdm(enumerate(two_chain_results_array)):
            pre_cell, mid_cell, post_cell = row  # j: pre-cell, k: middle cell, i: post-cell
            if pre_cell in assembly_pre_root_ids and post_cell in all_root_ids:
            # Get synapse weights for connections j -> k and k -> i
                w_jk = summed_size_connectome_df.loc[pre_cell, mid_cell] * (9 * 9 * 45) / (10**9) # cubic micrometers
                w_ki = summed_size_connectome_df.loc[mid_cell, post_cell] * (9 * 9 * 45) / (10**9) # cubic micrometers
            # Determine chain type (excitatory if middle cell is in excitatory set, else inhibitory)
                if pt_root_id_to_cell_type[mid_cell] == 'PTC' or pt_root_id_to_cell_type[mid_cell] == 'ProxTC':
                    count_PTC += 1
                    for connection_type in comparison_functions:
                        if connection_type(pre_cell, post_cell, A):
                            psd_volumes_PTC[connection_type.__name__].append(w_jk * w_ki)
                            psd_volumes_INH[connection_type.__name__].append(w_jk * w_ki)
                            # Sum into unique pairwise aggregator for PTC and lumped INH
                            key = (pre_cell, post_cell)
                            W_chain_by_type['PTC'][connection_type.__name__][key] = W_chain_by_type['PTC'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['PTC'][connection_type.__name__][key] = 1
                            W_chain_by_type['INH'][connection_type.__name__][key] = W_chain_by_type['INH'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['INH'][connection_type.__name__][key] = 1
                if pt_root_id_to_cell_type[mid_cell] == 'ITC' or pt_root_id_to_cell_type[mid_cell] == 'InhTC':
                    count_ITC += 1
                    for connection_type in comparison_functions:
                        if connection_type(pre_cell, post_cell, A):
                            psd_volumes_ITC[connection_type.__name__].append(w_jk * w_ki)
                            psd_volumes_INH[connection_type.__name__].append(w_jk * w_ki)
                            key = (pre_cell, post_cell)
                            W_chain_by_type['ITC'][connection_type.__name__][key] = W_chain_by_type['ITC'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['ITC'][connection_type.__name__][key] = 1
                            W_chain_by_type['INH'][connection_type.__name__][key] = W_chain_by_type['INH'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['INH'][connection_type.__name__][key] = 1
                if pt_root_id_to_cell_type[mid_cell] == 'STC' or pt_root_id_to_cell_type[mid_cell] == 'SparTC':
                    count_STC += 1
                    for connection_type in comparison_functions:
                        if connection_type(pre_cell, post_cell, A):
                            psd_volumes_STC[connection_type.__name__].append(w_jk * w_ki)
                            psd_volumes_INH[connection_type.__name__].append(w_jk * w_ki)
                            key = (pre_cell, post_cell)
                            W_chain_by_type['STC'][connection_type.__name__][key] = W_chain_by_type['STC'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['STC'][connection_type.__name__][key] = 1
                            W_chain_by_type['INH'][connection_type.__name__][key] = W_chain_by_type['INH'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['INH'][connection_type.__name__][key] = 1
                if pt_root_id_to_cell_type[mid_cell] == 'DTC' or pt_root_id_to_cell_type[mid_cell] == 'DistTC':
                    count_DTC += 1
                    for connection_type in comparison_functions:
                        if connection_type(pre_cell, post_cell, A):
                            psd_volumes_DTC[connection_type.__name__].append(w_jk * w_ki)
                            psd_volumes_INH[connection_type.__name__].append(w_jk * w_ki)
                            key = (pre_cell, post_cell)
                            W_chain_by_type['DTC'][connection_type.__name__][key] = W_chain_by_type['DTC'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['DTC'][connection_type.__name__][key] = 1
                            W_chain_by_type['INH'][connection_type.__name__][key] = W_chain_by_type['INH'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['INH'][connection_type.__name__][key] = 1
                elif pt_root_id_to_cell_type[mid_cell][0] == 'L' or pt_root_id_to_cell_type[mid_cell][1] == 'P':
                    count_PYR += 1
                    for connection_type in comparison_functions:
                        if connection_type(pre_cell, post_cell, A):
                            psd_volumes_PYR[connection_type.__name__].append(w_jk * w_ki)
                            key = (pre_cell, post_cell)
                            W_chain_by_type['PYR'][key] = W_chain_by_type['PYR'].get(key, 0) + (w_jk * w_ki)
                            B_chain_by_type['PYR'][key] = 1

            
            # # Updates weights and binary connectivity
            #     W_chain[(pre_cell, post_cell)] += (w_jk * w_ki)s
            #     B_chain[(pre_cell, post_cell)] = 1
        run_descriptor = params_a['run_descriptor']
        save_folder = 'master_freeze_produced_sets/chain_connections/'
        with open(f"{save_folder}{run_descriptor}W_chain_nonzero_pairwise_by_type.pkl", "wb") as f:
            pickle.dump(W_chain_by_type, f)
        with open(f"{save_folder}{run_descriptor}B_chain_pairwise_by_type.pkl", "wb") as f:
            pickle.dump(B_chain_by_type, f)

        print("Middle Cell Type Distribution")
        print('Count PTC:', count_PTC)
        print('Count DTC:', count_DTC)
        print('Count ITC:', count_ITC)
        print('Count STC:', count_STC)
        print('Count PYR:', count_PYR)
        # overwrite file (use 'a' to append)
        out_path = f"./draft_figures/middle_cell_type_counts_SquareChain_{params_a['run_descriptor']}_{output_string}.txt"
        with open(out_path, "w") as f:
            print("Middle Cell Type Distribution", file=f)
            print(f"Count PTC: {count_PTC}", file=f)
            print(f"Count DTC: {count_DTC}", file=f)
            print(f"Count ITC: {count_ITC}", file=f)
            print(f"Count STC: {count_STC}", file=f)
            print(f"Count PYR: {count_PYR}", file=f)

        # optional: also print to console
        print(f"Saved counts to {out_path}")

        for connection_type in comparison_functions:
            psd_volumes_DTC[connection_type.__name__] = np.array(psd_volumes_DTC[connection_type.__name__])
            psd_volumes_PTC[connection_type.__name__] = np.array(psd_volumes_PTC[connection_type.__name__])
            psd_volumes_ITC[connection_type.__name__] = np.array(psd_volumes_ITC[connection_type.__name__])
            psd_volumes_STC[connection_type.__name__] = np.array(psd_volumes_STC[connection_type.__name__])
            psd_volumes_INH[connection_type.__name__] = np.array(psd_volumes_INH[connection_type.__name__])
            psd_volumes_PYR[connection_type.__name__] = np.array(psd_volumes_PYR[connection_type.__name__])

        # Build per-type pairwise nonzero (summed) & binary dictionaries keyed by connection type name
        W_chain_nonzero_pairwise_by_type = {tk: {} for tk in W_chain_by_type.keys()}
        B_chain_pairwise_by_type = {tk: {} for tk in W_chain_by_type.keys()}
        for cond_function in comparison_functions:
            name = cond_function.__name__
            for tk in W_chain_by_type.keys():
                W_chain_nonzero_pairwise_by_type[tk][name] = {}
                B_chain_pairwise_by_type[tk][name] = {}
                for (j, i), w in W_chain_by_type[tk][name].items():
                    if cond_function(j, i, A):
                        B_chain_pairwise_by_type[tk][name][(j, i)] = 1 if w > 0 else 0
                        if w > 0:
                            W_chain_nonzero_pairwise_by_type[tk][name][(j, i)] = w

        # Diagnostic: show unique (pre,post) counts for each middle-type (summed pairs)
        print('\nPer-type unique (pre,post) nonzero connection counts:')
        for tk in sorted(W_chain_nonzero_pairwise_by_type.keys()):
            total_pairs = sum(len(v) for v in W_chain_nonzero_pairwise_by_type[tk].values())
            print(f"  {tk}: total_unique_nonzero_pairs={total_pairs}")

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

    ##################################################
    #####    ### ###   ##     ########################
    ##### ### ## ## ####### ##########################
    ##### ### ## ###  ##### ##########################
    ##### ### ## ##### #### ##########################
    #####    ### ###   #### ##########################
    ##################################################


    # %% [markdown]
    # ## Distance Analysis

    # %%
    from lsmm_data import LSMMData
    import json

    with open(f'FigureCode/Figure4/all_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        loaded_json = json.load(f)
    my_data = LSMMData.LSMMData(loaded_json)
    tables = my_data.data
    params = my_data.params
    dirs = my_data.dirs
    mappings = my_data.mappings
    
    print(tables['structural']['pre_cell'])
    print(tables['structural']['post_cell'])
    print(tables['structural']['synapse'])

    # %%
    '''---------------------When using LSMM-------------------------------------'''
    cell_table = tables['structural']['pre_cell']

    weight_matrix = tables['structural']['summed_size_connectome']

    # %%
    cell_table['pt_position_x_trafo'] = cell_table['pt_position_x_trafo']/1000
    cell_table['pt_position_y_trafo'] = cell_table['pt_position_y_trafo']/1000
    cell_table['pt_position_z_trafo'] = cell_table['pt_position_z_trafo']/1000

    x_pos = np.mean(cell_table['pt_position_x_trafo'])   # Centroid of the graph - cartesian co-ordinates system hence, centroid = mean
    y_pos = np.mean(cell_table['pt_position_y_trafo'])
    z_pos = np.mean(cell_table['pt_position_z_trafo'])
    cent_dist_no_a = []
    cent_dist_a = []


    # %%
    n_assembly, n_nonassembly = 0, 0
    for cell in mappings['assemblies_by_connectome_index']:
        if mappings['assemblies_by_connectome_index'][cell] == ['No A']:
            cent_dist_no_a.append(np.linalg.norm([cell_table['pt_position_x_trafo'][cell] -  x_pos, cell_table['pt_position_y_trafo'][cell] - y_pos], axis = 0))
            n_nonassembly += 1
        else:
            cent_dist_a.append(np.linalg.norm([cell_table['pt_position_x_trafo'][cell] - x_pos, cell_table['pt_position_y_trafo'][cell] -  y_pos], axis = 0))
            n_assembly += 1

    # %%
    # Perform Test
    stat, p_val = stats.ranksums(cent_dist_a, cent_dist_no_a)
    print(f"Rank-Sum Test Statistic: {stat:.4g}, P-value: {p_val:.4g}")

    # Calculate sample sizes
    y_labels = [f"Assembly\n(n={n_assembly})", f"Non-Assembly\n(n={n_nonassembly})"]

    # Build a frame for easier plotting
    data = pd.DataFrame({
        "Values": np.concatenate([cent_dist_a, cent_dist_no_a]),
        "Group": [y_labels[0]] * n_assembly + [y_labels[1]] * n_nonassembly
    })

    # Set up the plot
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    # --- Main plot (original RainCloud) ---
    ax = pt.RainCloud(
        y="Values",
        x="Group",
        data= data,
        palette=[(.4, .6, .8, .5), 'grey'],
        width_viol=0.3,
        alpha=0.8,
        move=0.25,
        point_size=6,
        orient="v",
    )

    # Annotate significance
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

    # Axis title and labels
    # ax.text(0.5, 1.08, "Distance from Centroid of Connectome Network", 
    #         transform=ax.transAxes, ha='center', va='bottom', fontsize=30, fontweight='bold')
    ax.text(0.5, 1.01, f"Rank-Sum P-value: {p_val:.2g}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=30)
    # ax.set_title(r'\textbf{Distance from Centroid of Connectome Network}' + f"\nRank-Sum P-value: {p_val:.2g}", size=30)
    ax.set_xlabel("Assigned Assembly Status", size=26)
    ax.set_ylabel(r"Euclidean Distance from Centroid ($\mu$m)", size=26)
    ax.tick_params(labelsize=26)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(f"draft_figures/a_vs_nona_distance_from_centroid_{params_a['run_descriptor']}.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

    # %% [markdown]
    # ## Tail Analysis

    # %%
    def perform_standard_em(x, K, seed=747, weights_init=None, means_init=None, precisions_init=None):
        """
        Estimate GMM's parameters by using the standard EM algorithm, with k-means clustering initialization.

        Args:
            x (1D numpy array): The observed data.
            K (int): The number of mixture components.
            seed (int): The random seed.
            weights_init (array): Optional initial weights for GMM components.
            means_init (array): Optional initial means for GMM components.
            precisions_init (array): Optional initial precisions for GMM components.

        Returns:
            results (dict): A dictionary containing estimated parameters (weights, means, std deviations).
        """
        # Convert input to DataFrame for compatibility with GaussianMixture
        x = pd.DataFrame(x)
        
        # Fit Gaussian Mixture Model
        model = mixture.GaussianMixture(
            n_components=K,
            random_state=seed,
            covariance_type='diag',
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init
        )
        model.fit(x)

        # Extract parameters and flatten
        weights = model.weights_.tolist()  # Flatten weights
        means = model.means_.flatten().tolist()  # Flatten means
        std = np.sqrt(model.covariances_.flatten()).tolist()  # Flatten std deviations

        results = {
            'pp': weights,
            'mu': means,
            'std': std
        }

        return results

    def gmm_pdf_cdf(x, weights, means, std_devs):
        """
        Compute the PDF and CDF for a Gaussian Mixture Model (GMM).

        Parameters:
            x (np.ndarray): Points at which to evaluate the PDF and CDF.
            means (list): Means of the Gaussian components.
            stds (list): Standard deviations of the Gaussian components.
            weights (list): Weights of the Gaussian components.

        Returns:
            pdf (np.ndarray): PDF values for the GMM.
            cdf (np.ndarray): CDF values for the GMM.
        """
        pdf = np.zeros_like(x)
        cdf = np.zeros_like(x)
        for weight, mean, std_dev in zip(weights, means, std_devs):
            pdf += weight * norm.pdf(x, mean, std_dev)
            cdf += weight * norm.cdf(x, mean, std_dev)
        return pdf, cdf

    def gmm_ppf(q, x, weights, means, std_devs):
        """
        Compute the PPF (percent-point function) for a Gaussian Mixture Model (GMM).

        Parameters:
            q (np.ndarray): Quantiles at which to compute the PPF.
            x (np.ndarray): Points used to evaluate the CDF.
            means (list): Means of the Gaussian components.
            stds (list): Standard deviations of the Gaussian components.
            weights (list): Weights of the Gaussian components.

        Returns:
            ppf (np.ndarray): PPF values for the GMM.
        """
        cdf_vals = gmm_pdf_cdf(x, weights, means, std_devs)[1]
        cdf_func = interp1d(cdf_vals, x, bounds_error=False, fill_value=(x[0], x[-1]))
        return cdf_func(q)

    def hist_with_GMM_fit_plus_qqplot_with_decision_boundary(PSD, means, std_devs, weights,save=True):
        """
        Plots a histogram of the provided data along with a Gaussian Mixture Model (GMM) fit,
        includes the decision boundary, and generates a QQ plot comparing empirical and theoretical quantiles.

        Parameters:
        PSD (np.ndarray): An array of data values (e.g., log10(PSD) values) to be plotted and fitted with the GMM.
        means (np.ndarray): The means of the Gaussian components in the GMM.
        std_devs(np.ndarray): The standard deviations of the Gaussian components in the GMM.
        weights(np.ndarray): The weights of the Gaussian components in the GMM.

        """
        # Calculate the decision boundary (quadratic formula for intersection)
        a = (1 / (2 * std_devs[0]**2)) - (1 / (2 * std_devs[1]**2))
        b = -(means[0] / (std_devs[0]**2)) + (means[1] / (std_devs[1]**2))
        c = ((means[0]**2) / (2 * std_devs[0]**2)) - ((means[1]**2) / (2 * std_devs[1]**2)) - \
            np.log(std_devs[1] / std_devs[0]) - np.log(weights[1] / weights[0])
        roots = np.roots([a, b, c])
        decision_boundary = min(roots)  # Take the smaller root as the decision boundary
        print('Decision Boundary: ', decision_boundary)
        
        # Initialize plots
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 2], wspace=0.3)
        
        # Set up styling
        hist = fig.add_subplot(gs[0])
        qq = fig.add_subplot(gs[1])
        sns.set_theme(style="whitegrid")

        # Get x range
        xmin, xmax = np.min(PSD), np.max(PSD)
        x_range = np.linspace(xmin, xmax, 500)

        # Fit GMM
        pdf, cdf = gmm_pdf_cdf(x_range, weights, means, std_devs)

        # Histogram with GMM fit overlaid
        hist.hist(PSD, bins='rice', density=True, alpha=0.6, color= (.4, .6, .8, .5), label="Empirical Data")
        hist.plot(x_range, pdf, color="gold", alpha=0.5, linewidth = 5, label="GMM Fit")
        hist.axvline(decision_boundary, color="red", linestyle="--", linewidth = 5, label=f"Decision Boundary ({decision_boundary:.2g})")
        hist.set_title("Distribution of Post-Synaptic Densities", fontsize=32)
        hist.set_xlabel(r"$\log_{10}$(Post-Synaptic Density)", fontsize=32)
        hist.set_ylabel("Density", fontsize=32)
        hist.tick_params(axis='both', which='major', labelsize=32)
        hist.legend(prop = { "size": 22 })

        # Generate Quantiles of Empirical, Theoretical Distributions
        empirical_quantiles = np.percentile(PSD, np.linspace(0, 100, len(PSD)))
        theoretical_quantiles = gmm_ppf(np.linspace(0, 1, len(PSD)), x_range, weights, means, std_devs)

        # QQ Plot
        qq.plot(theoretical_quantiles, empirical_quantiles, 'o', color="grey", markersize=5, label='Quantiles')
        qq.plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
                [min(theoretical_quantiles), max(theoretical_quantiles)], 'g--', label='Y=X', linewidth=4, color = (.4, .6, .8, .5))
        qq.set_xlabel('Theoretical Quantiles', fontsize=32)
        qq.set_ylabel('Empirical Quantiles', fontsize=32)
        qq.set_title('Quantile-Quantile Plot', fontsize=32)
        qq.tick_params(axis='both', which='major', labelsize=32)
        qq.legend(prop = { "size": 22 })

        plt.tight_layout()
        if save == True:
            save_figure("GMM_Fit_and_Decision_Boundary")
        #plt.show()
        return decision_boundary

    # %% [markdown]
    # ### Prep Data

    # %%
    # connections into df
    connection_data = []

    for connection_type, connections in W_nonzero_pairwise.items():
        for (pre, post), size in connections.items():
            connection_data.append({
                "pre": pre,
                "post": post,
                "size": size,
                "connection_type": connection_type
            })

    connections_df = pd.DataFrame(connection_data)

    connections_df['size'] = connections_df['size']
    connections_df['log_size'] = np.log10(connections_df['size'])

    # expectation-maximization for GMM fit 
    component_params = perform_standard_em(connections_df['log_size'], K=2, seed=747)

    # GMM plot and fit evaluation
    decision_boundary = hist_with_GMM_fit_plus_qqplot_with_decision_boundary(
        connections_df['log_size'],
        component_params['mu'], 
        component_params['std'],
        component_params['pp']
    )

    # print component parameters
    print("GMM Parameters:")
    print(component_params)

    # get tail boundary and tail df
    tail_minimum = decision_boundary

    connections_df_tail = connections_df[connections_df['log_size'] >= tail_minimum]

    # expected and observed proportions of connection types
    categories = ['Shared', 'Disjoint']
    shared_count = len(connections_df[connections_df['connection_type'] == 'shared'])
    disjoint_count = len(connections_df[connections_df['connection_type'] == 'disjoint'])
    total_count = shared_count + disjoint_count

    prop_shared = shared_count / total_count
    prop_disjoint = disjoint_count / total_count

    tail_shared_count = len(connections_df_tail[connections_df_tail['connection_type'] == 'shared'])
    tail_disjoint_count = len(connections_df_tail[connections_df_tail['connection_type'] == 'disjoint'])

    total_tail_count = tail_shared_count + tail_disjoint_count
    expected_shared = total_tail_count * prop_shared
    expected_disjoint = total_tail_count * prop_disjoint

    observed = [tail_shared_count, tail_disjoint_count]
    expected = [expected_shared, expected_disjoint]
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    observed_props = [tail_shared_count / total_tail_count, tail_disjoint_count / total_tail_count]
    expected_props = [prop_shared, prop_disjoint]

    # frequency table
    frequency_table = tabulate(
        [[cat, obs, f"{exp:.2f}"] for cat, obs, exp in zip(categories, observed, expected)],
        headers=["Connection Type", "Observed Frequency", "Expected Frequency"],
        tablefmt="pretty"
    )

    # Construct the proportion table
    proportion_table = tabulate(
        [[cat, f"{obs:.4g}", f"{exp:.4g}"] for cat, obs, exp in zip(categories, observed_props, expected_props)],
        headers=["Connection Type", "Observed Proportion", "Expected Proportion"],
        tablefmt="pretty"
    )

    # %% [markdown]
    # ### Plot Results

    # %%
    print('Observed vs Expected Frequencies:')
    print(frequency_table, "\n")

    print("Observed vs Expected Proportions:")
    print(proportion_table, "\n")

    print(f"Chi-squared statistic: {chi2_stat:.4g}")
    print(f"P-value: {p_value}")

    # data = pd.DataFrame(zip(categories, observed, expected), columns=['Connection Type', 'Observed Frequency', 'Expected Frequency'])

    # chi_squared_analysis_v2(data, save=True, figure_name=f'ChiSquared_AboveBoundary_Analysis_{params_a["run_descriptor"]}')


    #############################
    #############################
    ## Outlier Tail Analysis ####
    #############################
    #############################


    with open(f'FigureCode/Figure4/all_cells_proofread_connectome_{scan_session_affinity_filestring}.json') as f:
        loaded_json = json.load(f)
    my_data = LSMMData.LSMMData(loaded_json)
    tables = my_data.data
    params = my_data.params
    dirs = my_data.dirs
    mappings = my_data.mappings
    
    print(tables['structural']['pre_cell'])
    print(tables['structural']['post_cell'])
    print(tables['structural']['synapse'])

    # %%
    '''---------------------When using LSMM-------------------------------------'''
    cell_table = tables['structural']['pre_cell']

    weight_matrix = tables['structural']['summed_size_connectome']

    print(data_a['structural']['synapse_count_connectome'].shape)
    psd_volume_array = data_a['structural']['summed_size_connectome']
    print(np.max(psd_volume_array))
    # psd_volume_array
    print(np.min(np.sum(data_a['structural']['synapse_count_connectome'], axis=1)))
    print(np.min(np.sum(data_a['structural']['synapse_count_connectome'], axis=1)))
    print(len(np.where(np.sum(data_a['structural']['synapse_count_connectome'], axis=1) < 20)[0]))

    import numpy as np
    from scipy import stats

    # # Outlier Flow: Log values, Grubbs test, find outliers.
    log_psd_volume_array = np.log(psd_volume_array.nonzero())

    def grubbs_test(array, alpha=0.05): # Not currently using grubbs test
        outliers = []
        for row in array:
            print(np.min(row.nonzero()[0]))
            print(np.max(np.isnan(row)))
            values = np.log10(row.nonzero()[0])
            
            while len(values) > 2:
                mean = np.mean(values)
                std_dev = np.std(values, ddof=1)
                N = len(values)
                
                # Calculate Grubbs' test statistic
                G = np.max(np.abs(values - mean)) / std_dev
                
                # Calculate the critical value
                t_dist = stats.t.ppf(1 - alpha / (2 * N), N - 2)
                G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(t_dist**2 / (N - 2 + t_dist**2))
                
                # Is the maximum G value is greater than the critical value
                if G > G_critical:
                    outlier_index = np.argmax(np.abs(values - mean))
                    outliers.append(values[outlier_index])
                    values = np.delete(values, outlier_index)
                else:
                    break
        return outliers

    def calculate_top_ten(array):
        iqr_values = []
        q75_values = []
        for row in array:
            # Filter out zero values
            filtered_row = row[row != 0]
            if len(filtered_row) > 0:
                # outlier_connections = grubbs_test(filtered_row)
                
                q90, q10 = np.percentile(filtered_row, [90, 10])
                iqr = q90
                iqr_values.append(iqr)
                q75_values.append(q10)
            else:
                iqr_values.append(0)  # Append 0 if the row has no non-zero values
                q75_values.append(0)
        return iqr_values, q75_values

    # The code below is calculating it on summed PSDs, not individual synapses
    # Function to calculate IQR for each row, excluding zero values
    def calculate_iqr_excluding_zeros(array):
        iqr_values = []
        q75_values = []
        for row in array:
            # Filter out zero values
            filtered_row = row[row != 0]
            if len(filtered_row) > 0:
                # outlier_connections = grubbs_test(filtered_row)
                
                q75, q25 = np.percentile(filtered_row, [75, 25])
                iqr = q75 - q25
                iqr_values.append(iqr)
                q75_values.append(q75)
            else:
                iqr_values.append(0)  # Append 0 if the row has no non-zero values
                q75_values.append(0)
        return iqr_values, q75_values

    # Calculate IQR for each row in the synapse_array
    iqr_values, q75_values = calculate_iqr_excluding_zeros(psd_volume_array)
    # iqr_values, q75_values = calculate_top_ten(psd_volume_array)


    # outliers = grubbs_test(psd_volume_array)
    # print(outliers)

    # Print the IQR values
    # print(iqr_values)

    num_outliers = 0
    outlier_connections = {}
    outlier_pre_post_root_id_pairs = []
    print(psd_volume_array.shape)
    for pre_cell in range(psd_volume_array.shape[0]):
        for post_cell in range(psd_volume_array.shape[1]):
            # if psd_volume_array[pre_cell, post_cell] > (iqr_values[pre_cell]) * 1.5 + q75_values[pre_cell]: # for IQR
            if psd_volume_array[pre_cell, post_cell] > iqr_values[pre_cell]:
                outlier_pt_root_id_pair = (data_a['structural']['pre_cell'].iloc[pre_cell].pt_root_id, data_a['structural']['post_cell'].iloc[post_cell].pt_root_id)
                outlier_pre_post_root_id_pairs.append(outlier_pt_root_id_pair)
                outlier_connections[(pre_cell, post_cell)] = psd_volume_array[pre_cell, post_cell]
                num_outliers += 1
    print(num_outliers)
    print(num_outliers / data_a['structural']['synapse_count_connectome'].shape[0])

    synapses_df_null = synapses_df.copy()
    filter_df = pd.DataFrame(outlier_pre_post_root_id_pairs, columns=['pre_pt_root_id', 'post_pt_root_id'])
    synapses_df = synapses_df.merge(filter_df, on=['pre_pt_root_id', 'post_pt_root_id']).reset_index(drop=True)

    # Prep the sets

    # Collect w and s
    w_null = {}
    s_null = {}
    b_null = {}
    print(w_null)
    for pre in pre_root_ids:
        for post in post_root_ids:
            if pre != post:
                w_null[(pre, post)] = 0
                s_null[(pre, post)] = 0
                b_null[(pre, post)] = 0

    for i, row in synapses_df_null.iterrows():
        pre = row['pre_pt_root_id']
        post = row['post_pt_root_id']
        w_null[(pre, post)] += row['size']
        s_null[(pre, post)] += 1
        b_null[(pre, post)] = 1


    # print(w_null)
    print(type(w_null))
    print('~~~~')
    print([(k, v) for k, v in w_null.items() if v > 0])

    # Collect w and s
    w = {}
    s = {}
    b = {}
    for pre in pre_root_ids:
        for post in post_root_ids:
            if pre != post:
                w[(pre, post)] = 0
                s[(pre, post)] = 0
                b[(pre, post)] = 0

    for i, row in synapses_df.iterrows():
        pre = row['pre_pt_root_id']
        post = row['post_pt_root_id']
        w[(pre, post)] += row['size']
        s[(pre, post)] += 1
        b[(pre, post)] = 1

    # Plot Distributions

    # temp = np.array(list(w_null.values())).concatenate(list(w.values()), axis=0)
    # print(temp.shape)
    # Build both numeric-only lists (for histograms) and paired (k,v) lists (for later unpacking)
    null_hypothesis_values = [v for v in w_null.values() if v > 0]
    alternative_values = [v for v in w.values() if v > 0]
    null_hypothesis_set = [(k, v) for k, v in w_null.items() if v > 0]
    alternative_set = [(k, v) for k, v in w.items() if v > 0]

    print(len(null_hypothesis_values), len(alternative_values))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n_bins = 20
    print(w_null)

    _ = ax.hist([null_hypothesis_values, alternative_values], bins=n_bins, histtype='bar', stacked=True, label='Full Distribution')
    # _ = ax.hist(, color='blue', bins=n_bins, alpha=0.5, label='Outliers')
    plt.show()
    plt.savefig(f"draft_figures/Q75_outlier_tail_analysis_hist_{params_a['run_descriptor']}.png", dpi=300, bbox_inches='tight')

    # Calculate the product of the weighted summed values

    backup_pre_root_ids = pre_root_ids
    backup_post_root_ids = post_root_ids

    # W_prime_out_nonzero = {}
    # for c in C:
    #     W_prime_out_nonzero[c.__name__] = {}
    #     for j in pre_root_ids:
    #         post_root_ids = backup_post_root_ids - set([j])
    #         if len([i for i in post_root_ids if c(j, i, A)]) > 0:
    #             # Get chains from j to i
    #             for i in post_root_ids:
    #                 if c(j, i, A):
    #                     post_cell = post_cell_table[post_cell_table['pt_root_id'] == i]['connectome_index'].values[0]
    #                     pre_cell = cell_table[cell_table['pt_root_id'] == j]['connectome_index'].values[0]
    #                     chain = 
    #             two_chain_results_array[np.where(np.logical_and(two_chain_results_array[:,0] == pre_cell, two_chain_results_array[:,-1] == post_cell))]
    #             W_prime_out_nonzero[c.__name__][j] = sum([w[(j, i)] for i in post_root_ids if c(j, i, A)]) / len([i for i in post_root_ids if c(j, i, A)])

    W_out = {}
    for c in comparison_functions:
        W_out[c.__name__] = {}
        for j in pre_root_ids:
            post_root_ids = backup_post_root_ids - set([j])
            if len([i for i in post_root_ids if c(j, i, A)]) > 0:
                W_out[c.__name__][j] = sum([w[(j, i)] for i in post_root_ids if c(j, i, A)]) / len([i for i in post_root_ids if c(j, i, A)])

    W_out_nonzero = {}
    for c in comparison_functions:
        W_out_nonzero[c.__name__] = {}
        for j in pre_root_ids:
            post_root_ids = backup_post_root_ids - set([j])
            if len([i for i in post_root_ids if c(j, i, A) and w[(j, i)] > 0]) > 0:
                W_out_nonzero[c.__name__][j] = sum([w[(j, i)] for i in post_root_ids if c(j, i, A)]) / len([i for i in post_root_ids if c(j, i, A) and w[(j, i)] > 0])

    ## Add W_in nonzero, S_out and S_in nonzero

    W_in = {}
    for c in comparison_functions:
        W_in[c.__name__] = {}
        for i in post_root_ids:
            pre_root_ids = backup_pre_root_ids - set([i])
            if len([j for j in pre_root_ids if c(j, i, A)]) > 0:
                W_in[c.__name__][i] = sum([w[(j, i)] for j in pre_root_ids if c(j, i, A)]) / len([j for j in pre_root_ids if c(j, i, A)])

    S_out = {}
    for c in comparison_functions:
        S_out[c.__name__] = {}
        for j in pre_root_ids:
            post_root_ids = backup_post_root_ids - set([j])
            if len([i for i in post_root_ids if c(j, i, A)]) > 0:
                S_out[c.__name__][j] = sum([s[(j, i)] for i in post_root_ids if c(j, i, A)]) / len([i for i in post_root_ids if c(j, i, A)])

    S_out_nonzero = {}
    for c in comparison_functions:
        S_out_nonzero[c.__name__] = {}
        for j in pre_root_ids:
            post_root_ids = backup_post_root_ids - set([j])
            if len([i for i in post_root_ids if c(j, i, A) and s[(j, i)] > 0]) > 0:
                S_out_nonzero[c.__name__][j] = sum([s[(j, i)] for i in post_root_ids if c(j, i, A)]) / len([i for i in post_root_ids if c(j, i, A) and s[(j, i)] > 0])

    S_in = {}
    for c in comparison_functions:
        S_in[c.__name__] = {}
        for i in post_root_ids:
            pre_root_ids = backup_pre_root_ids - set([i])
            if len([j for j in pre_root_ids if c(j, i, A)]) > 0:
                S_in[c.__name__][i] = sum([s[(j, i)] for j in pre_root_ids if c(j, i, A)]) / len([j for j in pre_root_ids if c(j, i, A)])

    B_out = {}
    for c in comparison_functions:
        B_out[c.__name__] = {}
        for j in pre_root_ids:
            post_root_ids = backup_post_root_ids - set([j])
            if len([i for i in post_root_ids if c(j, i, A)]) > 0:
                B_out[c.__name__][j] = sum([b[(j, i)] for i in post_root_ids if c(j, i, A)]) / len([i for i in post_root_ids if c(j, i, A)])

    B_in = {}
    for c in comparison_functions:
        B_in[c.__name__] = {}
        for i in post_root_ids:
            pre_root_ids = backup_pre_root_ids - set([i])
            if len([j for j in pre_root_ids if c(j, i, A)]) > 0:
                B_in[c.__name__][i] = sum([b[(j, i)] for j in pre_root_ids if c(j, i, A)]) / len([j for j in pre_root_ids if c(j, i, A)])

    # All paired binary shared and disjoint
    B_out_paired = {}
    c1 = comparison_functions[0]
    c2 = comparison_functions[1]
    B_out_paired[c1.__name__] = {}
    B_out_paired[c2.__name__] = {}
    for j in pre_root_ids:
        post_root_ids = backup_post_root_ids - set([j])
        if len([i for i in post_root_ids if c1(j, i, A)]) and len([i for i in post_root_ids if c2(j, i, A)]) > 0:
            B_out_paired[c1.__name__][j] = sum([b[(j, i)] for i in post_root_ids if c1(j, i, A)]) / len([i for i in post_root_ids if c1(j, i, A)])
            B_out_paired[c2.__name__][j] = sum([b[(j, i)] for i in post_root_ids if c2(j, i, A)]) / len([i for i in post_root_ids if c2(j, i, A)])


    # # All paired PSD volume shared and disjoint
    # W_out_paired = {}
    # c1 = C[0]
    # c2 = C[1]
    # W_out_paired[c1.__name__] = {}
    # W_out_paired[c2.__name__] = {}
    # for j in pre_root_ids:
    #     post_root_ids = backup_post_root_ids - set([i])
    #     if len([i for i in post_root_ids if c1(j, i, A)]) and len([i for i in post_root_ids if c2(j, i, A)]) > 0:
    #         c1_w = sum([w[(j, i)] for i in post_root_ids if c1(j, i, A)]) / len([i for i in post_root_ids if c1(j, i, A)])
    #         c2_w = sum([w[(j, i)] for i in post_root_ids if c2(j, i, A)]) / len([i for i in post_root_ids if c2(j, i, A)])
    #         if c1_w > 0 and c2_w > 0:
    #             W_out_paired[c1.__name__][j] = c1_w
    #             W_out_paired[c2.__name__][j] = c2_w

    print(W_out)
    print(W_in)

    pre_root_ids = backup_pre_root_ids
    post_root_ids = backup_post_root_ids

    plt.figure(figsize=(10, 10))
    plt.boxplot([list(W_out_nonzero['shared'].values()), list(W_out_nonzero['disjoint'].values())])
    print(len(W_out_nonzero['shared'].values()))
    print(len(W_out_nonzero['disjoint'].values()))
    plt.savefig(f"draft_figures/Q75_outlier_tail_analysis_boxplot_{params_a['run_descriptor']}.png", dpi=300, bbox_inches='tight')


    shared_values = [v for k, v in alternative_set if comparison_functions[0](k[0], k[1], A) and v > 0]
    disjoint_values = [v for k, v in alternative_set if comparison_functions[1](k[0], k[1], A) and v > 0]


    group_one_count = len(shared_values)
    group_two_count = len(disjoint_values)
    total_connections = group_one_count + group_two_count

    # Observed frequencies
    observed = [group_one_count, group_two_count]

    # Expected frequencies (assuming even distribution)
    expected = [total_connections / 2, total_connections / 2]

    # Perform chi-squared test
    chi2, p_value = chi2_contingency([observed, expected])[:2]


    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n_bins = 20
    print(w_null)

    _ = ax.hist([shared_values,disjoint_values], bins=n_bins, histtype='bar', stacked=True, label='Alternative Set Distribution')
    # _ = ax.hist(, color='blue', bins=n_bins, alpha=0.5, label='Outliers')
    plt.show()
    plt.savefig(f"draft_figures/Q75_outlier_tail_analysis_alternative_hist_{params_a['run_descriptor']}.png", dpi=300, bbox_inches='tight')

    shared_values = [v for k, v in null_hypothesis_set if comparison_functions[0](k[0], k[1], A) and v > 0]
    disjoint_values = [v for k, v in null_hypothesis_set if comparison_functions[1](k[0], k[1], A) and v > 0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n_bins = 20
    print(w_null)

    _ = ax.hist([shared_values,disjoint_values], bins=n_bins, histtype='bar', stacked=True, label='Null Set Distribution')
    # _ = ax.hist(, color='blue', bins=n_bins, alpha=0.5, label='Outliers')
    plt.title('Outlier Tail Analysis: f"Chi-squared: {chi2}, p-value: {p_value}"')
    plt.show()
    plt.savefig(f"draft_figures/Q75_outlier_tail_analysis_null_hist_{params_a['run_descriptor']}.png", dpi=300, bbox_inches='tight')

    # Count the occurrences in each group
    # shared_values = [v for k, v in w.items() if comparison_functions[0](k[0], k[1], A) and v > 0]
    # disjoint_values = [v for k, v in w.items() if comparison_functions[1](k[0], k[1], A) and v > 0]

    print(f"Chi-squared: {chi2}, p-value: {p_value}")
    print(shared_values)
    print(disjoint_values)

    # data_for_chi_squared = pd.DataFrame([shared_values, disjoint_values]).T
    # print(data_for_chi_squared)
    # chi_squared_analysis(data_for_chi_squared, save=True, figure_name=f'Q75_Outlier_Tail_ChiSquared_{params_a["run_descriptor"]}')
    # %%
