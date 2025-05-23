## Setup Part 1: Clone the Repository

Execute `git clone https://github.com/AllenInstitute/HebbsVision`, or otherwise clone https://github.com/AllenInstitute/HebbsVision into a directory of your choice.  If your cloning method supports git large files, this should take a while and download <X> gigs of files.

## Setup Part 2: Environment Setup
First, ensure that anaconda (conda) is installed on your system.
Then, to create the environment and install the conda and pip packages, at a command prompt execute `conda env create -n HebbsVision -f conda_files.yaml`
In the case of failure installing NeuroAnalysisTools, due to the lack of `requirements.txt`, you must create the environment first then download NeuroAnalysisTools from `https://github.com/zhuangjun1981/NeuroAnalysisTools`.  Once it exists in a local directory, you can install it locally from pip with `pip install -e ./<path_to_NeuroAnalysisTools>.`  The same workaround can be employed with V1DD-Physiology if a similar error occurs.

The LSMM library must be installed via a local wheel file at the moment, and the .whl is included in the root of the repository.  Install it with `pip install lsmm_data-0.1.2-py2.py3-none-any.whl`

## Assembly Extraction and Hyperparameter Tuning (Optional):
Note that this step can take some time, so for those simply interested in reproducing our analysis of the extracted assemblies, the HebbsVision repository includes all the hyperparameter files that were generated using Scan 1-4 data following the instructions in Molter et al. 2024: "Similarity Graph Clustering for Neural Assembly Detection" p. 175, and we also include the file generated from Scan 1-3 (the scan on which the remaining analysis is focused) using the selected best hyperparameters.

## Figure 1

To recreate panel B of figure 1 (the only one displaying data from our analysis), open `AssemblyPaperFigures/FigureCode/Figure1/Figure1.ipynb` and execute all the cells in order.  It will create files in the `AssemblyPaperFigures/FigureCode/Figure1/` directory.  
* Figure 1B uses `AssemblyPaperFigures/FigureCode/Figure1/3D_Scan_Images.png`, `AssemblyPaperFigures/FigureCode/Figure1/Example_Activity_Traces.png`, and `AssemblyPaperFigures/FigureCode/Figure1/raster_plot_red.png`.

## Figure 2

To recreate panels B, C, D, and the data for panel E of figure 2, open `AssemblyPaperFigures/FigureCode/Figure2/Figure2.ipynb` and execute all the cells in order.  
It will create files in the `AssemblyPaperFigures/FigureCode/Figure2/` directory.  
* Figure 2B uses `AssemblyPaperFigures/FigureCode/Figure2/Assemblies_Intersection_Upset_Plot.png`.  
* Figure 2C uses `AssemblyPaperFigures/FigureCode/Figure2/Assemblies_Plotted_In_Recording_Space_Same_Plot.png`.  
* Figure 2D uses `AssemblyPaperFigures/FigureCode/Figure2/Spatial_Distribution_of_Cells_by_Assembly.png`.  
* The data for Figure 2E are printed in the .ipynb notebook, following the cell where `compare_assemblies_spatial_distribution` is called.

## Figure 3

To recreate all panels in Figure 3, open `AssemblyPaperFigures/FigureCode/Figure3/Figure3.ipynb` and execute all the cells in order.  It will create files in the `AssemblyPaperFigures/FigureCode/Figure3/` directory.

* Figure 3A uses `AssemblyPaperFigures/FigureCode/Figure3/correlations_assemblies_vs_random_ensembles_raincould_plot.png`
* Figure 3B uses `AssemblyPaperFigures/FigureCode/Figure3/sparsity_with_Gini_coefficient_by_assembly_and_random_ensembles.png`
* Figure 3C uses `AssemblyPaperFigures/FigureCode/Figure3/oracle_scores_dff_all_sets_raincloud.png`
* Figure 3D uses `AssemblyPaperFigures/FigureCode/Figure3/assembly_balanced_clip_id_percentage_decoder_MLPClassifier.png`
* Figure 3E uses `AssemblyPaperFigures/FigureCode/Figure3/random_ensemble_balanced_clip_id_percentage_decoder_MLPClassifier.png`
* Figure 3F uses `AssemblyPaperFigures/FigureCode/Figure3/Trigger_Frame_Assembly_4_real_assemblies_mean_trigger_frame.png`, `AssemblyPaperFigures/FigureCode/Figure3/Trigger_Frame_Assembly_4_random_ensembles_mean_trigger_frame.png`, and `AssemblyPaperFigures/FigureCode/Figure3/Trigger_Frame_Assembly_4_difference_squared_in_mean_trigger_frame_assembly_minus_random.png`

## Figure 4

To recreate all panels in Figure 4, open `AssemblyPaperFigures/FigureCode/Figure4/Figure4_Master.ipynb` and execute all the cells in order.  It will create files in the `AssemblyPaperFigures/FigureCode/Figure4/` directory.

* Figure 4A uses `AssemblyPaperFigures/FigureCode/Figure4/connectivity_plot_final.png`
* Figure 4B uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Betweenness_Centrality_All.png`
* Figure 4C uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Outdegree_Centrality_All.png`
* Figure 4D uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/A_No_A_Prob_Conn_by_Conn_Type_v2.png` and `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Prob_Conn_by_Conn_Type_v2.png`
* Figure 4E uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Nonzero_PSD_by_Conn_with_side_plot.png`
* Figure 4F uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/A_No_A_Prob_Conn_by_Conn_Type_E_Chains_v2.png` and `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Prob_Conn_by_Conn_Type_E_Chains_v2.png`
* Figure 4G uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Nonzero_PSD_by_Conn_E_Chain_with_side_plot.png`
* Figure 4H uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/A_No_A_Prob_Conn_by_Conn_Type_I_Chains_v2.png` and `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Prob_Conn_by_Conn_Type_I_Chains_v2.png`
* Figure 4I uses `AssemblyPaperFigures/FigureCode/Figure4/draft_figures/Nonzero_PSD_by_Conn_I_Chain_with_side_plot.png`

Additional tests referenced in the statistical table are printed out throughout the .ipynb files above as they are run.   

