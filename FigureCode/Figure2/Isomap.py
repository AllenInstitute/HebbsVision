# %% [markdown]
# ## Figure 1 Code to Produce Figures
# 
# This figure will focus on the presentation of Discrete vs Continuous.

# %%
# importing packages
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
from scipy import stats
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import statsmodels.api as sm
plt.rcParams.update({'font.size': 13})
plt.rcParams["figure.figsize"] = (10,10)

# %%
###################### Load in Calcium Fluoresence Data ######################
session13_fluorescence = np.load("../Data/Session13/sessionM409828_13_CALCIUM-FLUORESCENCE.npy")

### Create Pandas data frame of the data
ns = [f"n{i}" for i in range(session13_fluorescence[0,:].shape[0])]
session13_fluorescence_df = pd.DataFrame(session13_fluorescence, columns = ns)
session13_fluorescence_df

# %%
### Produce Correlation Matrix for Session 13 Data
corr_matrix = pd.DataFrame(np.corrcoef(session13_fluorescence_df.values, rowvar=False), columns=session13_fluorescence_df.columns)
np.fill_diagonal(corr_matrix.values, 0) # Set All Self Correlations to 0
plt.matshow(corr_matrix)
plt.title('Correlations for All Neurons in Session 13 of V1DD')
plt.clim(-0.1, 0.6)
plt.colorbar(shrink=0.8, aspect=20)
plt.savefig('Correlation_Matrix.png', dpi = 300)
plt.show()

# %% [markdown]
# ### Hierarchical Clustering
# 
# To first compare discrete vs. continuous, we sought to exemplify clustering of neuronal data using Hierarchical Clustering
# 
# Notes From Before that are worth it to check out:
# - Might be important to apply the Cophonetic Correlation to compare this dendrogram across different linkages. See if 'ward' is actually the best representation. Hard to tell with neuronal data
# - As for the distances, right now we're using something simple for the dissimilarity being the inverse of the correlation. however, it might be helpful afterwards to check how this clustering does when we use the geodesic distance of the Isomap for those dissimilarities

# %%
### Run Hierarchical Clustering Algorithm on data
plt.figure(figsize=(10,7))
corr = pd.DataFrame(np.corrcoef(session13_fluorescence_df.values, rowvar=False), columns= session13_fluorescence_df.columns)
dissimilarity = 1 - corr
# ‘complete’ linkage uses the maximum distances between all observations of the two sets
Z = abs(linkage(dissimilarity, 'ward'))

dendrogram(Z, labels= session13_fluorescence_df.columns, orientation='top', leaf_rotation=90);

# %%
original_dists = dissimilarity.values  # Original Disimilarity between Neurons
cophenetic_dists = squareform(cophenet(Z)) # Cophenetic Distances Between Neurons
corr_coef = np.corrcoef(original_dists, cophenetic_dists)[0,1]
corr_coef

# %%
# Clusterize the data
threshold = 30 # Threshold of 13 produces 15 distinct clusters
labels_corr = fcluster(Z, threshold, criterion='distance') # Show the unique clusters: np.unique(labels_corr)

labels_corr_order = np.argsort(labels_corr) # Keep the indices to sort labels

# Build a new dataframe based on the sorted columns
clustered_corr = [session13_fluorescence[:,i] for i in labels_corr_order]
clustered_corr = pd.DataFrame(np.array(clustered_corr).T)
        
clustered_corr_matrix = pd.DataFrame(np.corrcoef(clustered_corr.values, rowvar=False), columns= clustered_corr.columns)
np.fill_diagonal(clustered_corr_matrix.values, 0) # Set All Self Correlations to 0     
plt.matshow(clustered_corr_matrix)
plt.title("Hierarchical Clustering on Corr of Scan1.3 in V1DD")
plt.clim(-0.1, 0.6)
plt.colorbar(shrink=0.8, aspect=20)
plt.savefig('Hierarchical_Clustering.png', dpi = 300)

# %%
labels_corr

# %% [markdown]
# ### Continuing with Isomap

# %%
# open a file, where you stored the pickled data
with open("manifold_2D_on_corr_V1DD_Session13.pickle", 'rb') as f:
    # dump information to that file
    manifold_2D_on_corr_V1DD_Session_13 = pickle.load(f)

manifold_2D_on_corr_V1DD_Session_13

# %%
%matplotlib inline

fig = plt.figure()
fig.set_size_inches(11,9)
ax = fig.add_subplot(111)
ax.set_title('2D Components from Isomap of Scan 1.3 Correlations in V1DD')
ax.set_xlabel('Component: 1')
ax.set_ylabel('Component: 2')

# Show 2D components plot
ax.scatter(manifold_2D_on_corr_V1DD_Session_13['Component 1'], manifold_2D_on_corr_V1DD_Session_13['Component 2'], marker='o',alpha=0.7)
#fig.colorbar(mapper, orientation='vertical')

m, b = np.polyfit(manifold_2D_on_corr_V1DD_Session_13['Component 1'], manifold_2D_on_corr_V1DD_Session_13['Component 2'], 1)
plt.plot(manifold_2D_on_corr_V1DD_Session_13['Component 1'], m * manifold_2D_on_corr_V1DD_Session_13['Component 1'] + b)
plt.savefig("Isomap_Plot.png", dpi = 300)
plt.show()

# %%
# open a file, where you stored the pickled data
with open("plotted_corr_fluorescence_vs_isomap_V1DD_Session13.pickle", 'rb') as f:
    plotted_corr_fluorescence_vs_isomap_V1DD_Session13 = pickle.load(f)

plotted_corr_fluorescence_vs_isomap_V1DD_Session13

# %%
%matplotlib inline

fig = plt.figure()
fig.set_size_inches(9,9)
ax = fig.add_subplot(111)
ax.set_title('Plotting Fluorescence Correlation vs Isomap Distance on Scan 1.3')
ax.set_xlabel('Distance (Isomap)')
ax.set_ylabel('Correlation (between neurons)')
matplotlib.rcParams['agg.path.chunksize'] = 100000

# Show 2D components plot
step = plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,0].shape[0] // 3000000 # sample 3000000 sampled points
sample = np.arange(0, plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,0].shape[0], step)
curr_sample_X = plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,0][sample]
curr_sample_Y = plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,1][sample]
ax.scatter(curr_sample_X, curr_sample_Y , marker='o',alpha=0.05, s = 0.5)

# Fit a linear regression
# Fit a linear regression
X_input = sm.add_constant(plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,0])
res = sm.OLS(plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,1], X_input).fit()
ax.plot(plotted_corr_fluorescence_vs_isomap_V1DD_Session13[:,0], res.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')
plt.savefig("Corr_vs_Isomap", dpi = 300)
plt.show()

# %% [markdown]
# ### Plot Correlation Matrix Using 1-D Isomap Distance

# %%
# Extract the first component from the 2D Isomap
first_component = manifold_2D_on_corr_V1DD_Session_13['Component 1']

# Get the sorted indices based on the first component
sorted_1D_Isomap_indices = np.argsort(first_component)

# Reorder the correlation matrix
corr_matrix = np.corrcoef(session13_fluorescence_df.values, rowvar=False)
reordered_corr_matrix = pd.DataFrame(corr_matrix[sorted_1D_Isomap_indices][:, sorted_1D_Isomap_indices], columns=session13_fluorescence_df.columns)
np.fill_diagonal(reordered_corr_matrix.values, 0) # Set All Self Correlations to 0
plt.matshow(reordered_corr_matrix)
plt.title('Corr of Scan 1.3 in V1DD: Isomap Ordering')
plt.clim(-0.1, 0.6)
plt.colorbar(shrink=0.8, aspect=20)
plt.savefig('Correlation_Matrix_1D_Isomap_Ordered.png', dpi = 300)
plt.show()

# %% [markdown]
# ### Plot Hierarchical Clustering Using 1-D Isomap Distance

# %%
first_component

# %%
np.unique(labels_corr)

# %%
# Create a new array to store the reordered clustered matrix
if not isinstance(clustered_corr_matrix, np.ndarray):
    clustered_corr_matrix = clustered_corr_matrix.to_numpy()
reordered_clustered_matrix = np.zeros_like(clustered_corr_matrix)

# Get unique cluster labels (assumed to be from the clustering output)
unique_clusters = np.unique(labels_corr)

# Initialize an empty list to collect the final sorted indices
final_sorted_indices = []

# Iterate over each cluster and reorder based on the first component of Isomap
for cluster in unique_clusters:
    # Get the indices of neurons in the current cluster, sort by first Isomap Component
    cluster_indices = np.where(labels_corr == cluster)[0]
    cluster_sorted_indices = cluster_indices[np.argsort(first_component[cluster_indices])]

    # Append the sorted indices to the final list
    final_sorted_indices.extend(cluster_sorted_indices)

# Reorder the clustered correlation matrix by the sorted indices
reorder_clustered_corr = [session13_fluorescence[:,i] for i in final_sorted_indices]
reorder_clustered_corr = pd.DataFrame(np.array(reorder_clustered_corr).T)
reordered_clustered_matrix = pd.DataFrame(np.corrcoef(reorder_clustered_corr.values, rowvar=False), columns= reorder_clustered_corr.columns)

np.fill_diagonal(reordered_clustered_matrix.values, 0) # Set All Self Correlations to 0     
plt.matshow(reordered_clustered_matrix)
plt.title("Hierarchical Clustering on Corr of Scan1.3 in V1DD: Isomap Ordered")
plt.clim(-0.1, 0.6)
plt.colorbar(shrink=0.8, aspect=20)
plt.savefig('Hierarchical_Clustering_1D_Isomap_Ordered.png', dpi = 300)



