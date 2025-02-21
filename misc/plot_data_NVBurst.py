# %%
import numpy as np
import matplotlib.pyplot as plt

# read from a csv, columns are methodName, num_clusters, prePoolPCA dims, and Recall
data = np.genfromtxt('../out/data.csv', delimiter=',', skip_header=1)

# plot recall vs num_clusters*prePoolPCA dims for two methods, where method 1 and 2 are in alternate rows
plt.scatter(data[::2,1]*data[::2,2], data[::2,3], label='NV')
plt.scatter(data[1::2,1]*data[1::2,2], data[1::2,3], label='NV++')
# mark the points in terms of num clusters and prePoolPCA dims
for i in range(data.shape[0]//2):
    plt.text(data[i*2,1]*data[i*2,2], data[i*2,3], f"{int(data[i*2,1])}x{int(data[i*2,2])}")
    plt.text(data[i*2+1,1]*data[i*2+1,2], data[i*2+1,3], f"{int(data[i*2+1,1])}x{int(data[i*2+1,2])}")
plt.xlabel('Output Dims')
plt.ylabel('Recall')
plt.legend()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

# Set the aesthetics for the plots
sns.set_theme(style="whitegrid")

# Load the data
file_path = '../out/data.csv'
data = np.genfromtxt(file_path, delimiter=',', dtype=None, names=True, encoding='utf-8')

# Data cleaning and manipulation

# Calculate the Output Feature Dimension
clusters = data['Clusters'].astype(float)  # Ensure numeric
pca = data['PCA'].astype(float)  # Ensure numeric
output_feature_dimension = clusters * pca

# Filter out rows where 'Msls-val' is NaN
msls_val = data['Mslsval'].astype(float)  # Assuming 'Msls-val' in your CSV is 'Msls_val'
valid_indices = ~np.isnan(msls_val)
data_clean = data[valid_indices]
output_feature_dimension_clean = output_feature_dimension[valid_indices]
msls_val_clean = msls_val[valid_indices]

# Categorize each entry
def correct_categorize_method(system_names):
    categories = []
    for name in system_names:
        if "_AB_" in name:
            categories.append("NV_AB")
        elif "_Rand" in name:
            categories.append("NV_Rand")
        elif "_Mlp" in name:
            categories.append("NV_MLP")
        else:
            categories.append("NV")
    return np.array(categories)

corrected_categories = correct_categorize_method(data_clean['System'])

# Define marker styles and sizes for each category
markers = {"NV": "o", "NV_AB": "*", "NV_Rand": "^", "NV_MLP": "s"}
sizes_dict = {"NV": 100, "NV_AB": 400, "NV_Rand": 100, "NV_MLP": 50}
sizes = np.array([sizes_dict[cat] for cat in corrected_categories])

# Set colors
colors = plt.cm.tab10(np.linspace(0, 1, len(set(corrected_categories))))
colors = ['b', 'r', 'g', 'c']  # Manually set colors for each category
color_dict = {cat: color for cat, color in zip(set(corrected_categories), colors)}

# Plot
plt.figure(figsize=(14, 10))
for category in set(corrected_categories):
    indices = corrected_categories == category
    plt.scatter(output_feature_dimension_clean[indices], msls_val_clean[indices], label=category,
                marker=markers[category], s=sizes[indices], color=color_dict[category])
    # write the cluster dim and pca dim value on the points using color_dict
    for i in np.where(indices)[0]:
        plt.text(output_feature_dimension_clean[i], msls_val_clean[i], f"{int(clusters[i])}x{int(pca[i])}", color=color_dict[category])

plt.ylim(msls_val_clean.min() - 0.05, msls_val_clean.max() + 0.05)
plt.title('Recall vs. Output Feature Dimension Across All Methods')
plt.xlabel('Output Feature Dimension')
plt.ylabel('Recall')
# legend inside
plt.legend(title='Method Category', loc='lower right', borderaxespad=0)
plt.tight_layout()
plt.savefig('../out/recall_vs_output_dims_texted.png')
plt.show()

# %%
