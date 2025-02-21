
#%%
import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = ['SALAD', 'SegVLAD 0.2', 'SegVLAD 0.4', 'SegVLAD 0.6', 'SegVLAD 0.8', 'SegVLAD 1.0'][:-1]
storage_amstertime = [0.08, 0.03, 0.05, 0.08, 0.18, 0.98][:-1]
time_amstertime = [2.8, 1.2, 3.1, 4.2, 9.3, 42.3][:-1]
r1_amstertime = [55.4, 54.0, 58.9, 58.1, 58.2, 58.9][:-1]
r5_amstertime = [75.6, 69.2, 76.2, 77.3, 79.5, 79.3][:-1]

# r1_amstertime = np.array(r1_amstertime)
# r1_amstertime = (r1_amstertime - r1_amstertime.min()) / (r1_amstertime.max() - r1_amstertime.min())

storage_pitts30k = [0.62, 0.19, 0.31, 0.51, 1.18, 6.65][:-1]
time_pitts30k = [25.1, 8.0, 12.3, 19.2, 43.2, 251.1][:-1]
r1_pitts30k = [92.6, 91.8, 92.6, 92.8, 92.4, 93.2][:-1]
r5_pitts30k = [96.5, 96.2, 96.7, 96.8, 96.8, 96.8][:-1]

# Create a figure and axis
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Plot for AmsterTime
sc_amstertime = ax[0].scatter(storage_amstertime, time_amstertime, s=[r * 100 for r in r1_amstertime], c='blue', alpha=0.6, label='R@1')
sc_amstertime_2 = ax[0].scatter(storage_amstertime, time_amstertime, s=[r * 10 for r in r5_amstertime], c='red', alpha=0.3, label='R@5')
for i, txt in enumerate(methods):
    ax[0].annotate(txt, (storage_amstertime[i], time_amstertime[i]))

# Plot for Pitts30K
sc_pitts30k = ax[1].scatter(storage_pitts30k, time_pitts30k, s=[r * 10 for r in r1_pitts30k], c='blue', alpha=0.6, label='R@1')
sc_pitts30k_2 = ax[1].scatter(storage_pitts30k, time_pitts30k, s=[r * 10 for r in r5_pitts30k], c='red', alpha=0.3, label='R@5')
for i, txt in enumerate(methods):
    ax[1].annotate(txt, (storage_pitts30k[i], time_pitts30k[i]))

# Titles and labels
ax[0].set_title('AmsterTime Dataset')
ax[0].set_xlabel('Storage (GB)')
ax[0].set_ylabel('Retrieval Time (ms)')
ax[0].legend(loc='upper left')

ax[1].set_title('Pitts30K Dataset')
ax[1].set_xlabel('Storage (GB)')
ax[1].set_ylabel('Retrieval Time (ms)')
ax[1].legend(loc='upper left')

# General settings
fig.suptitle('Storage vs. Retrieval Time with Recall Rates as Point Sizes', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()

# %%


methods = ['SALAD', '$(Ours)\psi$=0.2', 'SegVLAD 0.4', 'SegVLAD 0.6', 'SegVLAD 0.8', 'SegVLAD 1.0'][:-1]
time_amstertime = [2.8, 1.2, 3.1, 4.2, 9.3, 42.3][:-1]
r1_amstertime = [55.4, 54.0, 58.9, 58.1, 58.2, 58.9][:-1]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot recall vs compute time
# plt.scatter(time_amstertime, r1_amstertime, color='blue', s=100, alpha=0.7)
plt.plot(time_amstertime[1:], r1_amstertime[1:], 'bo-', alpha=0.7)
plt.plot(time_amstertime[:1], r1_amstertime[:1], 'ro-', alpha=0.7)

plt.legend()
# Annotate each point with the method name
# for i, method in enumerate(methods):
    # plt.annotate(method, (time_amstertime[i], r1_amstertime[i]), fontsize=10, ha='right')

# Set the title and labels
plt.title('Recall (R@1) vs Compute Time for AmsterTime Dataset')
plt.xlabel('Compute Time (ms)')
plt.ylabel('Recall at Rank 1 (R@1)')

# Display grid
plt.grid(True)

# Show the plot
plt.show()


# %%

import matplotlib.pyplot as plt

# Data from the table for AmsterTime
methods = ['SALAD', 'SegVLAD 0.2', 'SegVLAD 0.4', 'SegVLAD 0.6', 'SegVLAD 0.8']
time_amstertime = [2.8, 1.2, 3.1, 4.2, 9.3]
r1_amstertime = [55.4, 54.0, 58.9, 58.1, 58.2]
storage_amstertime = [0.08, 0.03, 0.05, 0.08, 0.18]

# Create the plot
plt.figure(figsize=(5, 3))

# Plot recall vs compute time
plt.plot(time_amstertime[1:], r1_amstertime[1:], 'bo-', alpha=0.7, label='SegVLAD (Ours)')
plt.plot(time_amstertime[:1], r1_amstertime[:1], 'ro-', alpha=0.7, label='SALAD')

# Annotate each point with the storage value
for i in range(len(methods)):
    ha = 'right' if i == 1 else 'left'  # Align left for the first point to avoid it going out of bounds
    plt.annotate(f'{storage_amstertime[i]} GB', (time_amstertime[i], r1_amstertime[i]), fontsize=10, ha='center', va='bottom')

# Set the title and labels
# plt.title('R@1 vs Compute Time')
plt.xlabel('Compute Time (ms)')
plt.ylabel('R@1')

# Display grid
plt.grid(True)

# Add legend
plt.legend()
plt.savefig('r1_vs_time_amstertime.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# Show the plot
plt.show()


# %%
