import numpy as np
import matplotlib.pyplot as plt
import os

from local_config import save_dir

fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for row, n_size in enumerate([40, 60, 120]):
    images = np.load(f'{save_dir}/psvrt_sd_m4_n{n_size}/train_images.npy')
    labels = np.load(f'{save_dir}/psvrt_sd_m4_n{n_size}/train_labels.npy')

    print(images.shape, labels.shape)
    
    # Get 3 "same" and 3 "different" examples
    # same_idx = np.where(labels == 1)[0][:3]
    # diff_idx = np.where(labels == 0)[0][:3]

    same_idx = np.random.choice(np.where(labels == 1)[0], 3, replace=False)
    diff_idx = np.random.choice(np.where(labels == 0)[0], 3, replace=False)
    
    for col, idx in enumerate(list(same_idx) + list(diff_idx)):
        axes[row, col].imshow(images[idx, :, :, 0], cmap='gray', 
                               interpolation='nearest', vmin=0, vmax=1)
        label_str = 'Same' if labels[idx] == 1 else 'Different'
        axes[row, col].set_title(label_str, fontsize=10)
        axes[row, col].axis('off')
    
    axes[row, 0].set_ylabel(f'n={n_size}', fontsize=14, rotation=0, 
                             labelpad=50, va='center')

plt.suptitle('PSVRT SD samples (m=4, k=2)\nLeft 3: Same | Right 3: Different', 
             fontsize=14)
plt.tight_layout()
plt.savefig('psvrt_random_samples.png', dpi=150, bbox_inches='tight')
plt.show()