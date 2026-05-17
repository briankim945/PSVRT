from instances.psvrt_new import n_sd_k
import numpy as np
import os

def generate_psvrt_split(n_size, m_size=4, num_train=8000, 
                         num_val=1000, num_test=1000):
    total = num_train + num_val + num_test
    batch_size = 100
    
    generator = n_sd_k([n_size, n_size, 1], batch_size=batch_size)
    generator.initialize_vars(
        item_size=[m_size, m_size],
        box_extent=[n_size, n_size],
        n=1, k=2,
        num_item_pixel_values=1
    )
    
    all_images, all_labels = [], []
    num_batches = total // batch_size
    for _ in range(num_batches):
        imgs, targets, _, _ = generator.single_batch()
        labels = targets[:, 0, 0, 1]  # 1 = same, 0 = different
        all_images.append(imgs)
        all_labels.append(labels)
    
    images = np.concatenate(all_images)
    labels = np.concatenate(all_labels)
    images = (images + 1) / 2.0  # normalize {-1,1} to {0,1}
    
    # Shuffle before splitting
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    splits = {
        'train': (images[:num_train], labels[:num_train]),
        'val': (images[num_train:num_train+num_val], labels[num_train:num_train+num_val]),
        'test': (images[num_train+num_val:], labels[num_train+num_val:]),
    }
    
    out_dir = f'/users/bkim53/scratch/psvrt_sd_m{m_size}_n{n_size}'
    os.makedirs(out_dir, exist_ok=True)
    
    for split_name, (imgs, lbls) in splits.items():
        np.save(os.path.join(out_dir, f'{split_name}_images.npy'), imgs)
        np.save(os.path.join(out_dir, f'{split_name}_labels.npy'), lbls)
        print(f"  {split_name}: {len(imgs)} samples, "
              f"class balance: {lbls.mean():.3f}")

# for n_size in [40, 60, 120]:
for n_size in [30, 90, 150, 180]:
    print(f"\nGenerating SD m=4, n={n_size}")
    generate_psvrt_split(n_size)