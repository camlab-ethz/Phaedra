import torch
from torch.utils.data import DataLoader, Subset

def bootstrap_mean_std_dataloader(dataloader, num_bootstrap_samples=100):
    """
    Estimate mean and std using bootstrapping from a PyTorch DataLoader.

    Parameters:
        dataloader: torch.utils.data.DataLoader
            Your training dataloader (can include multiple datasets).
        num_bootstrap_samples: int
            Number of bootstrap samples to draw.
        device: str
            The device to use for tensor operations ('cpu' or 'cuda').

    Returns:
        (mean, std): Tuple of torch.Tensors
            Estimated mean and std across all data in the dataloader.
    """
    all_samples = []

    print("Collecting data...")
    for batch_idx, batch in enumerate(dataloader):
        x = batch["field_variables_in"]

        if isinstance(x, torch.Tensor):
            all_samples.append(x.view(x.size(0), -1))  # flatten per-sample

        if batch_idx > num_bootstrap_samples:
            break

    all_data = torch.cat(all_samples, dim=0)

    print(f"Collected {all_data.shape[0]} samples with {all_data.shape[1]} features each.")
    print(f"all_data {all_data.mean()} +/- {all_data.std()}, [{all_data.min()}, {all_data.max()}]")
    means = []
    stds = []

    print(f"Bootstrapping with {num_bootstrap_samples} samples...")
    for i in range(num_bootstrap_samples):
        indices = torch.randint(0, all_data.shape[0], (all_data.shape[0],))
        sample = all_data[indices]
        means.append(sample.mean())
        stds.append(sample.std())

    
    mean = torch.stack(means).mean()
    std = torch.stack(stds).mean()

    return mean, std

def bootstrap_mean_std_dataset(dataset, field_name="field_variables_in", num_samples=500, num_bootstraps=5):
    """
    Compute mean and std via bootstrapping over a dataset.
    """
    all_means = []
    all_stds = []

    sample_indices = torch.randint(0,len(dataset),(num_samples,))
    samples = [dataset[i][field_name].float() for i in sample_indices]  # Convert to float32
    stacked = torch.stack(samples)  # Shape: [num_samples, C, H, W] or similar

    mean = stacked.mean(dim=(0, 2, 3))  # Mean over samples and spatial dims
    std = stacked.std(dim=(0, 2, 3))    # Same for std

    all_means.append(mean)
    all_stds.append(std)

    # Final mean/std as average over bootstraps
    mean = torch.stack(all_means).mean(dim=0)
    std = torch.stack(all_stds).mean(dim=0)
    
    return mean, std

def calculate_mean_std(dataset, field_name="field_variables_in", num_samples=2500, num_workers=4, batch_size=32):
    """
    Calculates mean and std of a dataset using parallel data loading.
    
    Args:
        dataset: The PyTorch dataset.
        field_name: The key to access the tensor in the dataset item.
        num_samples: Total number of random samples to use for estimation.
        num_workers: Number of parallel processes to load data.
    """
    print(f"Calculating stats using {num_samples} samples and {num_workers} workers...")
    
    # 1. Randomly select indices from the entire dataset
    indices = torch.randint(0, len(dataset), (num_samples,)).tolist()
    
    # 2. Create a Subset and wrap it in a DataLoader for parallel loading
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        drop_last=False
    )

    # 3. Iterate and accumulate
    all_means = []
    all_vars = [] # We accumulate variance to be mathematically cleaner
    
    # We need to know the channel dimension size to initialize accumulators
    # but we can also just accumulate lists if RAM permits (usually fine for 2-3k samples)
    accumulated_data = []

    for batch in loader:
        # Assuming your dataset returns a dict
        data = batch[field_name].float() 
        accumulated_data.append(data)

    # 4. Concatenate all batches: Shape [num_samples, C, H, W]
    full_sample = torch.cat(accumulated_data, dim=0)
    
    # 5. Compute global statistics
    # Reduce over Batch(0), Height(2), Width(3). Keep Channel(1).
    # We dynamically find dims to support 1D/2D/3D data, assuming Channel is dim 1.
    reduce_dims = [0] + list(range(2, full_sample.ndim))
    
    mean = full_sample.mean(dim=reduce_dims)
    std = full_sample.std(dim=reduce_dims)
    
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std:  {std}")
    
    return mean, std

def normalize_dataset(dataset, field_name="field_variables_in", new_field_name="field_variables_in_normalized", mean=None, std=None):
    """
    Applies normalization and augments dataset with normalized field and stats.
    """
    eps = 1e-6  # Avoid division by zero

    for i in range(len(dataset)):
        original = dataset[i][field_name].float()
        normed = (original - mean[:, None, None]) / (std[:, None, None] + eps)

        dataset[i][new_field_name] = normed
        dataset[i][field_name] = original
        dataset[i]["mean"] = mean
        dataset[i]["std"] = std

def normalize_tensor(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Normalize data using per-sample mean and std.
    
    Args:
        data: Tensor of shape (B, C, H, W)
        mean: Tensor of shape (B, C) or (B, C, 1, 1)
        std: Tensor of shape (B, C) or (B, C, 1, 1)
    Returns:
        Normalized tensor
    """
    if mean.ndim == 2:
        mean = mean[:, :, None, None]
    if std.ndim == 2:
        std = std[:, :, None, None]
    return (data - mean) / (std + eps)

def denormalize_tensor(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize data using per-sample mean and std.

    Args:
        data: Tensor of shape (B, C, H, W)
        mean: Tensor of shape (B, C) or (B, C, 1, 1)
        std: Tensor of shape (B, C) or (B, C, 1, 1)
    Returns:
        Denormalized tensor
    """
    if mean.ndim == 2:
        mean = mean[:, :, None, None]
    if std.ndim == 2:
        std = std[:, :, None, None]
    return data * std + mean

