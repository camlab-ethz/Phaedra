
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-config'
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np

def save_contourf_comparison(inputs, reconstructions, save_path, cmap="turbo"):
    """
    Save contourf plots of the first 10 inputs, reconstructions, and their absolute differences.
    Each row has its own consistent colorbar scaling for the first two columns.
    The third column (absolute difference) has its own scale and colormap per row.

    Args:
        inputs (torch.Tensor or np.ndarray): Input tensor of shape (B, C, H, W) or (B, H, W)
        reconstructions (torch.Tensor or np.ndarray): Reconstructed tensor of same shape
        save_path (str): Path to save the generated plot image
    """
    # Convert to numpy if necessary
    if hasattr(inputs, 'detach'):
        inputs = inputs.detach().cpu().numpy()
    if hasattr(reconstructions, 'detach'):
        reconstructions = reconstructions.detach().cpu().numpy()

    # Handle shape: (B, C, H, W) or (B, H, W)
    if inputs.ndim == reconstructions.ndim == 4:
        for channel in range(inputs.shape[1]):
            save_contourf_comparison(
                inputs[:, channel],
                reconstructions[:, channel],
                save_path.replace('.png', f'_channel{channel}.png'),
                cmap=cmap
            )
        return
    else:
        if inputs.ndim == 4:
            inputs = inputs[:, 0]
        if reconstructions.ndim == 4:
            reconstructions = reconstructions[:, 0]

        diff = (inputs - reconstructions)
        num_samples = min(10, inputs.shape[0])
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2.5 * num_samples))

        if num_samples == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(num_samples):
            row_input = inputs[i]
            row_recon = reconstructions[i]
            row_diff = diff[i]

            # Per-row consistent scale for input and recon
            row_min = min(row_input.min(), row_recon.min())
            row_max = max(row_input.max(), row_recon.max())

            # Plot input (use imshow with nearest interpolation to avoid smoothing/interpolation)
            ax_input = axes[i, 0]
            im_input = ax_input.imshow(row_input, cmap=cmap, vmin=row_min, vmax=row_max,
                        interpolation='nearest', origin='upper', aspect='auto')
            plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)

            # Plot reconstruction (native resolution, no interpolation)
            ax_recon = axes[i, 1]
            im_recon = ax_recon.imshow(row_recon, cmap=cmap, vmin=row_min, vmax=row_max,
                        interpolation='nearest', origin='upper', aspect='auto')
            plt.colorbar(im_recon, ax=ax_recon, fraction=0.046, pad=0.04)

            # Plot abs diff (native resolution)
            ax_diff = axes[i, 2]
            diff_abs_max = np.max(np.abs(row_diff))
            im_diff = ax_diff.imshow(row_diff, cmap='bwr', vmin=-diff_abs_max, vmax=diff_abs_max,
                        interpolation='nearest', origin='upper', aspect='auto')
            plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

            # Titles
            if i == 0:
                ax_input.set_title("Input")
                ax_recon.set_title("Reconstruction")
                ax_diff.set_title("Difference")

            for ax in (ax_input, ax_recon, ax_diff):
                ax.axis('off')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
def save_contourf(inputs, save_path, cmap="turbo", v_min=None, v_max=None, colorbar=True):
    """
    Saves a contourf plot of the first sample in the batch without axes or margins.

    Args:
        inputs (torch.Tensor or np.ndarray): Input tensor of shape (B, C, H, W) or (B, H, W)
        save_path (str): Path to save the generated plot image
        cmap (str): Colormap for the contourf plot
    """
    # Convert to numpy if necessary
    if hasattr(inputs, 'detach'):
        inputs = inputs.detach().cpu().numpy()

    # Handle shape: (B, C, H, W) or (B, H, W) -> Select first batch (H, W)
    if inputs.ndim == 4:
        data = inputs[0, 0] # Take first batch, first channel
    elif inputs.ndim == 3:
        data = inputs[0]    # Take first batch
    else:
        data = inputs

    # Create the figure
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Generate contourf with 200 levels
    ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max,
                    interpolation='nearest', origin='upper', aspect='equal')

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if colorbar:
        fig.colorbar(ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max,
                    interpolation='nearest', origin='upper', aspect='auto'))
    
    # Save with no whitespace
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_spectrum_comparison(inputs, reconstructions, save_path):
    """
    Plot the 1D spectrum (mean power over spatial axes) of the first 10 inputs and reconstructions.
    Args:
        inputs (torch.Tensor or np.ndarray): Input tensor of shape (B, C, H, W) or (B, H, W)
        reconstructions (torch.Tensor or np.ndarray): Reconstructed tensor of same shape
        save_path (str): Path to save the generated plot image
    """
    # Convert to numpy if necessary
    if hasattr(inputs, 'detach'):
        inputs = inputs.detach().cpu().numpy()
    if hasattr(reconstructions, 'detach'):
        reconstructions = reconstructions.detach().cpu().numpy()

    # Handle shape: (B, C, H, W) or (B, H, W)
    if inputs.ndim == 4:
        inputs = inputs[:, 0]
    if reconstructions.ndim == 4:
        reconstructions = reconstructions[:, 0]

    num_samples = min(10, inputs.shape[0])
    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 2.5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        input_2d = inputs[i]
        recon_2d = reconstructions[i]

        # Compute 2D FFT, shift to center, and take magnitude
        input_fft = np.fft.fftshift(np.fft.fft2(input_2d))
        recon_fft = np.fft.fftshift(np.fft.fft2(recon_2d))
        input_power = np.abs(input_fft) ** 2
        recon_power = np.abs(recon_fft) ** 2

        # Radially average to get 1D spectrum
        def radial_average(power):
            y, x = np.indices(power.shape)
            center = np.array([power.shape[1] / 2.0, power.shape[0] / 2.0])
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            r = r.astype(np.int32)
            tbin = np.bincount(r.ravel(), power.ravel())
            nr = np.bincount(r.ravel())
            radialprofile = tbin / np.maximum(nr, 1)
            return radialprofile

        input_spectrum = radial_average(input_power)
        recon_spectrum = radial_average(recon_power)

        ax = axes[i]
        ax.plot(input_spectrum, label='Input', color='blue')
        ax.plot(recon_spectrum, label='Reconstruction', color='orange')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (radial)')
        ax.set_ylabel('Power')
        ax.legend()
        ax.set_title(f'Sample {i+1} Spectrum')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()