import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import transformation functions directly from your data_loader module
from data_loader import (
    fetch_input_output,
    perform_gramian_transform,
    perform_matrix_transform,
    normalize_data,
    fetch_scaler
)
from helper_functions import DATA_TYPES, MATRIX_TYPES

def plot_permutations(
    X_single_row: np.ndarray, 
    permutations_dict: dict, 
    method: str = MATRIX_TYPES.GAF_TRANSFORM.value,
    colormap: str = 'inferno',
    save_filename: str = ''
):
    """
    Transforms and plots a single feature vector across multiple feature order permutations.
    
    :param X_single_row: 1D or 2D array representing a single sample of 16 features.
    :param permutations_dict: Dict mapping a label to a 1D permutation index array/list.
    :param method: Matrix transform type from MATRIX_TYPES (default: GAF).
    :param colormap: Matplotlib colormap (e.g., 'inferno', 'gist_gray', 'viridis').
    :param save_filename: Optional path to save the generated figure.
    """
    num_perms = len(permutations_dict)
    fig, axes = plt.subplots(1, num_perms, figsize=(4 * num_perms, 4))
    if num_perms == 1:
        axes = [axes]

    mpl.rcParams.update({'font.size': 11})
    
    # Ensure 2D shape (1, 16)
    if X_single_row.ndim == 1:
        X_single_row = X_single_row.reshape(1, -1)

    for ax, (label, perm_indices) in zip(axes, permutations_dict.items()):
        # 1. Permute the feature order
        X_permuted = X_single_row[:, perm_indices]
        
        # 2. Perform 2D transformation using data_loader pipeline
        if method == MATRIX_TYPES.GAF_TRANSFORM.value:
            mat_2d = perform_gramian_transform(X_permuted, method='s')[0, :, :, 0]
        else:
            mat_2d = perform_matrix_transform(X_permuted, method)[0, :, :, 0]

        # Save each transformed matrix for potential further analysis as global variables (optional)
        globals()[f"{label.replace(' ', '_')}_matrix"] = mat_2d

        # 3. Render 2D matrix plot
        im = ax.imshow(mat_2d, cmap=colormap)
        ax.set_title(f"{label}", fontsize=12, pad=10)
        
        if mat_2d.shape[0] == 16:
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0, 5, 10, 15])
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_filename:
        plt.savefig(f"{save_filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# Example Usage & Permutation Generation
# ==============================================================================
if __name__ == '__main__':
    np.random.seed(42)

    # Sample the first test feature vector of 16 features
    sample_feature_vector, _ = fetch_input_output(DATA_TYPES.TESTING.value, is_multiclass=False)  # Fetch first row of test input
    sample_feature_vector = sample_feature_vector.iloc[0]  # Get the first row (1D array of 16 features)
    scaler = fetch_scaler((-1, 1), is_multiclass=False)
    normalized_vector = normalize_data(sample_feature_vector.values.reshape(1, -1), scaler)[0]

    # Define feature order permutations to test
    original_indices = list(range(16))
    
    # Random permutation 1
    rng_1 = np.random.default_rng(seed=101)
    perm_1 = rng_1.permutation(16)
    
    # Random permutation 2
    rng_2 = np.random.default_rng(seed=202)
    perm_2 = rng_2.permutation(16)
    
    # Reversed feature order
    perm_reverse = original_indices[::-1]

    permutations_to_test = {
        "Original Order": original_indices,
        "Permutation Seed 101": perm_1,
        "Permutation Seed 202": perm_2,
        "Reversed Order": perm_reverse
    }

    print("--- Feature Order Permutations ---")
    for name, indices in permutations_to_test.items():
        print(f"{name}: {list(indices)}")

    # 1. Plot GAF Transformations across permutations
    plot_permutations(
        X_single_row=normalized_vector,
        permutations_dict=permutations_to_test,
        method=MATRIX_TYPES.GAF_TRANSFORM.value,
        colormap='inferno',
        save_filename='../plots/gaf_permutation_comparison'
    )

    # 2. Plot Circulant Transformations across permutations
    plot_permutations(
        X_single_row=normalized_vector,
        permutations_dict=permutations_to_test,
        method=MATRIX_TYPES.CIRCULANT_TRANSFORM.value,
        colormap='gist_gray',
        save_filename='../plots/circulant_permutation_comparison'
    )