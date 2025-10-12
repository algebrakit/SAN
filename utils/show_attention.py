# In inference.py or a new script
import matplotlib.pyplot as plt
import numpy as np

# Visualize attention for a specific timestep
def visualize_attention(image, alpha, query_alpha, coverage_alpha, timestep, predicted_symbol):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,8))

    # Original image
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Attention heatmap (top-right) - always positive (softmax output)
    attention = alpha.cpu().numpy()
    ax2.imshow(image.squeeze(), cmap='gray', alpha=0.5)
    im2 = ax2.imshow(attention, cmap='hot', alpha=0.5)
    ax2.set_title(f'Attention at step {timestep}: {predicted_symbol}')
    ax2.axis('off')
    # Add colorbar with value range
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(f'min={attention.min():.3f}, max={attention.max():.3f}', fontsize=8)

    # Query Alpha (bottom-left) - can be positive/negative
    query_alpha_np = query_alpha.squeeze().cpu().numpy()
    query_vmax = max(abs(query_alpha_np.min()), abs(query_alpha_np.max()))
    im3 = ax3.imshow(query_alpha_np, cmap='RdBu_r', vmin=-query_vmax, vmax=query_vmax)
    ax3.set_title('Query + Features (red=boost, blue=suppress)')
    ax3.axis('off')
    # Add colorbar with value range
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(f'min={query_alpha_np.min():.3f}, max={query_alpha_np.max():.3f}', fontsize=8)

    # Coverage Alpha (bottom-right) - can be positive/negative
    coverage_alpha_np = coverage_alpha.squeeze().cpu().numpy()
    coverage_vmax = max(abs(coverage_alpha_np.min()), abs(coverage_alpha_np.max()))
    im4 = ax4.imshow(coverage_alpha_np, cmap='RdBu_r', vmin=-coverage_vmax, vmax=coverage_vmax)
    ax4.set_title('Coverage (red=boost, blue=suppress)')
    ax4.axis('off')
    # Add colorbar with value range
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label(f'min={coverage_alpha_np.min():.3f}, max={coverage_alpha_np.max():.3f}', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'attention_step_{timestep}.png', dpi=100, bbox_inches='tight')

