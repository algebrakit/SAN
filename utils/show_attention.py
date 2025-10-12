# In inference.py or a new script
import matplotlib.pyplot as plt
import numpy as np

# Visualize attention for a specific timestep
def visualize_attention(image, alpha, query_alpha, coverage_alpha, timestep, predicted_symbol):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,6))

    # Original image
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.set_title('Original Image')

    # Attention heatmap
    # attention = alpha[0, timestep].cpu().numpy()
    attention = alpha.cpu().numpy()
    ax2.imshow(image.squeeze(), cmap='gray', alpha=0.5)
    ax2.imshow(attention, cmap='hot', alpha=0.5)
    ax2.set_title(f'Attention at step {timestep}: {predicted_symbol}')
    ax3.imshow(query_alpha.squeeze(), cmap='hot', alpha=0.5)
    ax3.set_title('Coverage from query')
    ax4.imshow(coverage_alpha.squeeze(), cmap='hot', alpha=0.5)
    ax4.set_title('Coverage Alpha')

    plt.savefig(f'attention_step_{timestep}.png')

