import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread

# Define folder paths - replace these with your actual folder paths
original_folder = '/home/nitin/acps/3000/Planetscope'
generated_folder = '/home/nitin/acps/ESRGAN/results'

# List all image files in both folders
original_images = sorted([os.path.join(original_folder, f) for f in os.listdir(original_folder) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
generated_images = sorted([os.path.join(generated_folder, f) for f in os.listdir(generated_folder) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))])

# Ensure both folders have the same number of images
if len(original_images) != len(generated_images):
    raise ValueError("The number of images in the original and generated folders do not match.")

# Function to calculate SAM
def calculate_sam(img1, img2):
    """Calculate Spectral Angle Mapper (SAM) in radians"""
    # Reshape images to 2D arrays (pixels × bands)
    img1_flat = img1.reshape(-1, img1.shape[2]) if len(img1.shape) > 2 else img1.reshape(-1, 1)
    img2_flat = img2.reshape(-1, img2.shape[2]) if len(img2.shape) > 2 else img2.reshape(-1, 1)
    
    # Calculate dot product
    dot_product = np.sum(img1_flat * img2_flat, axis=1)
    
    # Calculate magnitudes
    norm_img1 = np.sqrt(np.sum(img1_flat**2, axis=1))
    norm_img2 = np.sqrt(np.sum(img2_flat**2, axis=1))
    
    # Calculate cosine of angle
    cos_angle = dot_product / (norm_img1 * norm_img2 + 1e-10)
    
    # Clip values to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in radians
    angle = np.arccos(cos_angle)
    
    # Return mean SAM value
    return np.mean(angle)

# Calculate metrics for each pair of images
results = []
for i, (original_path, generated_path) in enumerate(zip(original_images, generated_images)):
    original = imread(original_path)
    generated = imread(generated_path)
    
    # Ensure images have the same dimensions
    if original.shape != generated.shape:
        print(f"Warning: Image dimensions do not match for {original_path} and {generated_path}. Skipping.")
        continue
    
    # Calculate metrics
    psnr_value = peak_signal_noise_ratio(original, generated)
    
    if len(original.shape) == 2:  # Grayscale images
        ssim_value = structural_similarity(original, generated)
    else:  # Color images
        ssim_value = structural_similarity(original, generated, channel_axis=2)
    
    sam_value = calculate_sam(original, generated)
    
    # Store results
    results.append({
        'image_pair': i+1,
        'original': os.path.basename(original_path),
        'generated': os.path.basename(generated_path),
        'psnr': psnr_value,
        'ssim': ssim_value,
        'sam': sam_value
    })

# Print results
print("\nImage Quality Metrics for Each Pair:")
print("=" * 80)
print(f"{'Pair':<6}{'Original':<20}{'Generated':<20}{'PSNR (dB)':<12}{'SSIM':<12}{'SAM (rad)':<12}")
print("-" * 80)

for r in results:
    print(f"{r['image_pair']:<6}{r['original']:<20}{r['generated']:<20}"
          f"{r['psnr']:<12.4f}{r['ssim']:<12.4f}{r['sam']:<12.4f}")

# Calculate average metrics
avg_psnr = np.mean([r['psnr'] for r in results])
avg_ssim = np.mean([r['ssim'] for r in results])
avg_sam = np.mean([r['sam'] for r in results])

print("=" * 80)
print(f"Average: {'':<40}PSNR: {avg_psnr:.4f} dB    SSIM: {avg_ssim:.4f}    SAM: {avg_sam:.4f} rad")
