import argparse
import os
import multiprocessing as mp
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm  # Corrected import

def process_image(args):
    """Worker function for parallel processing"""
    input_path, output_dir, extensions = args
    try:
        # Skip non-image files
        if not any(input_path.lower().endswith(ext) for ext in extensions):
            return

        # Create output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.pt')

        # Skip existing files
        if os.path.exists(output_path):
            return

        # Process and save image
        img = Image.open(input_path).convert('RGB')
        tensor = ToTensor()(img)
        torch.save(tensor, output_path)

    except Exception as e:
        print(f"\nError processing {input_path}: {str(e)}")

def main(input_dir, output_dir, extensions, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            image_paths.append(os.path.join(root, f))
    
    tasks = [(path, output_dir, extensions) for path in image_paths]
    
    print(f"Processing {len(tasks)} images with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        # Wrap pool.imap with tqdm
        list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess satellite images')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--extensions', nargs='+', default=['.png', '.jpg', '.jpeg'])
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count())
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        extensions=args.extensions,
        num_workers=args.num_workers
    )