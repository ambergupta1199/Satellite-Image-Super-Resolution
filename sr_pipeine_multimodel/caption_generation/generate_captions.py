import os
import glob
import csv
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from accelerate import PartialState

def generate_captions(dataset_path, output_file, batch_size=4, confidence_threshold=0.7):
    """Generate captions for satellite images using BLIP-2 with confidence filtering"""
    
    # Setup distributed environment
    device_string = PartialState().process_index
    device = torch.device(device_string if torch.cuda.is_available() else "cpu")
    
    # Load model with optimized settings
    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        use_fast=True,
        torch_dtype=torch.float16
    )
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    ).eval()

    # Get image paths
    image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.png") + 
                                 glob.glob(os.path.join(dataset_path, "*.jpg"))))
    
    print(f"Found {len(image_paths)} images to process")

    # Batch processing with confidence filtering
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lr_path', 'caption', 'confidence'])
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            try:
                # Load and process batch
                images = []
                valid_paths = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        images.append(img)
                        valid_paths.append(path)
                    except Exception as e:
                        print(f"Error loading {path}: {str(e)}")
                        continue
                
                if not images:
                    continue

                # Generate captions with confidence
                inputs = processor(
                    images=images,
                    text=["Describe this satellite image in detail:"]*len(images),
                    return_tensors="pt",
                    padding=True
                ).to(device, torch.float16)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_beams=5
                )
                
                # Decode with confidence scores
                captions = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                confidences = torch.exp(outputs.sequences_scores).cpu().tolist()
                
                # Write valid results
                for path, caption, confidence in zip(valid_paths, captions, confidences):
                    if confidence >= confidence_threshold:
                        writer.writerow([path, caption, confidence])
                    else:
                        writer.writerow([path, "low_confidence", confidence])
                        
                print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)}")

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                for path in batch_paths:
                    writer.writerow([path, "generation_error", 0.0])

if __name__ == "__main__":
    generate_captions(
        dataset_path="/home/sushil/ACPS Project/Dataset/3000/Sentinel/",
        output_file="captions/captions_raw.csv",
        batch_size=4,  # Adjust based on GPU memory
        confidence_threshold=0.65
    )