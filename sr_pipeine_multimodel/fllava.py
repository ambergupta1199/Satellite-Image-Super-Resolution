import os
import csv
import torch
from PIL import Image
from transformers import AutoTokenizer
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    load_image_from_base64,
    tokenizer_image_token
)

def generate_geospatial_captions(image_dir, output_file):
    # Model configuration
    model_path = "liuhaotian/llava-v1.5-7b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model
    model_name = get_model_name_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        use_safetensors=True
    ).eval()
    
    # Satellite-specific prompt
    prompt = """[INST] Analyze this satellite image in detail:
1. Describe the primary land use type
2. Identify key geographical features
3. Note any spatial patterns
4. Highlight anomalies or unique characteristics
[/INST]"""
    
    # Process images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'caption'])
        
        for img_path in image_paths:
            try:
                # Load and process image
                image = Image.open(img_path).convert('RGB')
                image_tensor = process_images([image], model.config)[0]
                image_tensor = image_tensor.to(device, dtype=torch_dtype)
                
                # Tokenize inputs
                input_ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    return_tensors='pt'
                ).unsqueeze(0).to(device)
                
                # Generate caption
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0),
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=300,
                        use_cache=True
                    )
                
                # Decode and clean output
                caption = tokenizer.decode(
                    output_ids[0][len(input_ids[0]):], 
                    skip_special_tokens=True
                ).strip()
                
                writer.writerow([img_path, caption])
                print(f"Processed {img_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                writer.writerow([img_path, "generation_error"])

if __name__ == "__main__":
    generate_geospatial_captions(
        image_dir="/home/sushil/ACPS Project/Dataset/3000/Sentinel/",
        output_file="/home/sushil/ACPS Project/Dataset/sr_pipeline_deepseek/captions/llava_rs_captions.csv"
    )