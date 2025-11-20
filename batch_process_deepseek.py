from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
from tqdm import tqdm

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)
print("Model loaded successfully!\n")

# Define prompt
prompt = "<image>\n<|grounding|>Convert the document to html. "

# Configuration
base_size = 1024
image_size = 640
crop_mode = True
test_compress = True
save_results = True

# Process subfolders from 100 to 499
results_base = 'results'
start_folder = 484
end_folder = 500  # exclusive, so this will process 100-499

# Track progress and errors
successful = 0
failed = 0
errors = []

num_folders = end_folder - start_folder
print(f"Processing {num_folders} images (folders {start_folder} to {end_folder-1})...")
print(f"Configuration: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}\n")

for folder_id in tqdm(range(start_folder, end_folder), desc="Processing images"):
    folder_path = os.path.join(results_base, str(folder_id))
    image_file = os.path.join(folder_path, f'image_{folder_id}.png')

    # Check if image exists
    if not os.path.exists(image_file):
        print(f"Warning: Image not found at {image_file}")
        failed += 1
        errors.append(f"Folder {folder_id}: Image not found")
        continue

    try:
        # Run OCR inference
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=folder_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            test_compress=test_compress
        )

        successful += 1


    except Exception as e:
        failed += 1
        error_msg = f"Folder {folder_id}: {str(e)}"
        errors.append(error_msg)
        print(f"\nError processing folder {folder_id}: {str(e)}")


# Print summary
print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)
print(f"Total folders: {num_folders}")
print(f"Successfully processed: {successful}")
print(f"Failed: {failed}")

if errors:
    print("\nErrors encountered:")
    for error in errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")

print("\nResults saved in respective subfolders under 'results/'")
