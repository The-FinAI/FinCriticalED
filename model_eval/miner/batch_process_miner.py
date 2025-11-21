from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient
import os
from pathlib import Path
from tqdm import tqdm


print("Loading model...")
# for transformers>=4.56.0
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B",
    dtype="auto",  # use `torch_dtype` instead of `dtype` for transformers<4.56.0
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B",
    use_fast=True
)

client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor
)
print("Model loaded successfully!\n")

# Configuration
results_base = 'miner-results'
start_folder = 495
end_folder = 500  # Process all 500 folders (0-499)

# Track progress and errors
successful = 0
failed = 0
errors = []

num_folders = end_folder - start_folder
print(f"Processing {num_folders} images (folders {start_folder} to {end_folder-1})...")

for folder_id in tqdm(range(start_folder, end_folder), desc="Processing images"):
    folder_path = os.path.join(results_base, str(folder_id))
    image_file = os.path.join(folder_path, f'image_{folder_id}.png')

    # Check if image exists
    if not os.path.exists(image_file):
        print(f"\nWarning: Image not found at {image_file}")
        failed += 1
        errors.append(f"Folder {folder_id}: Image not found")
        continue

    # Check if output already exists (to support resuming)
    output_file = os.path.join(folder_path, f'result.txt')
    if os.path.exists(output_file):
        successful += 1
        continue

    try:
        # Load image
        image = Image.open(image_file)

        # Run OCR extraction
        extracted_blocks = client.two_step_extract(image)

        # Combine all content from extracted blocks
        combined_content = '\n\n'.join(block['content'] for block in extracted_blocks)

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_content)

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

print("\nResults saved in respective subfolders under 'miner-results/'")
