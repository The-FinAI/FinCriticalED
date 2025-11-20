from transformers import AutoModel, AutoTokenizer
import torch
import os
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_name = 'deepseek-ai/DeepSeek-OCR'

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# Define prompt and paths
prompt = "<image>\n<|grounding|>Convert this financial statement page into semantically correct HTML. Return html and nothing else. Use plain html only, no styling please."
csv_file = 'FinOCRBench_Task1_input.csv'
output_path = 'results'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Helper function to save text files
def save_text(text, output_file, suffix=".txt"):
    path = Path(output_file)

    # If the provided path has an image extension, replace it with the desired suffix
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        path = path.with_suffix(suffix)
    elif not path.suffix and suffix:
        path = path.with_suffix(suffix)
    elif suffix and path.suffix != suffix:
        path = path.with_suffix(suffix)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if isinstance(text, str) else str(text), encoding="utf-8")
    return str(path)

# Load CSV file
print(f"Loading CSV file: {csv_file}")
df = pd.read_csv(csv_file)

# Process each image in the CSV
results = []
for idx, row in df.iterrows():
    print(f"\nProcessing row {idx}/{len(df)-1}...")

    # Get the ID for this row
    row_id = row['id']

    # Create sub-folder for this ID
    id_folder = os.path.join(output_path, str(row_id))
    os.makedirs(id_folder, exist_ok=True)

    # Extract base64 image data
    image_data = row['data.image']

    # Remove the data URL prefix if present (e.g., "data:image/png;base64,")
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]

    # Decode base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    # Save image file in the ID sub-folder
    image_path = os.path.join(id_folder, f'image_{row_id}.png')
    image.save(image_path)
    print(f"Saved image to: {image_path}")

    # Run OCR inference
    try:
        res = model.infer(tokenizer,
                          prompt=prompt,
                          image_file=image_path,
                          output_path=id_folder,
                          base_size=1024,
                          image_size=640,
                          crop_mode=True,
                          save_results=False,  # We'll save manually
                          test_compress=True)

        # Save the HTML output using the save_text function
        output_file = os.path.join(id_folder, f'output_{row_id}.txt')
        saved_path = save_text(res, output_file, suffix=".txt")
        print(f"Saved HTML output to: {saved_path}")

        results.append({
            'id': row_id,
            'ocr_result': res,
            'original_title': row.get('data.title', ''),
            'image_path': image_path,
            'txt_path': saved_path
        })

        print(f"Row {idx} (ID: {row_id}) processed successfully")
    except Exception as e:
        print(f"Error processing row {idx} (ID: {row_id}): {str(e)}")
        results.append({
            'id': row_id,
            'ocr_result': f'ERROR: {str(e)}',
            'original_title': row.get('data.title', ''),
            'image_path': image_path,
            'html_path': 'ERROR'
        })

# Save results to CSV
output_df = pd.DataFrame(results)
output_csv = os.path.join(output_path, 'ocr_results.csv')
output_df.to_csv(output_csv, index=False)
print(f"\nAll results saved to: {output_csv}")
print(f"Total images processed: {len(results)}")