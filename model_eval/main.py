import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from agent import Agent

load_dotenv()

# All supported models. Comment out any you don't want to run.
MODELS = [
    ############### OCR Pipeline
    # "mineru",
    # "monkeyocr",
    # "paddleocr",
    # "paddleocrv5-ppstructure",
    ################# OCR Models
    # "deepseekocr",
    # "deepseekocr-2",
    # ############ Together AI
    # "google/gemma-3n-E4B-it",
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    # "moonshotai/Kimi-K2.5",
    #  ######### Open Source
    # "Qwen/Qwen2.5-VL-72B-Instruct",
    # "Qwen/Qwen3-VL-32B-Instruct",
    # ############### OpenAI
    # "gpt-4o",
    # "gpt-5",
    # ############### Anthropic
    # "claude-sonnet-4-6",
    # ################# Google
    # "gemini-2.5-pro",
    # ################# Z.ai
    # "glm-4.6v-flash",
    "glm-ocr",
]

# Path to the evaluation CSV (relative to this script's location)
DATA_CSV = "./data/raw_input.csv"

# Results are written to ../results/smallocr/{model_tag}/pred_{i}.txt
RESULTS_DIR = "./results"


def evaluate(model_name, experiment_tag="zero-shot", max_samples=None):
    df = pd.read_csv(DATA_CSV)

    out_dir = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '-')}_{experiment_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # Skip already-completed samples
    done = set()
    for fname in os.listdir(out_dir):
        if fname.startswith("pred_") and fname.endswith(".txt"):
            try:
                done.add(int(fname[5:-4]))
            except ValueError:
                pass
    df = df[~df.index.isin(done)]

    if max_samples is not None:
        df = df.head(max_samples)

    if df.empty:
        print(f"All samples already complete for {model_name}.")
        return

    agent = Agent(model_name)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
        image_path = row.get("image_path", row.get("image", row.get("data.image")))
        out_file = os.path.join(out_dir, f"pred_{i}.txt")
        try:
            result = agent.draft(image_path, local_version=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(result)
        except Exception as e:
            print(f"  Error on index {i}: {e}")


def main():
    max_samples = 1 # Set to an integer to limit samples per model, e.g. max_samples = 10
    for model in MODELS:
        print(f"\n=== {model} ===")
        try:
            evaluate(model, experiment_tag="zero-shot", max_samples=max_samples)
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
