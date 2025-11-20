import os
import pandas as pd
from tqdm import tqdm
from evaluate import load
from lib.tools import Tools
import re
import html
from bs4 import BeautifulSoup
import Levenshtein

rouge = load("rouge")


def html_to_text(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)

    if "<" not in s or ">" not in s:
        t = html.unescape(s)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    

    soup = BeautifulSoup(s, "lxml")  # if not lxml then use "html.parser"

    for tag in soup(["script", "style", "noscript", "template", "iframe"]):
        tag.decompose()
    # get pure text
    t = soup.get_text(separator=" ", strip=True)
    # data clean
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def calculate_edit_distance(reference, hypothesis):
    if not reference and not hypothesis:
        return 0.0

    if not reference or not hypothesis:
        return 1.0

    distance = Levenshtein.distance(reference, hypothesis)
    max_len = max(len(reference), len(hypothesis))

    return distance / max_len if max_len > 0 else 0.0

def evaluate_rouge(pred_dir, ground_truths, model_name="gpt-4o",lang='en'):
    records = []

    for i in tqdm(ground_truths.index, total=len(ground_truths), desc="Evaluating ROUGE"):
        gt = ground_truths.loc[i]
        if pd.isna(gt) or not isinstance(gt, str):
            continue

        pred_path = os.path.join(pred_dir, f"pred_{i}.txt")
        if not os.path.exists(pred_path):
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred = f.read().strip()
            clean_pred = pred
            if lang == "en":
                clean_pred = html_to_text(pred)
                gt = html_to_text(gt)

        try:
            rouge_score = rouge.compute(predictions=[clean_pred], references=[gt], use_stemmer=True)
            rouge_1_f1 = float(rouge_score["rouge1"])
            rouge_l_f1 = float(rouge_score["rougeL"])
            edit_distance = calculate_edit_distance(gt, clean_pred)
        except Exception as e:
            print(f"ROUGE error on index {i}: {e}")
            rouge_1_f1 = None

        records.append({
            "index": i,
            "ground_truth": gt,
            "prediction": pred,
            "ROUGE-1": rouge_1_f1,
            "ROUGE-L": rouge_l_f1,
            "Edit Distance": edit_distance,
            "Model": model_name,
            "Language": lang
        })

    df_eval = pd.DataFrame(records)

    # Create table format output
    result = {
        'language': [lang],
        'model': [model_name], 
        'sample_size': [len(df_eval)],
        'rouge1': [df_eval['ROUGE-1'].mean()],
        'rougeL': [df_eval['ROUGE-L'].mean()],
        'edit_distance': [df_eval['Edit Distance'].mean()]
    }
    df_result = pd.DataFrame(result)
    print(df_result.to_string(index=False, float_format='%.4f'))

    return df_eval, df_result

def run_rouge_eval(
    model_name="gpt-4o",
    experiment_tag="zero-shot",
    language="en",
    local_version = True, 
    local_dir = "./FinCriticalED", 
):
    LOCAL_FILES = {
        "smallocr": ["FinCriticalED/FinOCRBench_Task1_input.csv"]
    }
    

    valid_langs = {"smallocr"}
    if language not in valid_langs:
        raise ValueError(f"Invalid language '{language}'. Choose from {sorted(valid_langs)}.")
        
    if local_version:
        paths = [os.path.join(local_dir, p) for p in LOCAL_FILES[language]]
        if language == "smallocr":
            df = pd.read_csv(paths[0])
        else:
            print("Invalid Input")
    else:
        ds = load_dataset("TheFinAI/FinCriticalED", data_files=REMOTE_FILES[language])
        df = ds["train"].to_pandas()

    pred_dir = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}'

    # Extract indices from prediction files
    pred_indexes = []
    for fname in os.listdir(pred_dir):
        if fname.startswith(f"pred_") and fname.endswith(".txt"):
            try:
                idx = int(fname.replace(f"pred_", "").replace(".txt", ""))
                pred_indexes.append(idx)
            except:
                continue

    df = df.loc[df.index.intersection(pred_indexes)]
    df_eval, df_result = evaluate_rouge(pred_dir, df["data.matched_html"], model_name=model_name, lang=language)

    eval_fp = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}_rouge1_eval.csv'
    result_fp = f'./results/{language}/{model_name.replace("/", "-")}_{experiment_tag}_rouge1_result.csv'
    df_eval.to_csv(eval_fp, index=False)
    df_result.to_csv(result_fp, index=False)
    print(f"✅ Evaluation saved to CSV")
    return df_eval

def main():
    models = [
        "gpt-4o",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # "google/gemma-3-4b-it",
        # "google/gemma-3-27b-it",
        # "Qwen/Qwen2.5-Omni-7B",
        # "TheFinAI/FinLLaVA",
        # "Qwen/Qwen-VL-Max",
        # "liuhaotian/llava-v1.6-vicuna-13b",
        # "deepseek-ai/deepseek-vl-7b-chat",
        
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "google/gemma-3n-E4B-it",
        "gpt-5",
    ]
    languages = [
        "smallocr"
    ]

    for model in models:
        for language in languages:
            print(f"🟢 Start evaluating {model} in {language}")
            try:
                run_rouge_eval(
                    model_name=model,
                    experiment_tag="zero-shot",
                    language=language,
                    local_version = True, 
                    local_dir = "./FinCriticalED", 
                )
            except Exception as e:
                print(f"⚠️ Error with model {model} in {language}: {e}")
                continue

if __name__ == '__main__':
    main()


