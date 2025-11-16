# FinCriticalED
Repo for FinCriticalED


# How to use 
1. Before running gpt, please go to <b>main/lib/agent.py</b> and put in your openai_api_key
   
3. Run main/main.py for models to generate OCR output. 
The model is default to be gpt-4o, and language default to be English. If want to change languagr or model, or only run model on small sample, update this part:
```
def main():
    evaluate(model_name="gpt-4o",language = "en",local_version = False , sample = 20)
```

3. After running main.py, run <b>main/evaluation.py</b> to both output evaluation metrics (ROUGE-1). 
To control input output path, or change models, csv names etc., please update this part:
```
def main():
    run_rouge_eval(
        parquet_path="hyr_ocr_process/japanese_output_parquet/japanese_batch_0000.parquet",
        pred_dir="hyr_results/predictions_japanese/gpt-4o_zero-shot_financial",
        model_name="gpt-4o",
        lang = "jp",
        output_csv="hyr_results/eval_rouge/eval_japanese_gpt_4o_rouge.csv"
    )
```

# Dataset
- Dataset are available on HuggingFace: [TheFinAI/FinCriticalED](https://huggingface.co/datasets/TheFinAI/FinCriticalED)


