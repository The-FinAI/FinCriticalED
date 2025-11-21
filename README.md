## FinCriticalED
Annotation, model evaluation, and LLM-as-Judge evaluation framework for FinCriticalED. 


## Repository Structure
### 1. Annotation
```Annotation``` folder contains post-processing and annotation quality assessments. 

- ```Annotation/Highlight_Annotation.ipynb```: process expert annotations by wrapping financially critical entities with ```<"Number">``` and ```<"Time">``` labels to make it the gold standard of FinCriticalED dataset.

- ```Annotation/calculate_agreement.py``` and ```Annotation/run_agreement.sh```: calculate overall and pairwise annotator agreement scores to ensure annotation quality
### 2. Running Models
1. Before running models, configure <b>model_eval/agent.py</b> and OPEN_AI_API_KEY and TOGETHER_API_KEY for running models
2. Run ```model_eval/main.py``` to generate model OCR output. 
To change model, or only run model on small sample, update
```
def evaluate(
    model_name="gpt-4o", 
    experiment_tag="zero-shot", 
    language="en", 
    local_version=True, 
    local_dir="./FinCriticalED", 
    sample=None
):
```
3. DeepSeek-OCR and and MinerU2.5 can be run seperately in ```model_eval/deepseekocr/batch_process_deepseek.py``` and ```model_eval/miner/batch_process_miner.py```, respectively.
### 3. Running Evaluation on traditional OCR metrics
Upon running main.py, run <b>model_eval/evaluation.py</b> on ROUGE-1,ROUGE-L, Edit Distance. 
To control input output path, or change models, csv names etc., update
```
def main():
    models = [
        ...
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "google/gemma-3n-E4B-it",
        "gpt-5",
    ]
    languages = [
        "smallocr"
         ...
    ]
```
### 4. LLM-as-Judge on comparing financial OCR results to gold standards
In ```llm-as-a-judge.ipynb```, a large language model (GPT-4o) serves as the evaluator responsible for extracting financial facts from the ground-truth HTML and verifying their presence in the model-generated HTML. The LLM Judge processes both inputs under a structured evaluation prompt, enabling it to perform normalization, contextual matching, and fine-grained fact checking.
