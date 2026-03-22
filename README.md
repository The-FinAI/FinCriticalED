## FinOCRBench

Inference, model evaluation, and LLM-as-Judge evaluation framework for FinOCRBench — a financial document OCR benchmark evaluating vision-language models on HTML reconstruction of SEC filings and other financial documents.


## Repository Structure

### 1. Running Models

#### 1. Before running models, configure the `MODELS` list in **`main.py`** by uncommenting the model(s) you want to run.

#### 2. Create a `.env` file in `FinOCRBench/code/` with the relevant API keys:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
```

#### 3. Run `main.py` to generate model OCR output:
```bash
cd FinOCRBench/code
python main.py
```
- Results are saved to `results/{model_tag}_zero-shot/pred_{i}.txt`. Already-completed samples are skipped on re-runs.

- To limit the number of samples for a quick test, set max_samples in main():

`max_samples = 10`

- Supported Models
Cloud VLMs (API key required, no local setup):

Model key in MODELS|	Provider|	API Key
--- | --- | --- 
gpt-4o|	OpenAI|	OPENAI_API_KEY
gpt-5|	OpenAI|	OPENAI_API_KEY
claude-sonnet-4-6|	Anthropic	|ANTHROPIC_API_KEY
gemini-2.5-pro|	Google|	GOOGLE_API_KEY
meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8|	Together AI|	TOGETHER_API_KEY
Qwen/Qwen2.5-VL-72B-Instruct|	Together AI|	TOGETHER_API_KEY
moonshotai/Kimi-K2.5|	Together AI	|TOGETHER_API_KEY

- Local OCR Pipelines (no API key, local setup required):

Model key in MODELS	|Description
--- | --- 
paddleocrv5	|PP-OCRv5 plain text OCR — outputs raw text lines
paddleocrv5-table	|PP-OCRv5 + table recognition — outputs HTML with `<table>` and `<p>` tags
monkeyocr	|MonkeyOCR — requires a running local server
mineru	|MinerU2.5 — runs locally via HuggingFace
deepseekocr	|DeepSeek-OCR — runs locally via HuggingFace

- Setting up paddleocr and paddleocrv5-ppstructure
Linux or WSL is required. PaddlePaddle has limited Windows support.

Install dependencies:
```bash
pip install paddleocr paddlepaddle
# For paddleocrv5-ppstructure, also install:
pip install shapely pyclipper scikit-image imutils lmdb
```
Model weights (~200 MB) are downloaded automatically on first run.

`paddleocrv5` runs PP-OCRv5 locally on CPU and outputs plain text. Some samples may raise a PaddlePaddle oneDNN/PIR compatibility error — these are caught and skipped automatically. No configuration needed beyond the install above.

`paddleocrv5-table` uses `TableRecognitionPipelineV2` to detect tables and output them as HTML `(<table>/<tr>/<td>)`, with all other text as <p> tags — producing a full HTML document per page. Note: PPStructureV3 requires CUDA and is not supported in CPU-only environments.

- Setting up monkeyocr

MonkeyOCR requires a running HTTP server. Start the server separately before running main.py (see the MonkeyOCR repository for server setup instructions).

Set the server URL via environment variable (defaults to `http://localhost:8000`):

`MONKEYOCR_API_URL=http://your-server:8000`

### 2. Running Evaluation
Upon running `main.py`, run evaluation.py to compute ROUGE-1, ROUGE-L, and Edit Distance metrics:

python evaluation.py

Results are saved as results/{model_tag}_zero-shot_rouge1_eval.csv.

### 3. LLM-as-Judge

In llm-as-a-judge-prompt.py, GPT-4o serves as the evaluator responsible for extracting financial entities (Numbers, Dates, Monetary Units, etc.) from the ground-truth HTML and verifying their presence in the model-generated HTML. The LLM Judge performs normalization, contextual matching, and fine-grained fact checking under a structured evaluation prompt.


