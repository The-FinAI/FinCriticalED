<h1 align="center">FinCriticalED: A Visual Benchmark for Financial Fact-Level OCR Evaluation</h1>

<p align="center">
  Yueru He, Xueqing Peng*, Yupeng Cao, Yan Wang, Lingfei Qian, Haohang Li, Yi Han, Shuyao Wang, Ruoyu Xiang, Fan Zhang, Zhuohan Xie, Mingquan Lin, Prayag Tiwari, Jimin Huang, Guojun Xiong, Sophia Ananiadou
</p>

<p align="center">
  <sup>*</sup>Corresponding author, xueqing.peng2024@gmail.com
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.14998">📖 Paper</a> •
  <a href="https://huggingface.co/datasets/TheFinAI/FinCriticalED">🤗 Dataset</a> •
  <a href="https://github.com/The-FinAI/FinCriticalED">💻 Code</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.14998"><img src="https://img.shields.io/badge/arXiv-2511.14998-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/TheFinAI/FinCriticalED"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="Dataset"></a>
  <a href="https://github.com/The-FinAI/FinCriticalED/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

## 📜Abstract

Recent progress in multimodal large language models (MLLMs) has substantially improved document understanding, yet strong OCR performance on surface metrics does not necessarily imply faithful preservation of decision-critical evidence. This limitation is especially consequential in financial documents, where small visual errors，such as a missing negative marker, shifted decimal point, incorrect unit scale, or misaligned reporting date, can induce materially different interpretations.

To study this gap, we introduce **FinCriticalED** (**Fin**ancial **Critical** **E**rror **D**etection), a fact-centric visual benchmark for evaluating OCR and vision-language systems through the lens of evidence fidelity in high-stakes document understanding. **FinCriticalED** contains 859 real-world financial document pages paired with ground-truth HTML, with 9,481 expert-annotated facts spanning five financially critical field types: *Numbers*, *Monetary Units*, *Temporal Data*, *Reporting Entities*, and *Financial Concepts*.

We further develop an evaluation suite, including critical-field-aware metrics and a context-aware protocol, to assess whether model outputs preserve financially critical facts beyond lexical similarity. We benchmark 13 OCR pipelines, OCR-native models, open-source VLMs, and proprietary MLLMs on **FinCriticalED**. Results show that conventional OCR metrics can substantially overestimate factual reliability, and that OCR-specialized systems may outperform much larger general-purpose MLLMs in preserving critical financial evidence under complex layouts. **FinCriticalED** provides a rigorous benchmark for trustworthy financial OCR and a broader testbed for high-stakes multimodal document understanding.

## 🏆Results

Model performance on FinCriticalED benchmark:

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Size</th>
      <th colspan="4">General (%)</th>
      <th colspan="7">Fact-Level (%)</th>
    </tr>
    <tr>
      <th>R1</th><th>RL</th><th>E↓</th><th>Rank</th>
      <th>N-FFA</th><th>T-FFA</th><th>M-FFA</th><th>R-FFA</th><th>FC-FFA</th><th>FFA</th><th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="13" align="center"><em>OCR Pipelines</em></td></tr>
    <tr>
      <td>MinerU2.5</td><td>1.2B</td>
      <td>-</td><td>-</td><td>-</td><td>-</td>
      <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>PP-OCRv5</td><td>0.07B</td>
      <td>97.54</td><td>96.55</td><td>3.10</td><td>-</td>
      <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr><td colspan="13" align="center"><em>Specialized OCR VLMs</em></td></tr>
    <tr><td>DeepSeek-OCR</td><td>6B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>DeepSeek-OCR-2</td><td>3B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>GLM-OCR</td><td>0.9B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td colspan="13" align="center"><em>Open-source MLLMs</em></td></tr>
    <tr><td>Gemma-3n-E4B-it</td><td>4B</td><td>83.49</td><td>79.59</td><td>23.82</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>Qwen3-VL-8B-Instruct</td><td>8B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>Llama-4-Maverick</td><td>17B</td><td>98.00</td><td>97.62</td><td>3.70</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>Qwen3.5-397B-A17B</td><td>397B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td colspan="13" align="center"><em>Proprietary MLLMs</em></td></tr>
    <tr><td>GPT-4o</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>GPT-5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>Claude-Sonnet-4.6</td><td>-</td><td><strong>98.84</strong></td><td><strong>98.73</strong></td><td><strong>1.69</strong></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>Gemini-2.5-Pro</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody>
</table>

> R1 = ROUGE-1, RL = ROUGE-L, E↓ = Edit Distance (lower is better), FFA = Fact-level Financial Accuracy. `-` = results pending.


## ⚙️Usage

### 1. Running Models

#### 1. Before running models, configure the `MODELS` list in **`main.py`** by uncommenting the model(s) you want to run.

#### 2. Create a `.env` file in `model_eval/` with the relevant API keys:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
ZAI_API_KEY = ...
```

#### 3. Run `main.py` to generate model OCR output:
```bash
cd model_eval
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
glm-ocr|	ZAI	| ZAI_API_KEY

- Local OCR Pipelines (no API key, local setup required):

Model key in MODELS	|Description
--- | --- 
paddleocrv5	|PP-OCRv5 plain text OCR — outputs raw text lines
monkeyocr	|MonkeyOCR — requires a running local server
mineru	|MinerU2.5 — runs locally via HuggingFace
deepseekocr	|DeepSeek-OCR — runs locally via HuggingFace
deepseekocr2	|DeepSeek-OCR — runs locally via HuggingFace

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
#### Traditional OCR Metrics
Upon running `main.py`, run evaluation.py to compute ROUGE-1, ROUGE-L, and Edit Distance metrics:

python evaluation.py

Results are saved as results/{model_tag}_zero-shot_rouge1_eval.csv.

#### LLM-as-Judge

In `llm-as-a-judge-prompt.py`, GPT-4o serves as the evaluator responsible for extracting financial entities (Numbers, Dates, Monetary Units, etc.) from the ground-truth HTML and verifying their presence in the model-generated HTML. The LLM Judge performs normalization, contextual matching, and fine-grained fact checking under a structured evaluation prompt.

## 🪶Citation

If you find this work useful, please cite:

```bibtex
@misc{he2025fincriticaledvisualbenchmarkfinancial,
      title={FinCriticalED: A Visual Benchmark for Financial Fact-Level OCR Evaluation}, 
      author={Yueru He and Xueqing Peng and Yupeng Cao and Yan Wang and Lingfei Qian and Haohang Li and Yi Han and Ruoyu Xiang and Mingquan Lin and Prayag Tiwari and Jimin Huang and Guojun Xiong and Sophia Ananiadou},
      year={2025},
      eprint={2511.14998},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14998}, 
}

