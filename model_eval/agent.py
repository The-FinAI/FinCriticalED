import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import base64
import io
import tempfile
import requests

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT = (
    """
You are an expert OCR system for financial filings and tabular financial documents.

Transcribe the provided document image into one valid HTML document that faithfully preserves the page content and structure.

The output will be evaluated for exact financially critical OCR accuracy. Do not summarize, interpret, normalize, repair, or complete the content.

Critical field types that must be preserved exactly:
1. Number: integers, decimals, percentages, signed values, parenthesized values, comma-separated values. Examples: 1,234 ; 10.5 ; 25% ; (500)
2. Temporal: years, dates, quarters, months, time periods. Examples: 2024 ; December 31, 2023 ; Q1 2025
3. Monetary Unit: only currency markers and money scale markers, not full amounts. Examples: $ ; US$ ; million ; billion ; thousand
   - In $500, the monetary unit is $
   - In US$500 million, the monetary units are US$ and million
4. Reporting Entity: company names, legal entities, trusts, funds, organizations, subsidiaries, or identifying person names
5. Financial Concept: accounting and finance-specific concepts or line items. Examples: Revenue ; Net income ; Total assets ; Operating expenses

Critical errors to avoid:
1. Number Error: changing digits, commas, decimals, signs, parentheses, or percent marks
2. Temporal Error: changing any date, year, quarter, or time period
3. Monetary Unit Error: dropping or altering $, US$, million, billion, thousand, etc.
4. Reporting Entity Error: corrupting or substituting entity/person names
5. Financial Concept Error: replacing a financial term with a different term

Rules:
- Preserve exact visible text.
- Preserve punctuation, capitalization, commas, decimals, symbols, and abbreviations exactly.
- Do not normalize numbers or dates.
- Do not convert units.
- Do not fix spelling.
- Do not infer missing text.
- Do not paraphrase.
- Do not hallucinate.
- Preserve reading order.
- Preserve paragraph and heading structure.
- Preserve table structure with correct rows and columns.
- Keep values in the correct cells.
- Use clean semantic HTML only.

Use tags such as: <html>, <body>, <h1>, <h2>, <h3>, <p>, <table>, <tr>, <th>, <td>.

Do not output markdown, code fences, comments, JSON, XML, CSS, JavaScript, or any explanation.

Return exactly one HTML document and nothing else.

"""
)

TOGETHER_MODELS = frozenset({
    "google/gemma-3n-E4B-it",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "moonshotai/Kimi-K2.5",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3-VL-8B-Instruct",
})

GPT_MODELS = frozenset({"gpt-4o", "gpt-5"})
ANTHROPIC_MODELS = frozenset({"claude-sonnet-4-6"})
GOOGLE_MODELS = frozenset({"gemini-2.5-pro"})
GLM_MODELS = frozenset({"glm-4.6v-flash", "glm-ocr"})
OCRPIPELINE_MODELS = frozenset({"monkeyocr", "paddleocrv5", "paddleocrv5-table"})


class Agent:
    def __init__(self, model_name):
        self.model_name = model_name

        if model_name == "mineru":
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from mineru_vl_utils import MinerUClient

            _model = Qwen2VLForConditionalGeneration.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                dtype="auto",
                device_map="auto",
            )
            _processor = AutoProcessor.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                use_fast=True,
            )
            self.mineru_client = MinerUClient(
                backend="transformers",
                model=_model,
                processor=_processor,
            )

        elif model_name == "deepseekocr":
            import torch
            from transformers import AutoModel, AutoTokenizer

            _name = "deepseek-ai/DeepSeek-OCR"
            self.tokenizer = AutoTokenizer.from_pretrained(_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = (
                AutoModel.from_pretrained(
                    _name,
                    _attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                    use_safetensors=True,
                )
                .eval()
                .cuda()
                .to(torch.bfloat16)
            )

        elif model_name in TOGETHER_MODELS:
            from together import Together
            self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

        elif model_name in GPT_MODELS:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        elif model_name in ANTHROPIC_MODELS:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        elif model_name in GOOGLE_MODELS:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.gemini_model = genai.GenerativeModel(model_name)

        elif model_name in GLM_MODELS:
            from zai import ZaiClient
            self.client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

        # elif model_name in OPEN_SOURCE_MODELS:   # old — local HuggingFace Qwen (commented out)
        #     from transformers import AutoProcessor
        #     if model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
        #         from transformers import Qwen2_5_VLForConditionalGeneration
        #         self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #             model_name, torch_dtype="auto", device_map="auto"
        #         )
        #     else:  # Qwen/Qwen3-VL-32B-Instruct
        #         from transformers import Qwen3VLForConditionalGeneration
        #         self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        #             model_name, torch_dtype="auto", device_map="auto"
        #         )
        #     self.qwen_processor = AutoProcessor.from_pretrained(model_name)

        elif model_name in OCRPIPELINE_MODELS:
            if model_name == "monkeyocr":
                self.monkeyocr_url = os.getenv("MONKEYOCR_API_URL", "http://localhost:8000")
            elif model_name == "paddleocrv5":
                import paddle
                paddle.set_flags({"FLAGS_use_mkldnn": 0, "FLAGS_enable_pir_api": 0})
                from paddleocr import PaddleOCR
                self.paddleocr_client = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )

            elif model_name == "paddleocrv5-ppstructure":
                import paddle
                paddle.set_flags({"FLAGS_use_mkldnn": 0, "FLAGS_enable_pir_api": 0})
                from paddleocr import TableRecognitionPipelineV2
                self.ppstructure_client = TableRecognitionPipelineV2()

        else:
            all_models = (
                ["mineru", "deepseekocr"]
                + sorted(TOGETHER_MODELS | OPEN_SOURCE_MODELS | GPT_MODELS
                         | ANTHROPIC_MODELS | GOOGLE_MODELS | GLM_MODELS | OCRPIPELINE_MODELS)
            )
            raise ValueError(f"Unsupported model: {model_name!r}. Supported: {', '.join(all_models)}")

    def _is_base64(self, s):
        if not isinstance(s, str) or len(s) < 100:
            return False
        try:
            base64.b64decode(s.replace("data:image/png;base64,", ""), validate=True)
            return True
        except Exception:
            return False

    def _to_pil(self, image_path):
        if self._is_base64(image_path):
            data = base64.b64decode(image_path.replace("data:image/png;base64,", ""))
            return Image.open(io.BytesIO(data)).convert("RGB")
        return Image.open(image_path).convert("RGB")

    def _to_base64(self, image_path):
        if self._is_base64(image_path):
            return image_path.replace("data:image/png;base64,", "")
        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def draft(self, image_path, local_version=True):
        if self.model_name == "mineru":
            image = self._to_pil(image_path)
            blocks = self.mineru_client.two_step_extract(image)
            return "\n\n".join(b["content"] for b in blocks)

        elif self.model_name == "deepseekocr":
            if self._is_base64(image_path):
                img = self._to_pil(image_path)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                file_path = tmp.name
            else:
                file_path = image_path

            prompt = f"<image>\n<|grounding|>{PROMPT}"
            return self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=file_path,
                output_path=tempfile.gettempdir(),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True,
            )

        elif self.model_name in TOGETHER_MODELS:
            b64 = self._to_base64(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                temperature=0,
                max_tokens=4096,
            )
            return response.choices[0].message.content

        elif self.model_name in GPT_MODELS:
            b64 = self._to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            if self.model_name == "gpt-4o":
                response = self.client.chat.completions.create(
                    model="gpt-4o", messages=messages, temperature=0, max_tokens=2048
                )
            else:
                response = self.client.chat.completions.create(
                    model="gpt-5", messages=messages, max_completion_tokens=2048
                )
            return response.choices[0].message.content

        elif self.model_name in ANTHROPIC_MODELS:
            b64 = self._to_base64(image_path)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            return response.content[0].text

        elif self.model_name in GOOGLE_MODELS:
            img = self._to_pil(image_path)
            response = self.gemini_model.generate_content([img, PROMPT])
            return response.text

        elif self.model_name == "glm-ocr":
            b64 = self._to_base64(image_path)
            response = self.client.layout_parsing.create(
                model="glm-ocr",
                file=f"data:image/png;base64,{b64}",
            )
            return response.md_results

        elif self.model_name in GLM_MODELS:
            b64 = self._to_base64(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                temperature=0,
                max_tokens=4096,
            )
            return response.choices[0].message.content

        elif self.model_name in OCRPIPELINE_MODELS:
            if self.model_name == "monkeyocr":
                img = self._to_pil(image_path)  # decodes base64 → PIL (any source format)
                buf = io.BytesIO()
                img.save(buf, format="PNG")     # normalise to PNG for MonkeyOCR
                buf.seek(0)
                response = requests.post(
                    f"{self.monkeyocr_url}/ocr/text",
                    files={"file": ("image.png", buf, "image/png")},
                )
                response.raise_for_status()
                data = response.json()
                if not data.get("success"):
                    raise RuntimeError(f"MonkeyOCR error: {data.get('message')}")
                return data["content"]

            elif self.model_name == "paddleocrv5":
                import numpy as np
                img = self._to_pil(image_path)
                result = self.paddleocr_client.predict(np.array(img))
                lines = []
                for res in result:
                    lines.extend(res["rec_texts"])
                return "\n".join(lines)

            elif self.model_name == "paddleocrv5-table":
                import numpy as np
                img = self._to_pil(image_path)
                result = self.ppstructure_client.predict(np.array(img))
                parts = []
                for res in result:
                    for table in res.get("table_res_list", []):
                        parts.append(table["pred_html"])
                    ocr = res.get("overall_ocr_res", {})
                    for text in ocr.get("rec_texts", []):
                        parts.append(f"<p>{text}</p>")
                return f"<html><body>{''.join(parts)}</body></html>"
