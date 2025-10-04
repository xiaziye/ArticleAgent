# ArticleAgent: Constraint-Driven Small Language Models for Academic Concept Path Mining

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Hengzongshu/ArticleAgent)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/Hengzongshu/NSU-Academic-Concept-Paths)

> **Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers**  
> Ziye Xia, Sergei S. Ospichev (2025)

This repository implements a **four-stage agent framework** that combines **small language models (SLMs)** with the **OpenAlex knowledge graph** to extract structured **concept paths** from academic paper abstracts and identify **scientific innovation points**.

Our approach demonstrates that **rare structural combinations of mainstream concepts**—not just novel terms—are the primary source of academic novelty. By enforcing knowledge graph constraints, we achieve **97.24% precision** in end-to-end concept path extraction while mitigating LLM hallucination.

---

## 🌟 Key Features

- ✅ **Four-stage pipeline**: semantic segmentation → concept pair extraction → relation triplet generation → path refinement  
- ✅ **Knowledge-constrained generation**: all outputs aligned with OpenAlex concept taxonomy  
- ✅ **Innovation detection**: identifies papers with rare but meaningful concept paths  
- ✅ **Small-model efficiency**: fine-tuned **Qwen2.5-1.5B-Instruct** outperforms zero-shot LLMs by >10× in F1  
- ✅ **Human-in-the-loop validation**: expert-annotated dataset of 1,196 innovation points

---

## 📦 Resources

| Resource | Link |
|--------|------|
| **Paper** | [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(replace with your arXiv ID)* |
| **Fine-tuned Model** | [Hengzongshu/ArticleAgent](https://huggingface.co/Hengzongshu/ArticleAgent) |
| **Training Dataset** | [Hengzongshu/NSU-Academic-Concept-Paths](https://huggingface.co/datasets/Hengzongshu/NSU-Academic-Concept-Paths) |
| **Code** | This repository |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

---

### 2. Load the model
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Hengzongshu/ArticleAgent"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True
)

---

### 3. Run inference (Stage 2 example)
input_text = """<research_methods>... your abstract segment ...</research_methods>"""
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: [["Physics", "Superconductivity"], ["Materials Science", "High-Tc materials"]]

💡 For full pipeline usage, see examples/demo.ipynb.

---

📁 Project Structure
ArticleAgent/
├── data/                          # Data processing scripts
│   ├── import_openalex.py         # Import OpenAlex CSV to PostgreSQL
│   └── export_papers_to_json.py   # Export cleaned papers to JSON
├── config/                        # Training configurations
│   └── qwen2.5_finetune.yaml
├── pipeline/ 
│   ├── slm_invoker.py             # Core SLM invocation logic
│   └── process1-4.py                # Four-stage agent modules
├── train.py                       # Training script
├── requirements.txt               # Dependencies
├── scripts/                       # Utility scripts
└── examples/
    └── demo.ipynb                 # End-to-end inference demo

---

📄 Citation
If you use this work, please cite our paper:

@article{xia2025constraint,
  title={Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers},
  author={Xia, Ziye and Ospichev, Sergei S.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
