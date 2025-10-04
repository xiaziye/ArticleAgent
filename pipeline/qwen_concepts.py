import json
import time
import requests
import re


# 配置
SELECTED_PAPERS_FILE = "selected_100_papers.json"
OUTPUT_REPORT = "qwen_max_concepts.json"

# Qwen-Max (DashScope) 配置
QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
MODEL_NAME = "qwen3-max"

# 从 config.json 读取 DashScope API Key
with open("config.json") as f:
    config = json.load(f)
    DASHSCOPE_API_KEY = config["dashscope_api_key"]  


def extract_concepts_from_output(text: str):
    """从 <|concepts|>...<|/concepts|> 中提取逗号分隔的概念"""
    match = re.search(r"<\|concepts\|>\s*(.*?)\s*<\|/concepts\|>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    content = match.group(1).strip()
    if not content:
        return []
    concepts = [c.strip() for c in content.split(",")]
    return [c for c in concepts if c]


def call_qwen_max(abstract: str):
    prompt = f"""You are an expert in scientific concept extraction.
Given the following paper abstract, extract key concepts from OpenAlex.
Output ONLY a JSON list of concepts, like: <|concepts|>concept1, concept2, concept3<|/concepts|>
Do not include explanations, numbers, or extra text.

Abstract:
{abstract}
"""

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 500
        }
    }

    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        output_text = data["output"]["choices"][0]["message"]["content"]
        return extract_concepts_from_output(output_text)
    except Exception as e:
        print(f"⚠️ Qwen API error: {e}")
        return []


def main():
    # 加载论文
    with open(SELECTED_PAPERS_FILE, encoding="utf-8") as f:
        papers = json.load(f)
    if isinstance(papers, dict):
        papers = [papers]

    results = []
    total = len(papers)
    for i, paper in enumerate(papers):
        pid = paper.get("id")
        abstract = paper.get("abstract")
        if not pid or not abstract:
            continue

        print(f"[{i+1}/{total}] Processing {pid}...")
        concepts = call_qwen_max(abstract)
        results.append({"id": pid, "concepts": concepts})
        time.sleep(1)  # 避免触发 QPS 限制

    # 保存结果
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！结果已保存至 {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()