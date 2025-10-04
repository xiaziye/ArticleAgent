import json
import time
import requests
import re

# 配置
SELECTED_PAPERS_FILE = "selected_100_papers.json"
OUTPUT_REPORT = "deepseek_concepts.json"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  
MODEL_NAME = "deepseek-chat"

# 读取 API Key
with open("config.json") as f:
    API_KEY = json.load(f)["deepseek_api_key"]


def extract_concepts_from_output(text: str):
    """从 <|concepts|>...<|/concepts|> 中提取逗号分隔的概念列表"""
    print(f"🔍 模型原始输出:\n{text}\n{'-' * 50}")

    match = re.search(r"<\|concepts\|>\s*(.*?)\s*<\|/concepts\|>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        print("⚠️ 未检测到 <|concepts|>...<|/concepts|> 标签！")
        return []

    content = match.group(1).strip()
    print(f"📦 提取到标签内内容: '{content}'")

    if not content:
        print("⚠️ 标签内内容为空")
        return []

    concepts = [c.strip() for c in content.split(",")]
    concepts = [c for c in concepts if c]
    print(f"✅ 解析出概念: {concepts}")
    return concepts


def call_deepseek(abstract: str):
    prompt = f"""You are an expert in scientific concept extraction.
Given the following paper abstract, extract key concept form OpenAlex.
Output ONLY a JSON list of concepts, like: <|concepts|>concept1, concept2, concept3<|/concepts|>
Do not include explanations, numbers, or extra text.

Abstract:
{abstract}
"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500
    }

    try:
        print("\n📡 正在调用 DeepSeek API...")
        # 可选：打印摘要前 100 字（避免泄露全文）
        print(f"📄 摘要预览: {abstract[:100]}...")

        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)

        print(f"📡 HTTP 状态码: {resp.status_code}")
        if resp.status_code != 200:
            print(f"❌ API 返回错误: {resp.text}")
            return []

        data = resp.json()
        output_text = data["choices"][0]["message"]["content"]

        # 调试：打印原始模型输出
        print(f"\n🤖 模型完整回复:\n{output_text}\n{'=' * 60}")

        return extract_concepts_from_output(output_text)

    except Exception as e:
        print(f"💥 调用 DeepSeek 时发生异常: {e}")
        return []


def main():
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
            print(f"⚠️ 跳过无效论文 (id={pid})")
            continue

        print(f"\n[{i + 1}/{total}] 处理论文: {pid}")
        concepts = call_deepseek(abstract)
        results.append({"id": pid, "concepts": concepts})
        time.sleep(1)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！结果已保存至 {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()