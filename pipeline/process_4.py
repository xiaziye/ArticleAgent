import json
import re
import argparse
from typing import Dict, Any, List
from tqdm import tqdm
from slm_invoker import SLMInvoker


def stage4_process(test_file: str, output_file: str, model_invoker: SLMInvoker,
                   max_new_tokens: int = 512, temperature: float = 0.3):
    """
    阶段4: 基于 test.json 的概念相关性重标注（模型输出需用 <|concept_lable|> 包裹）
    """
    # 读取 test.json
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []

    # 指令：明确要求模型用 <|concept_lable|> 包裹输出
    instruction = (
        "You are a scientific paper data analyst. Given an abstract and a list of candidate concepts, "
        "determine for each concept whether it is scientifically relevant to the paper (label 0) or not (label 1). "
        "A concept is relevant (0) only if it is explicitly discussed or directly implied in the abstract. "
        "Output the result as a JSON array of [concept, label] pairs in the same order as the input candidate concepts, "
        "strictly enclosed within <|concept_lable|> and <|/concept_lable|>."
    )

    # 处理每条测试数据
    for item in tqdm(test_data, desc="Re-labeling concepts", unit="paper"):
        paper_id = item.get("id")
        abstract = item.get("abstract", "")
        original_concept_labels = item.get("concept_lable", [])

        # 提取候选概念（只取名称）
        candidate_concepts = [concept for concept, _ in original_concept_labels]
        concepts_str = ",".join(candidate_concepts)

        # 构建 input_text
        input_text = (
            f"<|abstract|>{abstract}<|/abstract|>"
            f"<|possible_concepts|>{concepts_str}<|/possible_concepts|>"
        )

        # 调用模型
        response = model_invoker.call_model(
            introduction=instruction,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # 从 <|concept_lable|>...<|/concept_lable|> 中提取内容
        new_concept_labels = extract_concept_labels_from_response(response)

        # 构建结果
        result = {
            "id": paper_id,
            "abstract": abstract,
            "original_concept_lable": original_concept_labels,
            "revised_concept_lable": new_concept_labels
        }

        results.append(result)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_concept_labels_from_response(response: str) -> List[List]:
    """
    从响应中提取 <|concept_lable|>...<|/concept_lable|> 之间的内容，并解析为 [[concept, label], ...]
    """
    # 正则匹配标签内的内容（注意：标签名是 concept_lable，不是 label）
    pattern = r"<\|concept_lable\|>(.*?)<\|/concept_lable\|>"
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return []

    content = match.group(1).strip()

    # 尝试解析 JSON
    try:
        data = json.loads(content)
        if isinstance(data, list) and all(
            isinstance(item, list) and len(item) == 2 and
            isinstance(item[0], str) and isinstance(item[1], int)
            for item in data
        ):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 如果 JSON 解析失败，尝试用正则提取 [concept, digit] 模式（备用）
    fallback_pattern = r'\[\s*"([^"]*)"\s*,\s*(0|1)\s*\]'
    matches = re.findall(fallback_pattern, content)
    if matches:
        return [[concept, int(label)] for concept, label in matches]

    return []


def main():
    parser = argparse.ArgumentParser(description='阶段4: 概念相关性重标注（输出需包裹在 <|concept_lable|> 中）')
    parser.add_argument('--test_file', type=str, required=True,
                        help='测试文件路径 (如: test.json)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调模型路径')
    parser.add_argument('--output_file', type=str, default='process4.json',
                        help='输出结果文件路径 (默认: process4.json)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成token数 (默认: 512)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='生成温度 (默认: 0.3)')

    args = parser.parse_args()

    model_invoker = SLMInvoker(model_path=args.model_path)

    stage4_process(
        test_file=args.test_file,
        output_file=args.output_file,
        model_invoker=model_invoker,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print(f"阶段4处理完成！结果已保存至 {args.output_file}")


if __name__ == "__main__":
    main()