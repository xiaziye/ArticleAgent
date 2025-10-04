import json
import re
import argparse
from typing import Dict, Any, List
from tqdm import tqdm  # ← 新增导入：用于进度条
from slm_invoker import SLMInvoker


def stage2_process(process1_file: str, output_file: str, model_invoker: SLMInvoker,
                   max_new_tokens: int = 512, temperature: float = 0.3):
    """
    阶段2: 关键概念对提取与专家校验

    Args:
        process1_file: 第一阶段输出文件路径 (process1.json)
        output_file: 输出结果文件路径 (process2.json)
        model_invoker: SLM调用器实例
        max_new_tokens: 最大生成token数
        temperature: 生成温度
    """
    # 读取第一阶段处理结果
    with open(process1_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    results = []

    # 处理每篇论文 —— 用 tqdm 添加进度条
    for paper in tqdm(processed_data, desc="Extracting concept pairs", unit="paper"):
        paper_id = paper.get("id")
        title = paper.get("title")
        abstract = paper.get("abstract", "")

        # 获取第一阶段分割的三个部分
        segmented_text = paper.get("segmented_text", {})
        related_research = segmented_text.get("related_research", "")
        research_methods = segmented_text.get("research_methods", "")
        conclusions = segmented_text.get("conclusions", "")

        # 构建输入文本
        input_text = (
            f"<|related_research|>{related_research}<|/related_research|>"
            f"<|research_methods|>{research_methods}<|/research_methods|>"
            f"<|conclusions|>{conclusions}<|/conclusions|>"
        )

        # 构建指令
        instruction = (
            "From the segmented abstract content (related_research, research_methods, and conclusions), "
            "identify meaningful concept pairs that represent semantic or functional associations "
            "(e.g., gene–pathway, stress–behavior, protein–receptor). "
            "Format each pair as 'conceptA|conceptB', with pairs separated by commas. "
            "Only include pairs that are explicitly or strongly implied in the text. "
            "Wrap the entire output with <|concept_pairs|> at the beginning and <|/concept_pairs|> at the end."
        )

        # 调用SLM模型
        response = model_invoker.call_model(
            introduction=instruction,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # 提取概念对内容
        concept_pairs = extract_concept_pairs_content(response)

        # 构建结果（保持与process1相同的结构，增加concept_pairs字段）
        result = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "segmented_text": {
                "related_research": related_research,
                "research_methods": research_methods,
                "conclusions": conclusions
            },
            "concept_pairs": concept_pairs
        }

        results.append(result)

    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_concept_pairs_content(response: str) -> List[str]:
    """
    从模型响应中提取概念对内容

    Args:
        response: 模型响应文本

    Returns:
        提取到的概念对列表，如果未找到则返回空列表
    """
    # 匹配 <|concept_pairs|>...<|/concept_pairs|> 内容
    pattern = r"<\|concept_pairs\|>(.*?)<\|/concept_pairs\|>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        content = match.group(1).strip()
        # 按逗号分割概念对，并去除空白
        pairs = [pair.strip() for pair in content.split(',') if pair.strip()]
        return pairs
    else:
        # 如果未找到匹配，返回空列表
        return []


def main():
    """
    主函数，处理命令行参数并执行阶段2处理
    """
    parser = argparse.ArgumentParser(description='阶段2: 关键概念对提取与专家校验')
    parser.add_argument('--process1_file', type=str, required=True,
                        help='第一阶段输出文件路径 (如: process1.json)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调模型路径')
    parser.add_argument('--output_file', type=str, default='process2.json',
                        help='输出结果文件路径 (默认: process2.json)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成token数 (默认: 512)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='生成温度 (默认: 0.3)')

    args = parser.parse_args()

    # 初始化SLM调用器
    model_invoker = SLMInvoker(model_path=args.model_path)

    # 执行阶段2处理
    stage2_process(
        process1_file=args.process1_file,
        output_file=args.output_file,
        model_invoker=model_invoker,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print(f"阶段2处理完成！结果已保存至 {args.output_file}")


if __name__ == "__main__":
    main()