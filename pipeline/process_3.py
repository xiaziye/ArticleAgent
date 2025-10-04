import json
import re
import argparse
from typing import Dict, Any, List
from tqdm import tqdm  # ← 新增导入：用于进度条
from slm_invoker import SLMInvoker


def stage3_process(process2_file: str, output_file: str, model_invoker: SLMInvoker,
                   max_new_tokens: int = 512, temperature: float = 0.3):
    """
    阶段3: 约束式关系三元组生成

    Args:
        process2_file: 第二阶段输出文件路径 (process2.json)
        output_file: 输出结果文件路径 (process3.json)
        model_invoker: SLM调用器实例
        max_new_tokens: 最大生成token数
        temperature: 生成温度
    """
    # 读取第二阶段处理结果
    with open(process2_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    results = []

    # 处理每篇论文 —— 用 tqdm 添加进度条
    for paper in tqdm(processed_data, desc="Generating relations", unit="paper"):
        paper_id = paper.get("id")
        title = paper.get("title")
        abstract = paper.get("abstract", "")

        # 获取第一阶段分割的三个部分
        segmented_text = paper.get("segmented_text", {})
        related_research = segmented_text.get("related_research", "")
        research_methods = segmented_text.get("research_methods", "")
        conclusions = segmented_text.get("conclusions", "")

        # 获取第二阶段提取的概念对
        concept_pairs = paper.get("concept_pairs", [])

        # 构建输入文本
        input_text = (
            f"<|related_research|>{related_research}<|/related_research|>"
            f"<|research_methods|>{research_methods}<|/research_methods|>"
            f"<|conclusions|>{conclusions}<|/conclusions|>"
            f"<|concept_pairs|>{','.join(concept_pairs)}<|/concept_pairs|>"
        )

        # 构建指令
        instruction = (
            "Convert the list of concept pairs (formatted as concept1|concept2,concept3|concept4,...) "
            "into a structured JSON array of relations. Each relation must be a two-element array: "
            "[\"conceptX\", \"conceptY\"]. Preserve the original order and casing of concepts. "
            "Do not add extra fields, explanations, or metadata. "
            "Enclose the final JSON array strictly within <|concept_relations|> and <|/concept_relations|>"
        )

        # 调用SLM模型
        response = model_invoker.call_model(
            introduction=instruction,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # 提取关系内容
        concept_relations = extract_concept_relations_content(response)

        # 构建结果（保持与之前阶段相同的结构，增加concept_relations字段）
        result = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "segmented_text": {
                "related_research": related_research,
                "research_methods": research_methods,
                "conclusions": conclusions
            },
            "concept_pairs": concept_pairs,
            "concept_relations": concept_relations
        }

        results.append(result)

    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_concept_relations_content(response: str) -> List[List[str]]:
    """
    从模型响应中提取概念关系内容

    Args:
        response: 模型响应文本

    Returns:
        提取到的概念关系列表，如果未找到则返回空列表
    """
    # 匹配 <|concept_relations|>...<|/concept_relations|> 内容
    pattern = r"<\|concept_relations\|>(.*?)<\|/concept_relations\|>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        content = match.group(1).strip()
        try:
            # 尝试解析JSON格式的数组
            relations = json.loads(content)
            # 确保是二维数组格式 [[str, str], ...]
            if isinstance(relations, list) and all(
                    isinstance(rel, list) and len(rel) == 2 and
                    all(isinstance(item, str) for item in rel) for rel in relations
            ):
                return relations
            else:
                return []
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回空列表
            return []
    else:
        # 如果未找到匹配，返回空列表
        return []


def main():
    """
    主函数，处理命令行参数并执行阶段3处理
    """
    parser = argparse.ArgumentParser(description='阶段3: 约束式关系三元组生成')
    parser.add_argument('--process2_file', type=str, required=True,
                        help='第二阶段输出文件路径 (如: process2.json)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调模型路径')
    parser.add_argument('--output_file', type=str, default='process3.json',
                        help='输出结果文件路径 (默认: process3.json)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成token数 (默认: 512)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='生成温度 (默认: 0.3)')

    args = parser.parse_args()

    # 初始化SLM调用器
    model_invoker = SLMInvoker(model_path=args.model_path)

    # 执行阶段3处理
    stage3_process(
        process2_file=args.process2_file,
        output_file=args.output_file,
        model_invoker=model_invoker,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print(f"阶段3处理完成！结果已保存至 {args.output_file}")


if __name__ == "__main__":
    main()