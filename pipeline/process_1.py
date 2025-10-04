import json
import re
import argparse
from tqdm import tqdm  # ← 新增导入
from slm_invoker import SLMInvoker


def stage1_process(data_file: str, output_file: str, model_invoker: SLMInvoker,
                   max_new_tokens: int = 512, temperature: float = 0.3):
    """
    阶段1: 结构化语义分割处理函数

    Args:
        data_file: 输入数据文件路径 (data.json)
        output_file: 输出结果文件路径 (process1.json)
        model_invoker: SLM调用器实例
        max_new_tokens: 最大生成token数
        temperature: 生成温度
    """
    # 读取输入数据
    with open(data_file, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)

    results = []

    # 处理每篇论文 —— 用 tqdm 包装循环，添加进度条
    for paper in tqdm(papers_data, desc="Processing", unit="paper"):
        paper_id = paper.get("id")
        title = paper.get("title")
        abstract = paper.get("abstract", "")

        # 构建指令和输入
        instruction = (
            "Split the provided abstract into three distinct scientific sections: "
            "(1) related_research (background and prior work), "
            "(2) research_methods (experimental design, subjects, procedures, or analytical approaches), and "
            "(3) conclusions (key findings, implications, or final statements). "
            "Each section must be non-overlapping and derived solely from the abstract text. "
            "Enclose each part with its corresponding XML-style tag: "
            "<|related_research|>...<|/related_research|>, "
            "<|research_methods|>...<|/research_methods|>, and "
            "<|conclusions|>...<|/conclusions|>."
        )

        input_text = abstract

        # 调用SLM模型
        response = model_invoker.call_model(
            introduction=instruction,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # 解析模型输出，提取三个部分的内容
        related_research = extract_content(response, "related_research")
        research_methods = extract_content(response, "research_methods")
        conclusions = extract_content(response, "conclusions")

        # 构建结果
        result = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "segmented_text": {
                "related_research": related_research,
                "research_methods": research_methods,
                "conclusions": conclusions
            }
        }

        results.append(result)

    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_content(response: str, tag: str) -> str:
    """
    从模型响应中提取指定标签的内容

    Args:
        response: 模型响应文本
        tag: 要提取的标签名

    Returns:
        提取到的内容，如果未找到则返回空字符串
    """
    # 构建正则表达式模式，匹配 <|tag|>content<|/tag|> 格式
    # 使用非贪婪匹配确保正确提取内容
    pattern = f"<\\|{tag}\\|>(.*?)<\\|/{tag}\\|>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        # 返回匹配内容并去除首尾空白
        return match.group(1).strip()
    else:
        # 如果未找到匹配，返回空字符串
        return ""


def main():
    """
    主函数，处理命令行参数并执行阶段1处理
    """
    parser = argparse.ArgumentParser(description='阶段1: 结构化语义分割')
    parser.add_argument('--data_file', type=str, required=True,
                        help='输入数据文件路径 (如: data.json)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调模型路径')
    parser.add_argument('--output_file', type=str, default='process1.json',
                        help='输出结果文件路径 (默认: process1.json)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成token数 (默认: 512)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='生成温度 (默认: 0.3)')

    args = parser.parse_args()

    # 初始化SLM调用器
    model_invoker = SLMInvoker(model_path=args.model_path)

    # 执行阶段1处理
    stage1_process(
        data_file=args.data_file,
        output_file=args.output_file,
        model_invoker=model_invoker,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print(f"阶段1处理完成！结果已保存至 {args.output_file}")


if __name__ == "__main__":
    main()