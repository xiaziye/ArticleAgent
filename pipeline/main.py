import subprocess
import sys
import argparse


def run_stage(stage_script: str, **kwargs):
    """
    执行指定阶段的脚本

    Args:
        stage_script: 阶段脚本文件名
        **kwargs: 传递给脚本的参数
    """
    cmd = ["python", stage_script]

    # 构建命令行参数
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(f"执行命令: {' '.join(cmd)}")

    # 执行脚本
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"阶段 {stage_script} 执行失败!")
        print(f"错误信息: {result.stderr}")
        sys.exit(1)
    else:
        print(f"阶段 {stage_script} 执行成功!")
        print(result.stdout)


def main():
    """
    主函数，依次执行四个阶段
    """
    parser = argparse.ArgumentParser(description='完整Agent流程执行器')

    # 公共参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='微调模型路径')
    parser.add_argument('--data_file', type=str, default='test.json',
                        help='初始数据文件路径 (默认: test.json)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成token数 (默认: 512)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='生成温度 (默认: 0.3)')

    # 阶段特定文件路径
    parser.add_argument('--process1_output', type=str, default='process1.json',
                        help='阶段1输出文件 (默认: process1.json)')
    parser.add_argument('--process2_output', type=str, default='process2.json',
                        help='阶段2输出文件 (默认: process2.json)')
    parser.add_argument('--process3_output', type=str, default='process3.json',
                        help='阶段3输出文件 (默认: process3.json)')
    parser.add_argument('--process4_output', type=str, default='process4.json',
                        help='阶段4输出文件 (默认: process4.json)')
    parser.add_argument('--concept_classification_file', type=str, required=True,
                        help='概念分类训练文件路径')

    args = parser.parse_args()

    print("开始执行完整Agent流程...")

    # 阶段1: 结构化语义分割
    print("\n=== 执行阶段1: 结构化语义分割 ===")
    run_stage(
        "process_1.py",
        data_file=args.data_file,
        model_path=args.model_path,
        output_file=args.process1_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # 阶段2: 关键概念对提取与专家校验
    print("\n=== 执行阶段2: 关键概念对提取与专家校验 ===")
    run_stage(
        "process_2.py",
        process1_file=args.process1_output,
        model_path=args.model_path,
        output_file=args.process2_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # 阶段3: 约束式关系三元组生成
    print("\n=== 执行阶段3: 约束式关系三元组生成 ===")
    run_stage(
        "process_3.py",
        process2_file=args.process2_output,
        model_path=args.model_path,
        output_file=args.process3_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # 阶段4: 层级校验与路径优化
    print("\n=== 执行阶段4: 层级校验与路径优化 ===")
    run_stage(
        "process_4.py",
        concept_classification_file=args.concept_classification_file,
        process3_file=args.process3_output,
        model_path=args.model_path,
        output_file=args.process4_output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    print(f"\n=== 完整流程执行完成! ===")
    print(f"阶段1输出: {args.process1_output}")
    print(f"阶段2输出: {args.process2_output}")
    print(f"阶段3输出: {args.process3_output}")
    print(f"阶段4输出: {args.process4_output}")


if __name__ == "__main__":
    main()