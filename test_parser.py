import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    构建一个用于测试参数传递的命令行解析器。

    用法示例：
      python test_parser.py task1 --model gpt-4 --lr 1e-4 --epochs 5 --use-cuda
    """
    parser = argparse.ArgumentParser(
        description="参数传递测试脚本（TypoAgent 项目示例）"
    )

    # 必选位置参数：例如任务名称
    parser.add_argument(
        "task",
        type=str,
        help="任务名称，例如：train / eval / debug 等",
    )

    # 一些常见可选参数
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="模型名称，默认：gpt-4",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="学习率，默认：1e-4",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数，默认：3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认：42",
    )

    # bool 类型开关
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="是否使用 CUDA / GPU（加上该参数即为 True，不加为 False）",
    )

    # 列表参数示例：多数据集 / 多文件
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="数据集名称列表，例如：--datasets ds1 ds2 ds3",
    )

    # 以 key=value 形式传递的额外配置
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="额外配置，例如：--extra a=1 b=2",
    )

    return parser


def parse_extra(extra_args):
    """将 ['a=1', 'b=2'] 解析成 dict。"""
    extra_dict = {}
    for item in extra_args:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        extra_dict[k] = v
    return extra_dict


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 将 Namespace 转成 dict，方便打印和调试
    args_dict = vars(args).copy()
    args_dict["extra"] = parse_extra(args.extra)

    print("====== 解析后的参数 ======")
    for k, v in args_dict.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

