#!/usr/bin/env python3
"""Create a non-overwriting reinforcement-learning experiment record."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="directory name, for example 2026-08-06-go2-baseline")
    parser.add_argument("--title", required=True, help="Chinese or English experiment title")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments"),
        help="experiment root directory (default: experiments)",
    )
    return parser.parse_args()


def render(title: str) -> str:
    return f"""# {title}

## 问题

这次实验只想弄清楚什么？

## 假设

预期会看到什么，为什么？

## 环境与配置

- 证据级别：
- Isaac Lab 版本或提交：
- 任务名称：
- 并行环境数：
- 训练迭代数：
- 随机种子：
- 基线：
- 只改变的变量：

## 命令

```bash
# 可直接复现的完整命令
```

## 产物

- 日志位置：
- 检查点位置：
- 图表或视频位置：

## 观察与指标

记录关键数值、现象和异常，不只记录“成功”或“失败”。

## 结论

数据是否支持假设？结论适用于哪个证据级别？

## 未验证与下一步

哪些结论仍然不能从本实验得到？下次只继续验证什么？
"""


def main() -> None:
    args = parse_args()
    if not NAME_PATTERN.fullmatch(args.name):
        raise SystemExit("name 必须为 YYYY-MM-DD-short-name，且 short-name 只含小写字母、数字和连字符")

    target = args.root / args.name
    record = target / "README.md"
    if target.exists():
        raise SystemExit(f"拒绝覆盖已存在的实验目录：{target}")

    target.mkdir(parents=True)
    record.write_text(render(args.title), encoding="utf-8")
    print(record)


if __name__ == "__main__":
    main()
