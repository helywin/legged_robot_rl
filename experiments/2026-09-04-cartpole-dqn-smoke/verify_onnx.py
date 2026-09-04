#!/usr/bin/env python3
"""验证 ONNX 结构，并与同一检查点的 PyTorch 输出逐项比较。

从仓库根目录运行：

    .venv/bin/python experiments/2026-09-04-cartpole-dqn-smoke/verify_onnx.py

这不是训练测试。它只回答：导出后，同一批观察是否得到几乎相同的 Q 值。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from learner_train import CartPoleQNetwork


ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "cartpole-dqn-from-scratch"
)
CHECKPOINT_PATH = ARTIFACT_DIRECTORY / "online-network.pt"
ONNX_PATH = ARTIFACT_DIRECTORY / "online-network.onnx"


def main() -> None:
    missing_paths = [
        path for path in (CHECKPOINT_PATH, ONNX_PATH) if not path.exists()
    ]
    if missing_paths:
        print("缺少训练产物，请先完成并运行 learner_train.py：")
        for path in missing_paths:
            print(f"- {path}")
        return

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    pytorch_network = CartPoleQNetwork()
    pytorch_network.load_state_dict(
        checkpoint["online_network_state_dict"]
    )
    pytorch_network.eval()

    observations = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, -0.1, 0.3, 0.0],
            [-0.4, 0.5, -0.2, 0.1],
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        pytorch_q_values = pytorch_network(
            torch.from_numpy(observations)
        ).numpy()

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    onnx_q_values = session.run(
        ["q_values"],
        {"observation": observations},
    )[0]

    maximum_absolute_difference = float(
        np.max(np.abs(pytorch_q_values - onnx_q_values))
    )
    outputs_match = np.allclose(
        pytorch_q_values,
        onnx_q_values,
        rtol=1e-5,
        atol=1e-6,
    )

    print("ONNX 结构检查：PASS")
    print("输入：observation float32[batch_size, 4]")
    print("输出：q_values float32[batch_size, 2]")
    print("本次动态 batch：3")
    print(f"PyTorch Q 值：\n{pytorch_q_values}")
    print(f"ONNX Runtime Q 值：\n{onnx_q_values}")
    print(f"最大绝对误差：{maximum_absolute_difference:.8g}")
    print(f"数值对照：{'PASS' if outputs_match else 'FAIL'}")

    if not outputs_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
