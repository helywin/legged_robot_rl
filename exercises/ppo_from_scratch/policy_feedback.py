"""暂缓的后续材料：两动作任务的抽样与策略更新。

2026-09-14：学习者要求先讲 PPO 基础。本文件不是当前作业，
无需补 TODO 或运行；当前请从课程/200-policy-feedback.md 开始。

场景：每回合只有一步，观察恒为 [1.0]。动作 0（左）奖励 -1，
动作 1（右）奖励 +1。环境只依赖标准库；网络用仓库已有 PyTorch。
目标：通过实际抽样获得反馈，让有利动作的概率逐步增大。

只修改 sample_action 与 update_policy 两个 TODO；不要改奖励或初始参数。
这两个函数组成同一训练模块，不另拆课程。

接口：
- logits：形状 [2] 的浮点张量，是网络输出的两个原始偏好分数，不是 Q 值。
- sample_action(logits) -> (action, log_prob)：action 是 Python int 0/1；
  log_prob 是该次抽中动作的对数概率，零维 Tensor，必须保留计算图。
  可用 torch.distributions.Categorical(logits=logits)，分布的 sample()
  返回整数 Tensor，log_prob(action_tensor) 返回对数概率；.item() 只用于
  把动作交给环境，不用于把 log_prob 转成数值。不能用 argmax 代替抽样。
- update_policy(optimizer, log_prob, reward) -> float：reward 是环境返回的
  Python float；以 -reward * log_prob 为损失，清旧梯度、反传、更新，
  返回用于打印的损失数值。optimizer 已绑定网络参数，不需要重新创建。

运行（仓库根目录）：
  .venv/bin/python exercises/ppo_from_scratch/policy_feedback.py --mode check
  .venv/bin/python exercises/ppo_from_scratch/policy_feedback.py --mode train

先预测：check 固定右动作、奖励 +1；初始概率各 0.5，学习率 0.2。
成功条件：check 的两个权重从 [0,0] 变为约 [-0.1,0.1]，
梯度约 [0.5,-0.5]，右概率约 0.549834；train 确有 100 次参数更新，
比较训练前后右概率和冻结参数下 1000 次抽样的右动作频率，解释是否一致。
频率不要求精确等于概率；若方向异常，检查动作与 log_prob 是否匹配及损失符号。
预算：CPU、一个单步环境、训练 seed=98、100 回合/更新，SGD lr=0.2；
冻结抽样各 1000 次使用独立 seed=1098。预期秒级，无仿真和模型文件。
这是单步策略学习，尚未验证多步反馈、完整 PPO 或机器人控制。
"""
from __future__ import annotations

import argparse
import torch


def reward_for(action: int) -> float:
    """教师提供的单步环境；动作以外的信息不进入策略。"""
    if action not in (0, 1):
        raise ValueError("动作必须是 0 或 1")
    return -1.0 if action == 0 else 1.0


def sample_action(logits: torch.Tensor) -> tuple[int, torch.Tensor]:
    # TODO：由当前输出创建分布，抽一个动作，返回动作及它的对数概率。
    raise NotImplementedError("请实现 sample_action：保留抽中动作的 log_prob 计算图。")


def update_policy(optimizer, log_prob: torch.Tensor, reward: float) -> float:
    # TODO：由反馈构造损失，并完成一次实际参数更新。
    raise NotImplementedError("请实现 update_policy：反馈 → 损失 → 梯度 → 参数更新。")


def make_policy():
    # 单个线性层，不设偏置；输入恒为 1，因此输出恰好等于两个权重。
    policy = torch.nn.Linear(1, 2, bias=False)
    torch.nn.init.zeros_(policy.weight)
    return policy


def probabilities(policy):
    with torch.no_grad():
        return torch.softmax(policy(torch.tensor([1.0])), dim=-1).tolist()


def frozen_frequency(policy):
    # 独立随机数流，评估不会消耗训练的随机状态或修改参数。
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(1098)
        logits = policy(torch.tensor([1.0]))
        return sum(sample_action(logits)[0] for _ in range(1000)) / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "train"), default="check")
    args = parser.parse_args()
    torch.manual_seed(98)
    policy = make_policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    print("初始权重：", policy.weight.detach().flatten().tolist())
    print("初始概率：", probabilities(policy))
    try:
        if args.mode == "check":
            # 固定数据用于核对更新；真正抽样在 train 模式由学习者实现。
            dist = torch.distributions.Categorical(logits=policy(torch.tensor([1.0])))
            log_prob = dist.log_prob(torch.tensor(1))
            loss = update_policy(optimizer, log_prob, reward_for(1))
            print("loss：", loss)
            print("梯度：", policy.weight.grad.flatten().tolist())
            print("更新后权重：", policy.weight.detach().flatten().tolist())
            print("更新后概率：", probabilities(policy))
        else:
            print("训练前冻结抽样右频率：", frozen_frequency(policy))
            total = 0.0
            for step in range(1, 101):
                action, log_prob = sample_action(policy(torch.tensor([1.0])))
                reward = reward_for(action)
                loss = update_policy(optimizer, log_prob, reward)
                total += reward
                if step in (1, 10, 25, 50, 100):
                    print(f"step={step} action={action} reward={reward:+.0f} "
                          f"loss={loss:.6f} p_right={probabilities(policy)[1]:.6f} "
                          f"mean_reward={total / step:.3f}")
            print("最终权重：", policy.weight.detach().flatten().tolist())
            print("训练后冻结抽样右频率：", frozen_frequency(policy))
    except NotImplementedError as error:
        print(f"练习尚未完成：{error}")
        print("请阅读文件顶部题目和课程 200，补全两个 TODO 后重跑。")


if __name__ == "__main__":
    main()
