"""五格走廊：自己连接第一版完整 PPO（状态 learning）。

问题：中间移动没有奖励，Actor 怎样通过后续反馈更经常到达右侧目标？
场景：0--1--2--3--4，从 2 出发；动作 0 左移、1 右移。到 0 奖励 -1，
到 4 奖励 +1 并终止，其余奖励 0。观察 [position / 2 - 1]，无物理单位。
目标：完成连续采样、GAE、裁剪更新，产生真实参数变化并独立评估。
这不是照抄正确动作标签。任务刻意允许“总向右”成功，不证明复杂导航能力。

只修改三个函数的 TODO：collect_rollout、compute_gae、update_ppo。
环境、网络结构、命令行、评估和保存由教师提供；不导入旧算法作为答案。
函数内给出了所需接口、形状、步骤和返回值。网络仅用仓库已有 PyTorch。

运行（仓库根目录，按需分阶段，不要求一次写完）：
  .venv/bin/python exercises/ppo_from_scratch/corridor_ppo.py --mode inspect
  .venv/bin/python exercises/ppo_from_scratch/corridor_ppo.py --mode sample
  .venv/bin/python exercises/ppo_from_scratch/corridor_ppo.py --mode train
  .venv/bin/python exercises/ppo_from_scratch/corridor_ppo.py --mode evaluate
inspect 只核对接口；sample 只验证采样，两者均不是训练完成。
未完成 TODO 会友好退出，不生成训练产物。

预算：CPU、单环境；seed=201；40 批，每批 128 步；每批 4 epoch，
minibatch=32；Adam lr=0.003，gamma=0.95，lambda=0.95，clip=0.2，
价值系数 0.5、熵系数 0.01。是本小任务固定起点，不是通用最优配置。
默认短训练预计秒到分钟级；实现前没有教师训练效果保证。
评估：冻结参数，按分布抽样，200 回合，独立 seed=1201；每局最多观察
64 步，未结束记 timeout，不视为环境自然失败；评估不参与学习。
产物：artifacts/ppo_corridor/checkpoint.pt（忽略提交），仅训练完成才保存。

完成依据：三个核心由学习者实现；采样记录正确、无跨回合 GAE 串接；
损失有限且确有参数更新；保存再加载能在同一评估协议复现结果。
期望训练后随机策略成功率较训练前显著改善，达到至少 90% 可作本任务
初步目标；达不到就根据轨迹和指标诊断，不改奖励或伪造评估来过关。
最终需要学习者实际运行结果。固定种子成功不证明一般 PPO 稳定性或机器人能力。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical


class ExerciseIncomplete(NotImplementedError):
    """仅用于尚未完成的学习者核心。"""


class Corridor:
    """教师提供的标准库环境；采样窗口不属于任务终止。"""

    def __init__(self):
        self.reset()

    def observation(self) -> list[float]:
        return [self.position / 2.0 - 1.0]

    def reset(self) -> list[float]:
        self.position = 2
        self.terminated = False
        return self.observation()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        if self.terminated:
            raise RuntimeError("本局已结束，需要 reset 后才能继续。")
        if type(action) is not int or action not in (0, 1):
            raise ValueError("动作须为 Python int：0 左、1 右。")
        self.position += -1 if action == 0 else 1
        self.terminated = self.position in (0, 4)
        reward = float(self.position == 4) - float(self.position == 0)
        return self.observation(), reward, self.terminated


class ActorCritic(nn.Module):
    """独立参数的两条支路；forward([N,1]) -> logits[N,2], values[N]。"""

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 2))
        self.critic = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
        # 固定最初左右等概率，便于比较；并非所有 PPO 初始化都如此。
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, observations):
        return self.actor(observations), self.critic(observations).squeeze(-1)


@dataclass
class Rollout:
    observations: torch.Tensor  # float32 [T,1]，动作前观察
    actions: torch.Tensor       # int64 [T]，0 或 1
    rewards: torch.Tensor       # float32 [T]，动作执行后的奖励
    terminated: torch.Tensor    # bool [T]，真正终止；不含采样窗口结束
    old_log_probs: torch.Tensor # float32 [T]，采样动作的旧对数概率
    old_values: torch.Tensor    # float32 [T]，动作前旧价值
    next_values: torch.Tensor   # float32 [T]，动作后观察的价值，终止时置 0


@torch.no_grad()
def collect_rollout(model: ActorCritic, env: Corridor, steps: int) -> Rollout:
    """TODO 1：连续推进 steps 次，返回 CPU 张量组成的 Rollout。

    env 在批次之间继续运行，不要在函数入口无条件 reset。
    每步读取 env.observation()，保存动作前观察；转 float32 [1,1] 输入 model。
    logits 构造 Categorical(logits=logits)，sample 得到动作张量；用其
    log_prob(action_tensor) 获取旧对数概率，.item() 仅用于环境动作和存档。
    env.step(int) 得到下一观察、reward、terminated。若未终止，下一观察
    经 model 得到 next_value；真正终止则 next_value=0，随后 reset 接下一局。
    切勿把 reset 后的价值当作上局尾部价值。最后一条未终止时保留其价值。
    每次收集包含观察、动作、奖励、结束标记、旧概率、旧价值、下一价值；
    字段必须同序，形状按 Rollout 注释。函数已有 no_grad，不更新参数。
    收集结束时 env 已处在下一次采样该接续的位置（或已重置的新局）。
    """
    raise ExerciseIncomplete("先实现 collect_rollout，再运行 --mode sample。")


def compute_gae(batch: Rollout, gamma: float, gae_lambda: float):
    """TODO 2：返回 (advantages, returns)，均 float32 [T]、不含计算图。

    old_values/next_values 是本批固定预测。先按单步奖励、下一价值和当前
    价值构造 delta；倒序汇总 GAE，终止处阻断与下一回合优势的连接。
    记录外没有更多误差，尾部 GAE 汇总从零开始；末条未终止时，它的
    next_value 仍参与 delta，不能一起清零。
    returns 用原始优势加同位置 old_values，不能用标准化优势构造。
    关系：delta[t] = rewards[t] + gamma*next_values[t] - old_values[t]；
    A[t] = delta[t] + gamma*gae_lambda*(1-terminated[t])*A[t+1]；
    returns[t] = A[t] + old_values[t]，记录外 A[T]=0。
    提示：torch.zeros_like(batch.rewards) 分配输出；reversed(range(T))
    访问原有索引，不要将最终返回顺序反过来。bool 掩码运算前转 float。
    """
    raise ExerciseIncomplete("采样完成后，再实现 compute_gae 的时间关系。")


def update_ppo(model, optimizer, batch, advantages, returns, *, epochs,
               minibatch_size, clip_epsilon, value_coef, entropy_coef):
    """TODO 3：真实更新参数，返回含 policy_loss/value_loss/entropy 的 float 字典。

    每个 epoch 用 torch.randperm(T) 产生共同索引，按 minibatch_size 分组。
    用该组 observations 重新前向；Categorical(logits=logits) 的
    log_prob(batch.actions[idx]) 是当前对“旧动作”的对数概率，不重新抽样。
    用 torch.exp(当前 log_prob - old_log_probs) 算比值；用固定优势构造
    PPO-Clip 策略损失（torch.clamp、torch.minimum，最后取负平均）。
    具体为 -mean(min(ratio*A, clamp(ratio,1-epsilon,1+epsilon)*A))。
    价值损失为当前 values 与固定 returns 的均方误差。entropy 为当前
    dist.entropy().mean()，总损失加入 value_coef*价值损失，减去
    entropy_coef*entropy。本版不做优势标准化、价值裁剪或 KL 提前停止。
    每小批次 optimizer.zero_grad()，loss.backward()，optimizer.step()。
    记录并返回各小批次指标的平均值；勿用 float/.item() 断开损失计算图。
    只日志值转 float；采样记录、advantages、returns 不能修改或重算。
    """
    raise ExerciseIncomplete("完成 GAE 后，再连接 update_ppo 的损失与参数更新。")


def validate_rollout(batch, steps):
    """接口诊断，不能代替学习者训练和轨迹复核。"""
    for name in batch.__dataclass_fields__:
        x = getattr(batch, name)
        expected = (steps, 1) if name == "observations" else (steps,)
        assert x.shape == expected, f"{name} 应为 {expected}，实际 {x.shape}"
        assert x.device.type == "cpu" and not x.requires_grad, name
        expected_dtype = torch.long if name == "actions" else (
            torch.bool if name == "terminated" else torch.float32)
        assert x.dtype == expected_dtype, f"{name} dtype 应为 {expected_dtype}"
        assert torch.isfinite(x).all(), f"{name} 出现非有限值"
    assert ((batch.actions == 0) | (batch.actions == 1)).all()
    assert (batch.next_values[batch.terminated] == 0).all(), "终止后的价值须为零"


@torch.no_grad()
def evaluate(model, episodes=200, seed=1201):
    """教师提供的独立冻结抽样评估；不消耗训练随机数状态。"""
    was_training = model.training
    wins = losses = timeouts = 0
    lengths = []
    model.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for _ in range(episodes):
                env = Corridor()
                for length in range(1, 65):
                    obs = torch.tensor([env.observation()], dtype=torch.float32)
                    logits, _ = model(obs)
                    action = int(Categorical(logits=logits).sample().item())
                    _, reward, terminated = env.step(action)
                    if terminated:
                        wins += int(reward > 0)
                        losses += int(reward < 0)
                        break
                else:
                    timeouts += 1
                lengths.append(length)
    finally:
        model.train(was_training)
    return dict(success_rate=wins / episodes, mean_reward=(wins-losses) / episodes,
                timeouts=timeouts, mean_steps=sum(lengths) / episodes)


def inspect(model):
    env = Corridor()
    with torch.no_grad():
        logits, value = model(torch.tensor([env.observation()]))
        print("起点观察:", env.observation(), "左右概率:", logits.softmax(-1).tolist())
        print("Critic 初始预测（未学习，不是真实回报）:", value.item())
    print("向右一步:", env.step(1))
    print("再向右一步:", env.step(1))
    print("这是接口演示，没有抽样训练或参数更新。三个 TODO 由你完成。")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["inspect", "sample", "train", "evaluate"],
                        default="inspect")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/ppo_corridor/checkpoint.pt"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(201)
    model = ActorCritic()
    if args.mode == "inspect":
        inspect(model)
        return
    if args.mode == "evaluate":
        if not args.checkpoint.exists():
            print("尚无训练检查点，请先完成 TODO 并运行 --mode train。")
            return
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        print("加载后的冻结评估:", evaluate(model))
        print("保存时的冻结评估:", saved["after"])
        return
    env = Corridor()
    try:
        if args.mode == "sample":
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            batch = collect_rollout(model, env, 128)
            validate_rollout(batch, 128)
            assert all(torch.equal(v, snapshot[k]) for k, v in model.state_dict().items())
            for i in range(8):
                print({k: getattr(batch, k)[i].tolist() for k in batch.__dataclass_fields__})
            print("采样接口检查通过；尚未训练，请核对时间对应与 reset 边界。")
            return
        before = evaluate(model)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        print("训练前冻结评估:", before)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        for iteration in range(40):
            batch = collect_rollout(model, env, 128)
            validate_rollout(batch, 128)
            advantages, returns = compute_gae(batch, 0.95, 0.95)
            for x in (advantages, returns):
                assert x.shape == (128,) and not x.requires_grad
                assert torch.isfinite(x).all(), "优势或回报目标出现非有限值"
            metrics = update_ppo(model, optimizer, batch, advantages, returns,
                epochs=4, minibatch_size=32, clip_epsilon=0.2,
                value_coef=0.5, entropy_coef=0.01)
            for key in ("policy_loss", "value_loss", "entropy"):
                assert torch.isfinite(torch.tensor(metrics[key])), key
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(f"完成采样批次 {iteration + 1}/40:", metrics)
        changes = {branch: max((v-initial[k]).abs().max().item()
                   for k,v in model.state_dict().items() if k.startswith(branch+"."))
                   for branch in ("actor", "critic")}
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert all(x > 0 for x in changes.values()), "Actor 和 Critic 都应有参数变化"
        after = evaluate(model)
        print("两支路最大参数变化:", changes, "训练后冻结评估:", after)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), before=before, after=after,
                        seed=201, batches=40, steps_per_batch=128), args.checkpoint)
        print("检查点已保存:", args.checkpoint)
    except ExerciseIncomplete as exc:
        print("练习尚未完成:", exc)
        print("本次没有生成训练检查点；接口说明与运行方式在本文件顶部。")


if __name__ == "__main__":
    main()
