"""090第二段：实现一批经验的DQN更新（学习者实现）。

本次场景：Task能够生成经验，现在让网络真的学习。先实现完整update与目标同步；
动作探索和经验回放在接入训练循环时继续完成，不复制旧实现。

已有接口：Agent(online, target, optimizer, gamma=0.99)。三个对象由调用者创建，
online与target必须结构相同且为不同对象；optimizer管理online.parameters()。
首版使用CPU。实际网络是network.py的QNetwork；检查器会注入可手算的线性网络。

只修改update和sync_target的TODO；可添加小型辅助函数。不要修改检查器。
运行（仓库根）：.venv/bin/python exercises/chromium_dqn/check_update.py

update(batch)输入Batch，六个张量均已准备好，无需从游戏读取或reset：
 observations / next_observations：float32，[B, observation_size]。
 actions：int64，[B]，每条经验实际执行的动作索引。
 rewards：float32，[B]，来自Task；不是累计游戏总分。
 terminated / truncated：bool，[B]，按任务语义保留。

本版计算契约：
 1. online(observations)得到[B,A]，每行只取actions指定的那个Q值，结果[B]。
    可用gather(1, actions.unsqueeze(1)).squeeze(1)：unsqueeze把[B]变[B,1]，
    gather按每行指定列读取，squeeze(1)只去掉列维，B=1时也保留批次。
 2. 在torch.no_grad()内：target(next_observations)得到[B,A]，沿动作维
    max(dim=1).values得到[B]。目标=reward + gamma * 下一状态最大Q * 非自然终止。
    terminated为True时屏蔽未来值；只有truncated时仍保留未来值。
    原因：本版truncated只是采集上限，不表示游戏未来价值真的为零。
 3. 本版loss使用F.mse_loss(predictions, targets)，默认取批次平方误差的平均。
 4. 每次先optimizer.zero_grad()清掉上一次梯度，再loss.backward()写新.grad，
    最后optimizer.step()改变在线网络参数。每次update只更新一次。
 5. 不更新目标网络；只有显式sync_target()才复制在线网络参数。
 6. 返回UpdateStats(loss, mean_prediction, mean_target, updates)，三个数值为
    本次更新前计算得到的Python float，updates为成功更新后的累计次数。
    .item()用于日志提取数值；不能先把loss转成float再调用backward。

sync_target()：将online.state_dict()复制到target，不得写成target=online，
否则两个变量指向同一网络，失去独立目标。使用target.load_state_dict(...)。

不要在update中自动reset、添加经验、改变探索率或同步目标；这些由训练流程调度。
数据流：online参数→预测→loss→backward→参数.grad→optimizer→新参数。
目标计算不进入求导图，target参数不获得梯度。

手算用例（检查器会执行你的update，不提供教师update答案）：
 B=2，观察分别[1,0]与[0,1]，执行动作1与2。在线预测分别0.5、0.2。
 两条奖励0.2、0.4；下一状态的最大目标Q分别0.8、0.7；gamma=0.9。
 第一条只截断，第二条自然终止。目标0.92、0.4，误差-0.42、-0.2，
 平均平方损失0.1082。诊断网络为无bias线性层，SGD学习率0.1，更新后
 对应权重0.5→0.542、0.2→0.22；目标网络和其他权重保持不变。
 真正的多层网络共享隐藏参数，更新后其他动作的输出也可能间接改变。

成功条件：手算一致、只有在线参数学习、连续更新不累积旧梯度、batch=1可运行、
目标显式同步生效。通过表示更新机制成立，不是游戏策略已训练成功。
"""
from dataclasses import dataclass
import random
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn import functional as F


@dataclass(frozen=True)
class Batch:
    observations: torch.Tensor  # float32，[B, observation_size]
    actions: torch.Tensor  # int64，[B]
    rewards: torch.Tensor  # float32，[B]
    next_observations: torch.Tensor  # float32，[B, observation_size]
    terminated: torch.Tensor  # bool，[B]
    truncated: torch.Tensor  # bool，[B]


@dataclass(frozen=True)
class UpdateStats:
    loss: float
    mean_prediction: float
    mean_target: float
    updates: int


class Agent:
    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        optimizer: Optimizer,
        gamma: float = 0.99,
        exploration_seed: int = 11,
    ) -> None:
        if online is target:
            raise ValueError('online与target必须是独立网络')
        if not 0 <= gamma <= 1:
            raise ValueError('gamma应在0到1之间')
        self.online: nn.Module = online
        self.target: nn.Module = target
        self.optimizer: Optimizer = optimizer
        self.gamma: float = gamma
        self.updates: int = 0
        self._exploration_rng: random.Random = random.Random(exploration_seed)
        # 目标的初始权重由调用者准备；这里不覆盖检查器给定的小数据。
        self.target.requires_grad_(False)
        self.target.eval()

    def update(self, batch: Batch) -> UpdateStats:
        q_values: torch.Tensor = self.online(batch.observations)
        selected_q_values = q_values.gather(dim=1,index=batch.actions.unsqueeze(dim=1)).squeeze(dim=1)

        with torch.no_grad():
            target_q: torch.Tensor = self.target(batch.next_observations)
            future_mask = (~batch.terminated).to(dtype=torch.float32)
            target_best_q = batch.rewards + target_q.max(dim=1).values * future_mask * self.gamma

        losses = F.mse_loss(target_best_q, selected_q_values)
        mean_loss = losses.mean()
        self.optimizer.zero_grad()
        mean_loss.backward()
        self.optimizer.step()
        self.updates += 1

        return UpdateStats(
            mean_loss.item(),
            selected_q_values.mean().item(),
            target_best_q.mean().item(),
            self.updates)

        

    def act(self, observation: list[float], epsilon: float) -> int:
        """根据当前观察，决定这一步做哪个动作，返回动作编号int。

        1. 看局面：用online算出各个动作的Q值。
        2. 决定是否探索：按epsilon概率随机选，否则选Q值最大的动作。
        3. 返回选中的动作编号。这里只作决定，不执行游戏、不更新网络。

        例如Q=[0.1, 0.5, 0.2]：按预测选就是动作1；探索则可选0、1、2。
        epsilon=0.2表示每次有20%的概率探索，不是只处理epsilon等于0或1。

        接口约定：epsilon在[0,1]，观察非空；输入float32的[1,N]张量，
        前向用no_grad。动作数从输出取得；随机使用self._exploration_rng，
        贪心平局取第一个最大值，返回Python int。非法输入抛ValueError。
        """
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon只允许0-1内")
            
        if not observation:
            raise ValueError("观察不能为空")

        with torch.no_grad():
            q_values : torch.Tensor = self.online(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))

        action_count = q_values.shape[1]

        if self._exploration_rng.random() < epsilon:
            return self._exploration_rng.randrange(action_count)
        else:
            return int(q_values.argmax(dim=1).item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
