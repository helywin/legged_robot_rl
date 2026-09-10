"""090第二段：从零实现动作价值网络（学习者实现）。

目标：输入一批观察[B, observation_size]，输出各动作价值[B, action_count]。
B是本批样本数；实际任务输入62项、输出18项。输出是预期累计奖励的估计，
不是动作概率，因此末尾不要softmax，也不要ReLU限制输出非负。

实现范围：两个TODO函数体，不导入旧网络代码。
__init__：先保留super().__init__()，再创建self.layers，结构依次为
Linear(observation_size,64)、ReLU、Linear(64,64)、ReLU、Linear(64,action_count)。
可用nn.Sequential组合这些层；隐藏层64是首版容量选择，不是游戏字段数量。
forward：把observations送入self.layers，返回结果。这里不做优化或选动作。

机制：nn.Module登记子层及参数；super().__init__先初始化登记机制。
调用network(observations)会进入forward；nn.Linear对最后一维做线性变换，
同时保留批次维。激活函数给隐藏层引入非线性。参数更新由Agent负责。

运行（仓库根）：.venv/bin/python exercises/chromium_dqn/check_update.py
需要同时完成agent.py的TODO才能完成整段检查；未完成有友好提示。
成功条件：62维观察批量输入，得到每行18个有限数值；各层参数能被优化器读取。
"""
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_size: int = 62, action_count: int = 18) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)
