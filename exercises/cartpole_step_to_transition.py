#!/usr/bin/env python3
"""第 59 课编程练习：把 CartPole 的真实交互结果保存成回放经验。

为什么有这道题
================

第 58 课使用的是预先写好的 `VectorTransition`。真实训练中没有现成经验，必须
先保存执行动作之前的观察，再把 `env.step(action)` 返回的结果与它组合起来。

一条经验记录的是同一个环境 step 的完整因果关系：

    执行动作前的 observation
    + 实际执行的 action
    + 执行后得到的 reward、next_observation、terminated、truncated
    = 一条 VectorTransition

固定的真实数据
================

本题真实启动 `CartPole-v1`，固定：

    seed = 20260903
    actions = (1, 1, 0)

第一步执行前，`reset_cartpole()` 返回：

    observation=(-0.024423, -0.016661, -0.029921, 0.029134)

执行动作 1 后，`step_cartpole()` 返回：

    next_observation=(-0.024756, 0.178877, -0.029338, -0.272838)
    reward=1.0, terminated=False, truncated=False

因此第 0 条经验的 `observation` 必须是 reset 的结果，`next_observation` 必须
是 step 的结果。保存后再执行：

    observation = next_observation

这样第 1 条经验的旧观察，才会等于第 0 条经验的新观察，时间链不会断。

已知的精确接口
================

本文件已经把 Gymnasium 返回值转换成带明确类型的两个函数，你不需要猜接口：

    observation = reset_cartpole(env=env, seed=seed)

返回一个包含 4 个 `float` 的元组。

    next_observation, reward, terminated, truncated = step_cartpole(
        env=env,
        action=action,
    )

右边返回 4 项，左边 4 个名字按位置接收它们，这叫“元组拆包”。各项类型依次为：

    tuple[float, float, float, float], float, bool, bool

构造经验的完整接口是：

    transition = VectorTransition(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        terminated=terminated,
        truncated=truncated,
    )

保存经验：

    replay_buffer.add(transition)

读取缓冲区当前内容：

    transitions = replay_buffer.snapshot()

你的任务
========

只修改 `collect_cartpole_transitions()` 中的 TODO：

1. 调用 `reset_cartpole(...)` 得到第一项旧观察；
2. 按顺序遍历 `actions`；
3. 每轮调用 `step_cartpole(...)` 并用 4 个名字拆包；
4. 用本轮动作执行前后的数据构造 `VectorTransition`；
5. 把经验加入缓冲区，再让 `observation` 指向 `next_observation`；
6. 若 `terminated or truncated` 为真，立即结束动作循环；
7. 返回 `replay_buffer.snapshot()`。

不要调用网络、loss、backward 或 optimizer，不要重新 reset，不要篡改动作序列，
也不要在构造经验前把 `observation` 提前替换成 `next_observation`。

运行命令
========

在仓库根目录执行：

    .venv/bin/python -m exercises.cartpole_step_to_transition

成功条件
========

最后应确认真实 CartPole 的 3 条经验字段、动作顺序、奖励、前后观察链、缓冲区
内容和结束时停止 step 的边界全部为 `PASS`，并显示：

    练习通过：CartPole 的 step 结果已形成连续的回放经验

TODO 未完成时只显示友好提示，不会输出异常堆栈或破坏全量测试。

当前没有验证
============

本题真实启动 CartPole 并收集三条经验，但动作仍是人工固定的；没有用网络选择
动作，没有抽样训练、完整 episode、检查点、独立评估、Isaac Lab 或真机验证。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import SupportsFloat, cast

import gymnasium as gym

from exercises.replay_sample_batch_update import VectorReplayBuffer
from exercises.replay_samples_to_tensors import VectorTransition


CartPoleObservation = tuple[float, float, float, float]


def to_cartpole_observation(
    values: Iterable[SupportsFloat],
) -> CartPoleObservation:
    """把 Gymnasium 的四项观察转换为 IDE 可识别的 float 元组。"""
    converted = tuple(float(value) for value in values)
    if len(converted) != 4:
        raise ValueError("CartPole observation 必须恰好包含 4 项")
    return cast(CartPoleObservation, converted)


def reset_cartpole(env: gym.Env, seed: int) -> CartPoleObservation:
    """调用真实 reset，并只返回训练需要的初始观察。"""
    observation, _info = env.reset(seed=seed)
    return to_cartpole_observation(observation)


def step_cartpole(
    env: gym.Env,
    action: int,
) -> tuple[CartPoleObservation, float, bool, bool]:
    """调用真实 step，并返回类型明确的四项训练数据。"""
    next_observation, reward, terminated, truncated, _info = (
        env.step(action)
    )
    return (
        to_cartpole_observation(next_observation),
        float(reward),
        bool(terminated),
        bool(truncated),
    )


def collect_cartpole_transitions(
    env: gym.Env,
    replay_buffer: VectorReplayBuffer,
    actions: tuple[int, ...],
    seed: int,
) -> tuple[VectorTransition, ...]:
    """依次执行动作，把每一步保存成时间连续的经验。"""
    # 1：调用 reset_cartpole(env=env, seed=seed) 得到 observation。
    observation = reset_cartpole(env, seed)

    # 2：按顺序遍历 actions；本轮循环变量就是实际 action。
    for action in actions:

    # 3：调用 step_cartpole(env=env, action=action)，用
    # next_observation、reward、terminated、truncated 四个名字拆包。
        next_observation, reward, terminated, truncated = step_cartpole(env, action)

    # 4：用动作执行前的 observation 和本轮 step 返回值构造
    # VectorTransition，再用 replay_buffer.add(...) 保存。
        transition = VectorTransition(observation, action, reward, next_observation, terminated, truncated)
        replay_buffer.add(transition)

    # 5：保存后令 observation = next_observation，为下一步接好时间链。
        observation = next_observation

    # 6：如果 terminated or truncated，立即 break。
        if terminated or truncated:
            break

    # 7：循环结束后返回 replay_buffer.snapshot()。
    return replay_buffer.snapshot()


class _OneStepEndingEnv:
    """检查终止后不能继续 step 的最小环境，不属于学习者 TODO。"""

    def __init__(self) -> None:
        self.step_calls = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[list[float], dict[str, object]]:
        del seed, options
        return [0.0, 0.0, 0.0, 0.0], {}

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
        del action
        self.step_calls += 1
        if self.step_calls > 1:
            raise RuntimeError("terminated 后不应再次调用 step")
        return [0.1, 0.2, 0.3, 0.4], 1.0, True, False, {}


def check_exercise() -> bool | None:
    """检查真实数据字段、时间链和终止边界。"""
    replay_buffer = VectorReplayBuffer(capacity=10)
    env = gym.make("CartPole-v1")
    try:
        try:
            transitions = collect_cartpole_transitions(
                env=env,
                replay_buffer=replay_buffer,
                actions=(1, 1, 0),
                seed=20260903,
            )
        except NotImplementedError as error:
            print(f"练习尚未完成：{error}")
            return None
    finally:
        env.close()

    expected_first_observation = (
        -0.024423254653811455,
        -0.016660941764712334,
        -0.02992112562060356,
        0.02913360297679901,
    )
    expected_first_next_observation = (
        -0.024756472557783127,
        0.17887704074382782,
        -0.029338452965021133,
        -0.2728375792503357,
    )

    count_ok = len(transitions) == 3
    actions_ok = count_ok and tuple(
        transition.action for transition in transitions
    ) == (1, 1, 0)
    rewards_ok = count_ok and all(
        transition.reward == 1.0 for transition in transitions
    )
    flags_ok = count_ok and all(
        not transition.terminated and not transition.truncated
        for transition in transitions
    )
    first_values_ok = count_ok and all(
        abs(actual - expected) < 1e-7
        for actual, expected in zip(
            transitions[0].observation,
            expected_first_observation,
        )
    ) and all(
        abs(actual - expected) < 1e-7
        for actual, expected in zip(
            transitions[0].next_observation,
            expected_first_next_observation,
        )
    )
    chain_ok = count_ok and all(
        current.next_observation == following.observation
        for current, following in zip(transitions, transitions[1:])
    )
    buffer_matches_return = transitions == replay_buffer.snapshot()

    ending_env = _OneStepEndingEnv()
    ending_buffer = VectorReplayBuffer(capacity=10)
    ending_transitions = collect_cartpole_transitions(
        env=cast(gym.Env, ending_env),
        replay_buffer=ending_buffer,
        actions=(0, 1),
        seed=0,
    )
    stops_after_done = (
        len(ending_transitions) == 1
        and ending_env.step_calls == 1
        and ending_transitions[0].terminated
    )

    checks = (
        ("three_real_transitions_collected=True", count_ok),
        ("actions_preserve_execution_order=(1, 1, 0)", actions_ok),
        ("rewards_are_three_ones=True", rewards_ok),
        ("first_transition_matches_real_step=True", first_values_ok),
        ("three_steps_not_done=True", flags_ok),
        ("next_observation_chains_to_next_old_observation=True", chain_ok),
        ("returned_snapshot_matches_buffer=True", buffer_matches_return),
        ("stops_immediately_after_done=True", stops_after_done),
    )
    for label, passed in checks:
        print(f"{label} {'PASS' if passed else 'FAIL'}")

    if count_ok:
        for index, transition in enumerate(transitions):
            print(
                f"transition[{index}] action={transition.action} "
                f"reward={transition.reward:.1f} "
                f"terminated={transition.terminated} "
                f"truncated={transition.truncated}"
            )

    return all(passed for _label, passed in checks)


def main() -> None:
    result = check_exercise()
    if result:
        print()
        print("练习通过：CartPole 的 step 结果已形成连续的回放经验")
    elif result is False:
        print()
        print(
            "还有检查未通过，请只修改 "
            "collect_cartpole_transitions() 中的 TODO"
        )


if __name__ == "__main__":
    main()
