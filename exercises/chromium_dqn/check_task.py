"""教师检查器：不包含Task实现；模拟检查不等于原生验证。

运行方法与完整练习契约见同目录task.py。
"""
import argparse
from dataclasses import replace
import math
from unittest.mock import patch
from runtime import Runtime, RawSnapshot, RawPlayer, RawEnemy, RawBullet, RawStep
from task import GameTask as Task, encode_observation, compute_reward as reward_for_version, OBSERVATION_SIZE
from functools import partial
GameTask = partial(Task, task_version='task-v2-powerups')
compute_reward = partial(reward_for_version, task_version='task-v2-powerups')


def fixture(x=0.0, score=0.0):
    return RawSnapshot('game', False, 0, 1, 1.0, 900,
                       RawPlayer((x, -3.0, 25.0), (0.0, 0.0), score, 4, -500.0,
                                 500.0, (0.0, 0.0, 0.0), True), (), ())


def check_vector(actual, expected):
    assert isinstance(actual, list) and len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert type(a) is float and math.isclose(a, b, abs_tol=1e-8), (index, a, b)


class ScriptedRuntime:
    """确定性测试输入，完全不启动C++。"""
    def __init__(self, mode='game', fail=False):
        self.calls = 0
        self.mode = mode
        self.fail = fail

    def reset(self, seed):
        return fixture(2.0, 100.0)

    def step(self, action, ticks):
        self.calls += 1
        if self.fail:
            raise OSError('模拟通信丢失')
        state = replace(fixture(2.5, 120.0), mode=self.mode)
        ended = self.mode != 'game'
        return RawStep(state, 1 if ended else ticks, ticks, ticks * 0.02, ended,
                       self.mode if ended else None)


def checks():
    base = fixture()
    check_vector(encode_observation(base), [0.0, -0.4, 0.0, 0.0, 1.0, -1.0, 1.0,
                                            0.0, 0.0, 0.0] + [0.0] * (OBSERVATION_SIZE - 10))
    # 非对称尺度：纵向距离7比横向距离8近，缩放后排序会反转。
    enemy_x = RawEnemy(0, (8.0, -3.0, 25.0), (0.0, 0.0, 0.0), (1.0, 1.0), -1.0)
    enemy_y = replace(enemy_x, position=(0.0, 4.0, 25.0))
    bullets = tuple(RawBullet(i, 0, (float(i), -3.0, 25.0), (0.1, -0.2, 0.0),
                              (1.0, 1.0), 5.0) for i in range(10, 0, -1))
    state = replace(base, enemies=(enemy_x, enemy_y), enemy_bullets=bullets)
    vector = encode_observation(state)
    assert len(vector) == OBSERVATION_SIZE
    check_vector(vector[10:16], [1.0, 0.0, 7.0 / 15, 1.0, 0.4, 0.0])
    for slot in range(8):
        check_vector(vector[22 + 5*slot:27 + 5*slot],
                     [1.0, (slot+1)/20, 0.0, 0.1, -0.2])
    assert state.enemy_bullets == bullets
    ties = (replace(bullets[0], id=2, position=(-1.0, -3.0, 25.0)),
            replace(bullets[0], id=1, position=(1.0, -3.0, 25.0)))
    check_vector(encode_observation(replace(base, enemy_bullets=ties))[22:25], [1.0, 0.05, 0.0])
    # 防止漏写dy平方；下方较远对象不能因dy为负而排在前面。
    near = replace(enemy_x, position=(2.0, -3.0, 25.0))
    far = replace(enemy_x, position=(0.0, -6.0, 25.0))
    check_vector(encode_observation(replace(base, enemies=(near, far)))[10:13], [1.0, 0.1, 0.0])
    near_bullet = replace(bullets[0], id=1, position=near.position)
    far_bullet = replace(bullets[0], id=2, position=far.position)
    check_vector(encode_observation(replace(base, enemy_bullets=(near_bullet, far_bullet)))[22:25],
                 [1.0, 0.1, 0.0])
    # 同距离时type优先于横坐标，避免误写Python内置type函数。
    type0 = replace(near, type=0)
    type1 = replace(near, type=1, position=(-2.0, -3.0, 25.0))
    check_vector(encode_observation(replace(base, enemies=(type1, type0)))[10:13], [1.0, 0.1, 0.0])
    assert math.isclose(compute_reward(fixture(score=100), fixture(score=120)), 0.2)
    for mode in ('game', 'hero_dead', 'level_over'):
        source = ScriptedRuntime(mode)
        task = GameTask(source, max_decisions=1)
        try:
            task.step(4)
            raise AssertionError('未reset却能step')
        except RuntimeError:
            pass
        start = task.reset(7)
        assert start.info == dict(score=100.0, lives_counter=4, decisions=0, actual_ticks=0, end_reason=None)
        for invalid in (True, -1, 18, 1.5):
            try:
                task.step(invalid)
                raise AssertionError('非法动作未拒绝')
            except ValueError:
                pass
        assert source.calls == 0
        result = task.step(4)
        assert math.isclose(result.reward, 0.2)
        assert math.isclose(result.observation[0], 0.25)
        assert math.isclose(start.observation[0], 0.2), '旧观察被修改'
        assert result.terminated == (mode != 'game')
        assert result.truncated == (mode == 'game')
        assert result.info['end_reason'] == ('decision_limit' if mode == 'game' else mode)
        assert result.info['actual_ticks'] == (5 if mode == 'game' else 1)
        try:
            task.step(4)
            raise AssertionError('结束后还能step')
        except RuntimeError:
            pass
        assert source.calls == 1
        assert task.reset(7).info['decisions'] == 0
    task = GameTask(ScriptedRuntime(fail=True))
    task.reset(7)
    try:
        task.step(0)
        raise AssertionError('通信失败被伪装成有效结果')
    except OSError:
        assert task._needs_reset
    # 连续两步分数均为120：第二步应为0，而不是再次相对开局给奖励。
    task = GameTask(ScriptedRuntime(), max_decisions=3)
    task.reset(7)
    assert math.isclose(task.step(4).reward, 0.2)
    second = task.step(4)
    assert second.reward == 0.0 and second.info['end_reason'] is None
    assert not task._needs_reset and second.info['decisions'] == 2
    for operation in ('reset', 'step'):
        task = GameTask(ScriptedRuntime())
        task.reset(7)
        with patch('task.encode_observation', side_effect=ValueError('模拟编码失败')):
            try:
                getattr(task, operation)(7 if operation == 'reset' else 4)
                raise AssertionError('编码失败被吞掉')
            except ValueError:
                assert task._needs_reset
        assert task._decisions == 0
        task.reset(7)
        assert not task._needs_reset
    with patch.object(task.runtime, 'reset', side_effect=OSError('模拟重置失败')):
        try:
            task.reset(7)
            raise AssertionError('重置失败被吞掉')
        except OSError:
            assert task._needs_reset
    print('Python任务逻辑检查通过；尚不代表原生游戏或DQN训练通过。')


def native():
    with Runtime() as game:
        task = GameTask(game, max_decisions=30)
        for episode in range(2):
            task.reset(7 + episode)
            for _ in range(30):
                result = task.step(13)
                game.render()
                if result.terminated or result.truncated:
                    print('回合', episode + 1, result.info)
                    break
            else:
                raise AssertionError('没有按回合上限停止')
    print('真实Task跨两回合通过；没有网络更新，不是训练成功。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true', help='逻辑通过后运行真实游戏两回合')
    args = parser.parse_args()
    try:
        checks()
        if args.native:
            native()
    except NotImplementedError as error:
        print('练习尚未完成：', error)
        print('请修改task.py的TODO；没有把本次检查记为通过。')
