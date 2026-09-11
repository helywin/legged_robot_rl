"""090完整实作：从零实现第一关Task（学习者主写）。

当前状态：学习者实现后按要求由教师补齐；逻辑与教师执行的真实两回合检查通过。
以下保留原练习契约。

场景：runtime.py已经能控制真实游戏；你把它的原始状态变成学习任务。
目标：实现观察编码、奖励、reset和step四部分，再跑完整回合并正确重开。
先读课程090“Task实作约定”；本文件包含独立完成练习所需的全部契约。

允许修改：四个TODO函数体；可以在本文件添加小型辅助函数。
不要修改runtime.py、检查器或通过导入旧代码完成。当前没有网络训练。
运行（仓库根目录）：
  .venv/bin/python exercises/chromium_dqn/task.py
  .venv/bin/python exercises/chromium_dqn/check_task.py
  .venv/bin/python exercises/chromium_dqn/check_task.py --native
未实现时会友好提示；默认不会启动游戏。--native会打开真实游戏窗口。

首版任务规格 task-v1（本轮确定的基线，不是已证明足够通关的观察）：
动作索引0..17直接对应原生0..17；每次5tick。自然结束由runtime决定。
外部每局上限max_decisions；每次成功step计数加1，与实际tick数不同。
奖励 = (新原始分数 - 旧原始分数) / 100.0。首版不叠加其他奖励项。
这个100只是奖励尺度；例如100→120的分数增量20，对应奖励0.2。

观察：list[float]，固定62项，全部从传入的同一份RawSnapshot计算。
索引0..9依次为：
  player.position[0]/10, position[1]/7.5,
  keyboard_motion[0]/20, keyboard_motion[1]/20,
  lives_counter/4, damage/500, shields/500,
  ammo_stock[0]/150, ammo_stock[1]/150, ammo_stock[2]/150。
坐标是世界单位。keyboard_motion是键盘累积量，不是世界速度；20为首版
量级参考，不是物理上限。damage是原始伤害记账值，初始-500→-1；
lives_counter初始4，shields初始500，ammo的150是补给量级参考，不是容量上限。
全部不裁剪；归一化不保证所有值都在[-1,1]，禁止改变原始状态。

索引10..21：最近4架敌机，每个槽位[mask, dx/20, dy/15]。
索引22..61：最近8颗敌弹，每槽[mask, dx/20, dy/15, vx/1, vy/1]。
dx=对象x-玩家x，dy=对象y-玩家y；20/15是场景全宽/全高的参考量级。
敌弹vx/vy取velocity_per_tick前两项，1为每tick一个世界单位的参考量级。
排序先用原始dx²+dy²，由近到远；不要使用缩放后的距离排序。
敌机同距离按(type, position[0], position[1])打破平局；敌弹按id打破平局。
这里没有假造敌机稳定ID。敌机类型仅用于确定排序，不作为首版输入字段。
真实对象mask=1.0；数量不足，整个空槽全填0.0；超出容量只取最近对象。
mask用于区分“没有对象”与“对象相对坐标刚好为零”。原始tuple不能原地排序。

GameTask保存：runtime、max_decisions、_previous（上一份原始快照）、
_decisions（本局成功决策数）、_needs_reset（当前是否禁止继续step）。

reset(seed) -> ResetResult：
  调runtime.reset(seed)，编码初始观察；成功后保存快照、计数0、允许step。
  info字典必须包含score、decisions、actual_ticks、end_reason，另含原始lives_counter供评测：
  分别为初始分数、0、0、None。没有动作，所以不返回奖励、不生成经验。

step(action) -> StepResult：
  未reset或上一局结束时抛RuntimeError；动作必须是int且在0..17，bool不接受。
  调runtime.step(action, TICKS)，取得RawStep。编码其snapshot，计算奖励；
  决策计数加1。terminated取原生返回；truncated为达到上限且未自然结束。
  若自然结束优先，end_reason取raw.termination_reason；否则若上限为
  'decision_limit'，否则None。info还包含新score、decisions和actual_ticks。
  完成后更新_previous与_decisions；任一结束标志为True则等待reset。
  返回的observation属于动作后的同一局；严禁在step内部偷偷调用reset。
  发生运行/编码异常时使Task等待reset并将异常继续抛出，不能伪造奖励或经验。
  非法动作在调用runtime之前拒绝，不能推进游戏或增加计数。

成功条件：固定形状/字段正确，最近对象选择正确且不改变输入；奖励仅由
当前步产生；终止/截断分别正确；结束后拒绝step；重开清零自己的状态；
真实运行至少跨两个回合。检查通过不是已经学会游戏，课程仍需后续DQN实现。
"""
from dataclasses import dataclass
from runtime import Runtime, RawSnapshot

TASK_VERSION = 'task-v1'
OBSERVATION_SIZE = 62
ACTION_COUNT = 18
TICKS = 5
X_SCALE = 10.0
Y_SCALE = 7.5
MOTION_SCALE = 20.0
MAX_LIVES = 4.0
MAX_DAMAGE = 500.0
MAX_SHIELDS = 500.0
MAX_AMMO = 150.0
DX_SCALE = 20.0
DY_SCALE = 15.0
ENEMY_COUNT = 4
ENEMY_BULLET_COUNT = 8

@dataclass(frozen=True)
class ResetResult:
    observation: list[float]
    info: dict


@dataclass(frozen=True)
class StepResult:
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


def encode_observation(snapshot: RawSnapshot) -> list[float]:
    """ 1：按文件顶部规格实现62项观察；这是策略真正能看到的数据。"""
    x = snapshot.player.position[0] / X_SCALE
    y = snapshot.player.position[1] / Y_SCALE
    motion_x = snapshot.player.keyboard_motion[0] / MOTION_SCALE
    motion_y = snapshot.player.keyboard_motion[1] / MOTION_SCALE
    lives = snapshot.player.lives_counter / MAX_LIVES
    damage = snapshot.player.damage / MAX_DAMAGE
    shields = snapshot.player.shields / MAX_SHIELDS
    ammo0 = snapshot.player.ammo_stock[0] / MAX_AMMO
    ammo1 = snapshot.player.ammo_stock[1] / MAX_AMMO
    ammo2 = snapshot.player.ammo_stock[2] / MAX_AMMO
    player = [
        x, y, motion_x, motion_y, lives, damage, shields, ammo0, ammo1, ammo2
    ]

    enemies: list[float] = []
    sorted_enemies = sorted(snapshot.enemies, key=lambda enemy: (
        (enemy.position[0] - snapshot.player.position[0]) ** 2 +
        (enemy.position[1] - snapshot.player.position[1]) ** 2,
        enemy.type, enemy.position[0], enemy.position[1]
    ))
    for i in range(0, ENEMY_COUNT):
        if i >= len(sorted_enemies):
            enemies += (0.0, 0.0, 0.0)
        else:
            dx = sorted_enemies[i].position[0] - snapshot.player.position[0]
            dy = sorted_enemies[i].position[1] - snapshot.player.position[1]
            enemies += (1.0, dx / DX_SCALE, dy / DY_SCALE)

    bullets: list[float] = []
    sorted_bullets = sorted(snapshot.enemy_bullets, key=lambda bullet: (
        (bullet.position[0] - snapshot.player.position[0]) ** 2 +
        (bullet.position[1] - snapshot.player.position[1]) ** 2,
        bullet.id
    ))
    for i in range(0, ENEMY_BULLET_COUNT):
        if i >= len(sorted_bullets):
            bullets += (0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            dx = sorted_bullets[i].position[0] - snapshot.player.position[0]
            dy = sorted_bullets[i].position[1] - snapshot.player.position[1]

            bullets += (
                1.0, dx / DX_SCALE, dy / DY_SCALE,
                sorted_bullets[i].velocity_per_tick[0],
                sorted_bullets[i].velocity_per_tick[1],
            )

    return player + enemies + bullets



def compute_reward(before: RawSnapshot, after: RawSnapshot) -> float:
    """ 2：按task-v1计算本步奖励，返回float。"""
    return (after.player.score - before.player.score) / 100.0


class GameTask:
    def __init__(self, runtime: Runtime, max_decisions=1000):
        if type(max_decisions) is not int or max_decisions <= 0:
            raise ValueError('max_decisions必须是正整数')
        self.runtime = runtime
        self.max_decisions = max_decisions
        self._previous = None
        self._decisions = 0
        self._needs_reset = True

    def reset(self, seed: int) -> ResetResult:
        """开始新局；全部准备成功后，才允许下一次step。"""
        self._needs_reset = True
        snapshot = self.runtime.reset(seed)
        observation = encode_observation(snapshot)
        result = ResetResult(
            observation=observation,
            info=dict(score=snapshot.player.score, lives_counter=snapshot.player.lives_counter, decisions=0,
                      actual_ticks=0, end_reason=None),
        )
        self._previous = snapshot
        self._decisions = 0
        self._needs_reset = False
        return result

    def step(self, action: int) -> StepResult:
        """成功时提交新状态；运行或编码失败时等待重置并保留异常。"""
        if type(action) is not int or not 0 <= action < ACTION_COUNT:
            raise ValueError("action必须是0到17的整数")
        if self._needs_reset or self._previous is None:
            raise RuntimeError("需要重置环境")

        try:
            rawstep = self.runtime.step(action, TICKS)
            observation = encode_observation(rawstep.snapshot)
            reward = compute_reward(self._previous, rawstep.snapshot)
            decisions = self._decisions + 1
            terminated = rawstep.terminated
            # 本版任务规定：自然结束优先，不再同时标记外部截断。
            truncated = decisions >= self.max_decisions and not terminated
            if terminated:
                end_reason = rawstep.termination_reason
            elif truncated:
                end_reason = "decision_limit"
            else:
                end_reason = None
            result = StepResult(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=dict(score=rawstep.snapshot.player.score, lives_counter=rawstep.snapshot.player.lives_counter, decisions=decisions,
                          actual_ticks=rawstep.actual_ticks, end_reason=end_reason),
            )
        except Exception:
            # 游戏可能已经执行动作；不能假装这一步没有发生并继续采样。
            self._needs_reset = True
            raise

        # 保持上一快照直到奖励计算完；全部成功后一起更新任务状态。
        self._previous = rawstep.snapshot
        self._decisions = decisions
        self._needs_reset = terminated or truncated
        return result


if __name__ == '__main__':
    print(__doc__)
    print('任务模块已实现；运行check_task.py检查，添加--native验证真实游戏。')
