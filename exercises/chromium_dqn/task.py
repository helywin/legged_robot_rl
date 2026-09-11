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

默认任务task-v3-events：110维观察与v2完全一致，新版奖励见下文。
候选task-v4-no-motion：同一事件奖励，删除26项运动字段，观察84维。
训练默认task-v7-hit-feedback：36维，子弹每次实际削减血量的比例给奖励，
子弹完成击毁额外+2；碰撞/重生爆炸不计攻击收益。其他项沿用当前v6。
保留自身8项、最近2敌机×3、最近4敌弹×3、最近1道具×10。
注意GameTask默认仍v3兼容教学检查，训练入口显式传入保存的任务版本。
历史task-v1/v2仍支持原分差奖励，下面的分差说明仅适用于旧版。：
动作索引0..17直接对应原生0..17；每次5tick。自然结束由runtime决定。
外部每局上限max_decisions；每次成功step计数加1，与实际tick数不同。
奖励 = (新原始分数 - 旧原始分数) / 100.0。首版不叠加其他奖励项。
这个100只是奖励尺度；例如100→120的分数增量20，对应奖励0.2。

观察：list[float]，新版固定110项（旧task-v1为前62项），全部从传入的同一份RawSnapshot计算。
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

索引62..109：最近4个道具，每槽12项：
[mask, dx/20, dy/15, next_dx/1, next_dy/1, power/1, 类型0..5的六项标志]。
类型依次为护盾、超级护盾、维修、弹药0、弹药1、弹药2。标志仅对应类型为1，
其他为0；类型编号不表示大小。next_dx/dy是下个tick边界裁剪前预计位移，
不是长时间速度预测；power是原生补给系数（不是道具分值），参考尺度1。
同距离按道具id排序，id不进入网络；空槽12个零。道具可在屏幕外生成，
本版不额外做可见区域筛选；容量4只是候选设计，并不表示已覆盖所有道具。

task-v3-events奖励规则：击毁敌机+1、拾取道具+0.2、每实际损命-5、进入level_over+20。
不使用原始游戏得分，不对漏机再次扣分（漏机已造成损命），不奖励存活时间或惩罚位置。
事件累计值来自原生episode_events，每步取差；计数缺失或倒退报错。击毁包含碰撞、
子弹、爆炸等引起的正常敌机销毁，不包含漏出屏幕和reset清理，不是精确子弹击杀归因。
例如同一步漏接得2500分并损命1，新版奖励-5；单独拾取一次奖励0.2。

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
from dataclasses import dataclass, asdict
from runtime import Runtime, RawSnapshot, RawPowerUp, RawEvents

TASK_VERSION = 'task-v3-events'
COMPACT_TASK_VERSION = 'task-v4-no-motion'
SHIELD_TASK_VERSION = 'task-v5-shield-damage'
SMALL_TASK_VERSION = 'task-v6-36'
HIT_TASK_VERSION = 'task-v7-hit-feedback'
TRAIN_TASK_VERSIONS = ('task-v2-powerups', TASK_VERSION, COMPACT_TASK_VERSION, SHIELD_TASK_VERSION, SMALL_TASK_VERSION, HIT_TASK_VERSION)
SHIELD_DAMAGE_SCALE = 100.0
OBSERVATION_SIZE = 110
POWERUP_COUNT = 4
POWERUP_SLOT_SIZE = 12


def observation_size_for(task_version: str) -> int:
    if task_version == 'task-v1':
        return 62
    if task_version in ('task-v2-powerups', TASK_VERSION):
        return OBSERVATION_SIZE
    if task_version in (COMPACT_TASK_VERSION, SHIELD_TASK_VERSION):
        return 84
    if task_version in (SMALL_TASK_VERSION, HIT_TASK_VERSION):
        return 36
    raise ValueError(f'不支持的任务版本：{task_version}')

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


def encode_observation(snapshot: RawSnapshot, include_powerups: bool = True) -> list[float]:
    """按任务规格编码；旧模型显式选择不含道具的62维输入。"""
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

    base: list[float] = player + enemies + bullets
    return base + encode_powerups(snapshot) if include_powerups else base


def encode_powerups(snapshot: RawSnapshot) -> list[float]:
    """4个最近道具槽位；类型使用六项独热编码，ID仅参与稳定排序。"""
    result: list[float] = []
    player_x, player_y = snapshot.player.position[:2]
    nearest: list[RawPowerUp] = sorted(snapshot.powerups, key=lambda item: (
        (item.position[0] - player_x) ** 2 + (item.position[1] - player_y) ** 2,
        item.id,
    ))[:POWERUP_COUNT]
    for item in nearest:
        kind: list[float] = [0.0] * 6
        kind[item.type] = 1.0
        result.extend([
            1.0, (item.position[0] - player_x) / DX_SCALE,
            (item.position[1] - player_y) / DY_SCALE,
            item.next_displacement[0], item.next_displacement[1], item.power,
            *kind,
        ])
    result.extend([0.0] * ((POWERUP_COUNT - len(nearest)) * POWERUP_SLOT_SIZE))
    return result


def encode_task_observation(snapshot: RawSnapshot, task_version: str) -> list[float]:
    """v4只删除运动字段：自身8、敌机4×3、敌弹8×3、道具4×10，共84项。

    从同一编码投影，保证排序、缩放、填充和其余信息与v3完全相同。
    v4/v5为84维；v6仅减少槽位数，8+2×3+4×3+1×10=36。
    没有历史帧或隐含速度输入；普通前馈网络不能从单帧确定运动方向。
    """
    observation_size_for(task_version)
    full: list[float] = encode_observation(snapshot, task_version != 'task-v1')
    if task_version not in (COMPACT_TASK_VERSION, SHIELD_TASK_VERSION, SMALL_TASK_VERSION, HIT_TASK_VERSION):
        return full
    compact: list[float] = full[:2] + full[4:22]
    for offset in range(22, 62, 5):
        compact.extend(full[offset:offset + 3])
    for offset in range(62, 110, 12):
        compact.extend(full[offset:offset + 3] + full[offset + 5:offset + 12])
    if task_version in (SMALL_TASK_VERSION, HIT_TASK_VERSION):
        # 84维中的自身8、前2敌机6、前4敌弹12、首个道具10。
        return compact[:14] + compact[20:32] + compact[44:54]
    return compact


def event_delta(before: RawSnapshot, after: RawSnapshot) -> RawEvents:
    """累计事件取差分；实际损命不会被同一步加命抵消。"""
    if before.events is None or after.events is None:
        raise ValueError('task-v3-events需要原生episode_events，不能把缺失事件当成零')
    changes = {name: getattr(after.events, name) - getattr(before.events, name)
               for name in RawEvents.__dataclass_fields__}
    if any(value < 0 for value in changes.values()):
        raise ValueError('事件计数倒退，不能跨reset计算奖励')
    return RawEvents(**changes)


def compute_reward(before: RawSnapshot, after: RawSnapshot,
                   task_version: str = TASK_VERSION) -> float:
    """v7按子弹削血比例+2×子弹击毁；v3..v6保留当前击毁×10规则。"""
    observation_size_for(task_version)
    if task_version in ('task-v1', 'task-v2-powerups'):
        return (after.player.score - before.player.score) / 100.0
    events: RawEvents = event_delta(before, after)
    completed: bool = before.mode != 'level_over' and after.mode == 'level_over'
    shield_penalty: float = (events.shield_damage / SHIELD_DAMAGE_SCALE
                             if task_version in (SHIELD_TASK_VERSION, SMALL_TASK_VERSION, HIT_TASK_VERSION) else 0.0)
    attack_reward: float = (events.projectile_damage_fraction + 2.0 * events.projectile_kills
                            if task_version == HIT_TASK_VERSION else events.enemies_destroyed * 10)
    return float(attack_reward + 0.2 * events.pickups - 5.0 * events.lives_lost
                 + 20.0 * completed - shield_penalty)


class GameTask:
    def __init__(self, runtime: Runtime, max_decisions: int = 1000,
                 task_version: str = TASK_VERSION) -> None:
        self.observation_size: int = observation_size_for(task_version)
        self.task_version: str = task_version
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
        if self.task_version in (TASK_VERSION, COMPACT_TASK_VERSION, SHIELD_TASK_VERSION, SMALL_TASK_VERSION, HIT_TASK_VERSION) and snapshot.events is None:
            raise ValueError('新版任务需要原生事件接口')
        observation = encode_task_observation(snapshot, self.task_version)
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
            observation = encode_task_observation(rawstep.snapshot, self.task_version)
            reward = compute_reward(self._previous, rawstep.snapshot, self.task_version)
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
            if self._previous.events is not None and rawstep.snapshot.events is not None:
                result.info['events'] = asdict(event_delta(self._previous, rawstep.snapshot))
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
