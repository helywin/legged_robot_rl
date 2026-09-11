"""新项目的原生通信层（教师实现；不导入旧客户端）。

从仓库根目录执行：.venv/bin/python exercises/chromium_dqn/runtime.py
该命令只做 reset → 右移 → render → close，不训练、不创建经验。

调用约定：每个Runtime最多一个未完成调用（可由线程池串行调用）；with负责关闭。
Runtime(headless=True)不创建窗口/GL，训练默认使用；GUI回放使用默认False。
reset(seed)返回RawSnapshot，
step(action_id, ticks)返回RawStep。所有快照不可变。位置为世界坐标，
keyboard_motion不是世界速度，velocity_per_tick不是每秒速度，damage不是剩余血量。
通信失败后实例关闭，不能重发同一步；游戏可能已经执行了丢失响应的动作。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from queue import Queue, Full
import subprocess
import tempfile
from threading import Thread
from uuid import uuid4


class RuntimeFailure(RuntimeError):
    """通信或状态契约损坏；实例已关闭。"""


class GameRejected(RuntimeError):
    """游戏明确拒绝命令；没有把它当作一次有效环境步。"""


def _int(value, low=None, high=None):
    if type(value) is not int or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"需要整数，范围[{low}, {high}]")
    return value


def _float(value):
    if type(value) not in (int, float):
        raise ValueError("需要有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("需要有限数值")
    return result


def _bool(value):
    if type(value) is not bool:
        raise ValueError("需要布尔值")
    return value


def _vector(value, size):
    if type(value) is not list or len(value) != size:
        raise ValueError(f"需要长度为{size}的数组")
    return tuple(_float(item) for item in value)


def _object(value):
    if type(value) is not dict:
        raise ValueError("需要JSON对象")
    return value


def _array(value):
    if type(value) is not list:
        raise ValueError("需要JSON数组")
    return value


@dataclass(frozen=True)
class RawPlayer:
    position: tuple[float, float, float]
    keyboard_motion: tuple[float, float]
    score: float
    lives_counter: int
    damage: float
    shields: float
    ammo_stock: tuple[float, float, float]
    visible: bool


@dataclass(frozen=True)
class RawEnemy:
    type: int
    position: tuple[float, float, float]
    raw_velocity: tuple[float, float, float]
    size: tuple[float, float]
    damage: float


@dataclass(frozen=True)
class RawBullet:
    id: int
    type: int
    position: tuple[float, float, float]
    velocity_per_tick: tuple[float, float, float]
    sprite_half_size: tuple[float, float]
    damage: float


@dataclass(frozen=True)
class RawPowerUp:
    id: int
    type: int  # 0护盾、1超级护盾、2维修、3/4/5三种弹药
    position: tuple[float, float, float]
    next_displacement: tuple[float, float]  # 下个tick边界限制前的位移
    power: float  # 原生补给系数，不是得分


@dataclass(frozen=True)
class RawEvents:
    enemies_destroyed: int = 0
    enemies_escaped: int = 0
    lives_lost: int = 0
    pickups: int = 0
    missed_powerups: int = 0
    pickup_score: float = 0.0
    missed_powerup_score: float = 0.0
    shield_damage: float = 0.0  # 累计实际受伤吸收量，不含自然衰减或重置
    projectile_kills: int = 0
    projectile_damage: float = 0.0
    projectile_damage_fraction: float = 0.0


@dataclass(frozen=True)
class RawSnapshot:
    mode: str
    paused: bool
    game_frame: int
    level: int
    speed_adjustment: float
    rng_cursor: int
    player: RawPlayer
    enemies: tuple[RawEnemy, ...]
    enemy_bullets: tuple[RawBullet, ...]
    powerups: tuple[RawPowerUp, ...] = ()
    events: RawEvents | None = None


@dataclass(frozen=True)
class RawStep:
    snapshot: RawSnapshot
    actual_ticks: int
    episode_tick: int
    simulated_seconds: float  # 从本局开始累计的模拟秒数，不是本次耗时。
    terminated: bool
    termination_reason: str | None


def _snapshot(value):
    obj = _object(value)
    if _int(obj['schema_version']) != 2:
        raise ValueError("只接受已验证的快照schema 2")
    mode = obj['mode']
    if mode not in ('game', 'menu', 'hero_dead', 'level_over'):
        raise ValueError("未知游戏模式")
    p = _object(obj['player'])
    player = RawPlayer(_vector(p['position'], 3), _vector(p['keyboard_motion'], 2),
                       _float(p['score']), _int(p['lives_counter']), _float(p['damage']),
                       _float(p['shields']), _vector(p['ammo_stock'], 3), _bool(p['visible']))
    enemies = []
    for e in _array(obj['enemies']):
        e = _object(e)
        enemies.append(RawEnemy(_int(e['type'], 0), _vector(e['position'], 3),
                               _vector(e['raw_velocity'], 3), _vector(e['size'], 2), _float(e['damage'])))
    bullets = []
    for b in _array(obj['enemy_bullets']):
        b = _object(b)
        bullets.append(RawBullet(_int(b['id'], 1), _int(b['type'], 0), _vector(b['position'], 3),
                                 _vector(b['velocity_per_tick'], 3), _vector(b['sprite_half_size'], 2),
                                 _float(b['damage'])))
    if len({b.id for b in bullets}) != len(bullets):
        raise ValueError("同一快照存在重复子弹ID")
    powerups: list[RawPowerUp] = []
    for raw in _array(obj['powerups']):
        p = _object(raw)
        kind = _int(p['type'], 0)
        if kind > 5:
            raise ValueError("未支持的道具类型")
        powerups.append(RawPowerUp(_int(p['id'], 1), kind, _vector(p['position'], 3),
                                  _vector(p['next_displacement'], 2), _float(p['power'])))
    if len({p.id for p in powerups}) != len(powerups):
        raise ValueError("同一快照存在重复道具ID")
    event = _object(obj['episode_events'])
    events = RawEvents(*(_int(event[key], 0) for key in (
        'enemies_destroyed', 'enemies_escaped', 'lives_lost', 'pickups', 'missed_powerups')),
        _float(event['pickup_score']), _float(event['missed_powerup_score']),
        _float(event['shield_damage']), _int(event['projectile_kills'], 0),
        _float(event['projectile_damage']), _float(event['projectile_damage_fraction']))
    if min(events.pickup_score, events.missed_powerup_score, events.shield_damage,
           events.projectile_damage, events.projectile_damage_fraction) < 0:
        raise ValueError('道具累计得分与护盾受损量不能为负')
    return RawSnapshot(mode, _bool(obj['paused']), _int(obj['game_frame'], 0),
                       _int(obj['level'], 1), _float(obj['speed_adjustment']),
                       _int(obj['rng_cursor'], 0), player, tuple(enemies), tuple(bullets), tuple(powerups), events)


class Runtime:
    """保持一个原生进程；必须先reset，再step。仅支持同步第一关。"""

    def __init__(self, render_each_step=False, timeout=10.0, binary=None, data_directory=None, headless: bool = False):
        _bool(render_each_step)
        _bool(headless)
        if headless and render_each_step:
            raise ValueError("无窗口模式不能逐步绘图")
        self.headless: bool = headless
        self.timeout = _float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout必须大于0")
        repository = Path(__file__).resolve().parents[2]
        native = repository / 'third_party/chromium-bsu-rl'
        binary = Path(binary or native / 'build/install/bin/chromium-bsu-rl').resolve()
        data = Path(data_directory or native / 'game/data').resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f"请先构建原生游戏：{binary}")
        if not (data / 'png').is_dir() or len(os.fsencode(data)) >= 180:
            raise ValueError("游戏资源路径不存在或超过原生安全长度")
        self.run_directory = Path(__file__).parent / 'runs' / uuid4().hex
        self.run_directory.mkdir(parents=True)
        self.log_path = self.run_directory / 'native.log'
        self._state = tempfile.TemporaryDirectory(prefix='dqn-', dir='/tmp')
        self._process = None
        self._reader = None
        self._closed = False
        self._id = 0
        self._last = None
        self._ticks = 0
        self._messages = Queue(maxsize=2)
        env = dict(os.environ)
        env.update(CHROMIUM_BSU_RL_PROTOCOL='1', CHROMIUM_BSU_RL_SYNCHRONOUS='1',
                   CHROMIUM_BSU_RL_RENDER='1' if render_each_step else '0',
                   CHROMIUM_BSU_RL_STATE_DIR=self._state.name,
                   CHROMIUM_BSU_SCORE=self._state.name + '/scores', CHROMIUM_BSU_DATA=str(data),
                   SDL_VIDEODRIVER='x11', CHROMIUM_BSU_RL_HEADLESS='1' if headless else '0')
        if headless:
            env.pop('DISPLAY', None)
            env.pop('WAYLAND_DISPLAY', None)
            env['SDL_VIDEODRIVER'] = 'dummy'
        try:
            with self.log_path.open('wb') as log:
                self._process = subprocess.Popen([str(binary), '--window', '--vidmode', '1', '--noaudio'],
                                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 stderr=log, env=env)
            self._reader = Thread(target=self._read_responses, daemon=True)
            self._reader.start()
            hello = _object(self._request('hello'))
            for capability in ('step', 'reset', 'seed', 'render_free_steps', 'powerups', 'episode_events', 'shield_damage', 'projectile_damage',
                               'headless' if headless else 'render'):
                if not _bool(hello[capability]):
                    raise ValueError(f"原生能力缺失：{capability}")
            if _int(hello['schema_version']) != 2:
                raise ValueError("原生快照版本不兼容")
            self.implementation = hello['implementation']
            if type(self.implementation) is not str:
                raise ValueError("原生版本字段错误")
            (self.run_directory / 'runtime.json').write_text(json.dumps({
                'binary': str(binary), 'data': str(data), 'implementation': self.implementation,
                'render_each_step': render_each_step, 'headless': headless, 'timeout': self.timeout,
            }, indent=2))
        except Exception:
            self.close()
            raise

    def _read_responses(self):
        try:
            while True:
                wire = self._process.stdout.readline(1024 * 1024 + 1)
                if not wire or len(wire) > 1024 * 1024 or not wire.endswith(b'\n'):
                    raise ValueError("响应中断、超长或缺少换行")
                self._messages.put_nowait(wire)
        except Exception as error:
            try:
                self._messages.put_nowait(error)
            except Full:
                pass  # 队列内未请求的响应也会因request_id不匹配而拒绝。

    def _fail(self, error):
        self.close()
        raise RuntimeFailure(f"原生状态不再可信：{error}；日志：{self.log_path}") from error

    def _request(self, command, **arguments):
        if self._closed:
            raise RuntimeFailure('实例已关闭，请创建新Runtime')
        self._id += 1
        try:
            wire = json.dumps(dict(protocol_version=1, request_id=self._id, command=command,
                                   **arguments), allow_nan=False).encode() + b'\n'
            # 单个固定小请求，小于POSIX管道容量；始终只有一个请求在途。
            if len(wire) > 4096:
                raise ValueError('请求过长')
            self._process.stdin.write(wire)
            self._process.stdin.flush()
            message = self._messages.get(timeout=self.timeout)
            if isinstance(message, Exception):
                raise message
            envelope = _object(json.loads(message))
            if _int(envelope['protocol_version']) != 1 or _int(envelope['request_id']) != self._id:
                raise ValueError('响应版本或请求编号不匹配')
            if not _bool(envelope['ok']):
                error = _object(envelope['error'])
                code, description = error['code'], error['message']
                if type(code) is not str or type(description) is not str:
                    raise ValueError('错误响应格式不合法')
                raise GameRejected(f'{code}: {description}')
            return envelope['result']
        except GameRejected:
            raise
        except Exception as error:
            self._fail(error)

    def reset(self, seed: int) -> RawSnapshot:
        _int(seed, 0, 4294967295)
        response = self._request('reset', seed=seed)
        try:
            state = _snapshot(response)
            if state.mode != 'game' or state.paused or state.level != 1 or state.game_frame != 0:
                raise ValueError('reset没有返回第一关初始状态')
            self._last, self._ticks = state, 0
            return state
        except Exception as error:
            self._fail(error)

    def step(self, action_id: int, ticks: int) -> RawStep:
        _int(action_id, 0, 17)
        _int(ticks, 1, 50)
        if self._last is None or self._last.mode != 'game':
            raise GameRejected('请先reset开始新局，再step')
        response = self._request('step', action=action_id, ticks=ticks)
        try:
            obj = _object(response)
            state = _snapshot(obj['snapshot'])
            actual = _int(obj['actual_ticks'], 1, ticks)
            total = _int(obj['episode_tick'], 1)
            seconds = _float(obj['simulated_seconds'])
            ended = _bool(obj['terminated'])
            if state.mode not in ('game', 'hero_dead', 'level_over') or state.paused:
                raise ValueError('同步step返回不支持的状态')
            if total != self._ticks + actual or not math.isclose(seconds, total * 0.02, abs_tol=1e-9):
                raise ValueError('实际tick计数或累计模拟时间不一致')
            if ended != (state.mode != 'game') or (actual < ticks and not ended):
                raise ValueError('实际推进步数与终止状态矛盾')
            result = RawStep(state, actual, total, seconds, ended, state.mode if ended else None)
            self._last, self._ticks = state, total
            return result
        except Exception as error:
            self._fail(error)

    def render(self) -> None:
        if self.headless:
            raise GameRejected("无窗口模式不支持绘图；回放请创建GUI运行时")
        if self._last is None:
            raise GameRejected('请先reset，再render')
        response = self._request('render')
        try:
            if _snapshot(response) != self._last:
                raise ValueError('绘图改变了游戏快照')
        except Exception as error:
            self._fail(error)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            # EOF是协议定义的关闭方式，不再发起可能超时的新请求。
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self._reader is not None:
                self._reader.join(timeout=1)
            process.stdout.close()
        self._state.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == '__main__':
    with Runtime() as game:
        initial = game.reset(7)
        result = game.step(4, 5)
        game.render()
        print('初始位置:', initial.player.position)
        print('动作后位置:', result.snapshot.player.position)
        print('本次tick:', result.actual_ticks, '本局累计tick:', result.episode_tick)
        print('结束原因:', result.termination_reason)
        print('原生日志:', game.log_path)
