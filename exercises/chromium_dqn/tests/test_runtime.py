"""从仓库根运行：RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from runtime import Runtime, RuntimeFailure, GameRejected


class TransportChecks(unittest.TestCase):
    def test_faults_close_owned_child(self):
        # 自建故障进程；只验证传输错误处理，不能冒充原生游戏验收。
        for fault in ('timeout', 'eof', 'id', 'snapshot', 'oversize', 'capability'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                peer = Path(directory) / 'peer'
                pidfile = Path(directory) / 'pid'
                peer.write_text(f'''#!{sys.executable}
import json, os, sys, time
from pathlib import Path
Path({str(pidfile)!r}).write_text(str(os.getpid()))
for line in sys.stdin:
    req = json.loads(line)
    result = dict(step=True, reset=True, seed=True, render=True, render_free_steps=True, powerups=True, episode_events=True, shield_damage=True,
                  schema_version=2, implementation='test-peer')
    if {fault!r} == 'capability': result['seed'] = False
    if req['command'] == 'reset':
        if {fault!r} == 'timeout': time.sleep(10)
        if {fault!r} == 'eof': sys.exit(0)
        if {fault!r} == 'oversize':
            sys.stdout.write('x' * (1024 * 1024 + 1)); sys.stdout.flush(); continue
        result = dict(schema_version=999)
        if {fault!r} == 'id': req['request_id'] += 1
    print(json.dumps(dict(protocol_version=1, request_id=req['request_id'], ok=True, result=result)), flush=True)
''')
                peer.chmod(0o700)
                if fault == 'capability':
                    with self.assertRaises(ValueError):
                        Runtime(timeout=0.5, binary=peer)
                else:
                    game = Runtime(timeout=0.5, binary=peer)
                    with self.assertRaises(RuntimeFailure):
                        game.reset(7)
                    game.close()
                    with self.assertRaises(RuntimeFailure):
                        game.reset(7)
                pid = int(pidfile.read_text())
                self.assertFalse(Path(f'/proc/{pid}').exists())


@unittest.skipUnless(os.environ.get('RUN_CHROMIUM_GUI_TESTS') == '1', 'explicit native display run required')
class NativeChecks(unittest.TestCase):
    def trace(self, game):
        rows = []
        for i in range(150):
            row = game.step(13 if i % 2 else 12, 10)
            rows.append(row)
            if row.terminated:
                break
        return rows

    def test_lifecycle_repeatability_and_immutable_snapshot(self):
        with Runtime() as game:
            pid = game._process.pid
            state_directory = Path(game._state.name)
            with self.assertRaises(GameRejected):
                game.step(4, 1)
            initial = game.reset(7)
            baseline = self.trace(game)
            self.assertTrue(any(row.snapshot.enemy_bullets for row in baseline))
            self.assertEqual(initial.player.position, (0.0, -3.0, 25.0))
            game.reset(8)
            game.step(4, 10)
            self.assertEqual(initial, game.reset(7))
            self.assertEqual(baseline, self.trace(game))
            game.render()
            for action, ticks in ((True, 1), (18, 1), (0, 0), (0, 51)):
                with self.assertRaises(ValueError):
                    game.step(action, ticks)
            game.reset(19)
            for _ in range(600):
                row = game.step(0, 50)
                if row.terminated:
                    break
            self.assertTrue(row.terminated)
            with self.assertRaises(GameRejected):
                game.step(0, 1)
            game.reset(19)
            self.assertEqual(game.step(0, 1).episode_tick, 1)
        game.close()
        self.assertFalse(state_directory.exists())
        self.assertFalse(Path(f'/proc/{pid}').exists())
        self.assertTrue(game.log_path.exists())
