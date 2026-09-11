"""原生道具快照回归；不依赖学习者尚未完成的观察编码。"""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import Runtime, RawSnapshot


@unittest.skipUnless(os.environ.get('RUN_CHROMIUM_GUI_TESTS') == '1', '需要原生显示服务')
class PowerUpChecks(unittest.TestCase):
    def trace(self, game: Runtime) -> list[RawSnapshot]:
        initial = game.reset(209)
        self.assertEqual(initial.powerups, ())
        rows = [initial]
        motion_checks = 0
        seen: set[int] = set()
        retired: set[int] = set()
        previous_ids: set[int] = set()
        for _ in range(1100):
            current = game.step(17, 1)
            snapshot = current.snapshot
            ids = {p.id for p in snapshot.powerups}
            self.assertFalse(ids & retired, '道具消失后ID不能复用')
            retired |= previous_ids - ids
            previous_ids = ids
            seen |= ids
            before = {p.id: p for p in rows[-1].powerups}
            for p in snapshot.powerups:
                self.assertIn(p.type, range(6))
                if p.id in before:
                    old = before[p.id]
                    self.assertEqual(p.type, old.type)
                    # y不做屏幕边界裁剪，存活对象的一tick差应匹配预测位移。
                    self.assertAlmostEqual(p.position[1]-old.position[1], old.next_displacement[1], places=4)
                    motion_checks += 1
            if snapshot.powerups and len(rows) % 50 == 0:
                game.render()  # Runtime内部比较绘图前后的完整快照，不相同就抛错
            rows.append(snapshot)
            if current.terminated:
                break
        self.assertGreater(len(seen), 0)
        self.assertGreater(motion_checks, 10)
        self.assertGreater(len(retired), 0)
        return rows

    def test_motion_ids_reset_and_read_only_render(self) -> None:
        with Runtime() as game:
            first = self.trace(game)
            self.assertEqual(first, self.trace(game))
        with Runtime(render_each_step=True) as game:
            self.assertEqual(first, self.trace(game))


if __name__ == '__main__':
    unittest.main()
