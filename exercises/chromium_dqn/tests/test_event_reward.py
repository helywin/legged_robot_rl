"""奖励目标与事件差分测试；不把手造事件当作原生验证。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_task import fixture
from runtime import RawEvents
from task import compute_reward, encode_observation, SHIELD_TASK_VERSION


class RewardChecks(unittest.TestCase):
    def test_loss_with_missed_score_and_offsetting_extra_life(self) -> None:
        before = replace(fixture(),events=RawEvents())
        after = replace(before,player=replace(before.player,score=2500.0),
                        events=RawEvents(enemies_escaped=1,lives_lost=1,missed_powerups=1,missed_powerup_score=2500.0))
        # 同步加命抵消掉命时，原始生命计数可不变；奖励必须仍惩罚实际损命。
        self.assertEqual(compute_reward(before,after),-5.0)
        self.assertEqual(compute_reward(before,after,'task-v2-powerups'),25.0)
        self.assertEqual(encode_observation(before),encode_observation(after))

    def test_shield_damage_is_event_based_and_versioned(self) -> None:
        before = replace(fixture(), events=RawEvents(shield_damage=20))
        # 同一步受损40又补盾：快照护盾反而增加，仍应扣0.4。
        after = replace(before, player=replace(before.player, shields=900),
                        events=RawEvents(shield_damage=60))
        self.assertAlmostEqual(compute_reward(before, after, SHIELD_TASK_VERSION), -0.4)
        self.assertEqual(compute_reward(before, after, 'task-v4-no-motion'), 0.0)
        # 自然衰减没有受伤事件；死亡清零只保留原损命惩罚。
        decay = replace(before, player=replace(before.player, shields=499.25))
        self.assertEqual(compute_reward(before, decay, SHIELD_TASK_VERSION), 0.0)
        dead = replace(before, player=replace(before.player, shields=0),
                       events=RawEvents(shield_damage=60, lives_lost=1))
        self.assertAlmostEqual(compute_reward(before, dead, SHIELD_TASK_VERSION), -5.4)

    def test_components_differences_terminal_and_no_position_reward(self) -> None:
        before=replace(fixture(),events=RawEvents(enemies_destroyed=5,pickups=2))
        after=replace(before,mode='level_over',events=RawEvents(enemies_destroyed=7,pickups=3))
        self.assertAlmostEqual(compute_reward(before,after),22.2)
        self.assertEqual(compute_reward(after,after),0.0)
        corner=replace(before,player=replace(before.player,position=(10.0,-7.5,25.0)))
        self.assertEqual(compute_reward(before,corner),0.0)
        with self.assertRaises(ValueError):compute_reward(fixture(),before)
        with self.assertRaises(ValueError):compute_reward(after,before)


if __name__ == '__main__':unittest.main()
