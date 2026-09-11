# 新增道具观察：接口与训练连接验证

## 问题与假设

现有62维输入不能告诉模型道具的位置、类型和运动。学习者要求教师直接完成实现。本轮先验证接口与110维训练闭环，随后以相同4000次更新预算检验道具观察是否改善行为，不修改得分增量奖励。

## 环境与冻结规格

原生Chromium、单游戏、CPU、seeded-reset-v3。原生快照schema2新增powerups能力，只有只读字段和对象标识，未改变物理、RNG或拾取计分规则。

task-v2-powerups：前62项与旧task-v1完全一致；末尾4个最近道具槽，每槽12项，总计110。槽位为mask、dx/20、dy/15、下一tick边界限制前dx、dy、power、六项类型独热编码。原始世界距离排序，同距按ID，ID不进入网络，缺失槽全零。类型是护盾、超级护盾、维修和三类弹药；power是原生补给系数，不是奖励。所有活跃道具均可进入选择，包括屏幕外生成的道具，容量4不保证保留所有物体。

正式候选预算：4000更新、最多5000决策、每局250决策，其余TrainConfig不变（游戏31、网络7、回放7、探索11；epsilon0.2，gamma0.99，SGD0.001，批量32，预填256，同步100）。对照为旧4000更新模型train-64dc3e5027014dddaf17d71b6506385e。新增输入必然改变第一层尺寸及初始化随机数消耗，不能声称两网络初始参数逐项相同。隐藏层宽度、动作与奖励保持相同。

## 命令

从仓库根目录：

```bash
.venv/bin/python exercises/chromium_dqn/check_task.py
RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 4000
```

训练打印task-v2-powerups及110维；完成后复制其打印的明确指定新policy.pt的评测命令。评测仍为模型/随机各20局、种子201..220、每局250决策。回放也应显式指定新checkpoint。旧权重根据保存的task-v1选择原62维编码，拒绝未知版本或维度不匹配；不通过截断/补零强行加载。

## 已执行验证与产物

- C++完整本地构建通过；子仓库13项测试全过，含真实道具位移、快照/绘图不推进状态、ID和reset复现。
- 独立runtime的原生测试覆盖ID消失后不复用、render_each_step开关轨迹一致。道具观察人工测试覆盖六类型、尺度、同距排序、截取容量、空槽及旧62维前缀。
- task、train、evaluate检查通过；旧policy加载通过。
- 教师仅冒烟训练1次更新：`runs/train-ef13f6212a1b4f85986870777412610a/`，256决策、1局结束、参数改变、loss19.661928、0.727秒。
- 新权重seed101经GUI回放路径执行206决策、5075分、hero_dead，参数不变；通过evaluate_episode另跑一局得到相同决策/得分，20.6模拟秒。未人工观察GUI画面，不称为视觉行为验收。
- 旧4000更新模型seed101在扩展后的原生版本仍为192决策、5000分、hero_dead，与此前数据一致。
- 生成功能验证日志位于/tmp/chromium-powerups-build.log、/tmp/chromium-native-regression.log、/tmp/chromium-powerups-tests.log、/tmp/chromium-v2-smoke.log、/tmp/chromium-v2-play.log；临时日志与runs不提交。

## 结果与边界

接口和新旧模型运行连接已经验证；尚无110维模型4000次更新的对照成绩。1次更新模型是工程冒烟产物，不能据此宣称学会接道具或生存改善。真实验证使用自然生成道具，不是六种道具每种拾取效果的穷举验收。

下一步用正式预算从头训练110维模型，比较中位分、平均存活和死亡/截断/过关数。旧4000更新对照：中位5000、平均存活19.326秒、20局全部死亡。原始得分依然混合击杀、拾取和漏接收益，新增观察并未解决奖励偏好问题。
