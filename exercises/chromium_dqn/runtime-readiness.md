# 原生接口核查记录（2026-09-10）

本记录用于新项目依赖验收，不导入旧代码到新项目。已有客户端仅用于检查既有原生程序，新Python通信层仍需重新实现和验证。

## 本轮证据

- 原生源码工作树：干净；HEAD：`23a1a36ee286c27afea617e375f140bec10900c2`。
- 测试对象：现有`third_party/chromium-bsu-rl/build/install/bin/chromium-bsu-rl`，本轮没有重新构建，不能单凭HEAD断言二进制与源码完全一致。
- 二进制SHA256：`033df972729e43169bc0b1d934022263f0602178197cded17249f79bfd08d51e`。
- 工作目录：`third_party/chromium-bsu-rl`。
- 命令：`RUN_CHROMIUM_GUI_TESTS=1 /home/jiang/code/legged_robot_rl/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`。
- 实际结果：`Ran 9 tests in 2.889s`，`OK`，无跳过；其中5项NativeTests、4项TransportTests。
- 原生检查使用X11显示路径。自动化启动及数据断言不等于人工视觉验收；不是headless证明或学习者实作。

## 覆盖与缺口

| 项目 | 本轮证据 | 仍需工作 |
|---|---|---|
| 同步动作 | 右移、释放、斜移、多tick、无命令不推进、非法输入拒绝通过 | 新runtime集成后复验；开火及全部18动作并未被本组测试逐项验收 |
| 子弹/绘图 | 稳定ID匹配、部分连续位移、绘图不推进断言通过 | 非所有类型/边界的穷举证据 |
| 单关终止 | 空闲推进直到hero_dead或level_over，随后step报错且快照不变 | 本测试不分别保证两种结束分支都走到 |
| 进程生命周期 | live读取、close及输入EOF退出通过；通信故障清理另有模拟测试 | 新runtime连续重开和超时仍需原生验证 |
| reset/seed | 源码SnapshotBridge.cpp的hello仍报告reset=false，命令分发无公开reset/seed | 补齐原生能力，构建并验证多次重置及同种子轨迹 |
| 奖励事实 | 本组不是得分、损命及资源语义的专项测试 | 新任务采用前补真实路径检查 |
| 新项目训练 | 无实现、无结果 | 教师新runtime与学习者核心实现后验证 |

接下来先处理原生reset/seed及新runtime契约；完成后才能把可运行的真实Task练习交给学习者。此清单不把尚未验证能力标为完成。
