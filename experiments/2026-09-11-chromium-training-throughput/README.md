# 无窗口并行采样与训练吞吐对照

## 问题与假设

用户要求优先优化训练速度，支持N个环境且训练不弹GUI。假设：原生固定休眠、重复张量构造与串行采样存在开销。本轮属于工程性能优化；多个实现优化一起投入，不把总加速归因于某一个因素。策略质量不是本次验收目标。

## 实现

- 原生增加真正headless模式：无SDL视频/窗口、无GL上下文/纹理/字体、无音频设备；保留影响游戏状态的更新。
- 同步循环改为poll等待输入，替换每轮固定SDL_Delay(2)，收到命令立即处理，模拟tick仍0.02秒。
- N个独立原生进程，通过线程池并发step，批量策略推理，按env_id确定顺序收集。单一共享模型/经验池，仍然每条达到预填门槛的新经验更新一次，不减少更新次数来制造速度优势。
- 每环境独立reset；第i环境第k局种子为31+i+k*N。最终批次按剩余预算缩小，不越过max_updates，不把reset观察存为终止经验。
- TensorReplay预分配CPU张量，以环形覆盖最旧经验，按逻辑时间顺序映射采样索引；抽样仍独立随机、无放回，结果不与库存共享内存。
- 日志保留每条记录，取消每步强制flush，改为1000决策一次以及正常关闭时刷新。

## 环境与配置

本机CPU、torch单线程，task-v2-powerups110维、18动作、每动作5tick、每局250决策，seeded-reset-v3。无环境安装或系统驱动改动。对照使用4000更新，预填256、批量32、容量10000、SGD0.001、gamma0.99、epsilon0.2、目标同步100；网络/回放种子7、探索11、游戏31。单环境优化前后同任务同种子；跨N比较保持更新/经验数量，初始网络一致，但采样与更新时序会变化。

## 完整命令

```bash
# 正常训练，无需显示服务
.venv/bin/python exercises/chromium_dqn/train.py --run --max-updates 40000 --num-envs 8
# 顺序基准：N=1/4/8，各3轮4000更新
.venv/bin/python experiments/2026-09-11-chromium-training-throughput/benchmark.py
# 原生/连接/调度回归
PYTHONPATH=third_party/chromium-bsu-rl RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s third_party/chromium-bsu-rl/tests -v
RUN_CHROMIUM_GUI_TESTS=1 .venv/bin/python -m unittest discover -s exercises/chromium_dqn/tests -v
.venv/bin/python exercises/chromium_dqn/train.py --check
```

## 产物与实测

旧版普通运行：`runs/train-313698159f7e4829ac64bd7d48830c86/`，4255决策、4000更新、9.902秒。使用改动前HEAD的train.py，以及当时保留在系统临时目录中的 `chromium-bsu-rl-before-speed` 原生二进制；其他旧路径行为未改变。另有cProfile运行用于查开销，耗时不用于下表。

新基准全部配置/权重/日志路径列在`exercises/chromium_dqn/runs/throughput-benchmark.json`；复现实验脚本会生成新随机命名目录。基准阶段顺序执行各组，不同时运行争抢CPU的训练。

| 实现 | 4000更新耗时 | 更新/秒（约） |
| --- | ---: | ---: |
| 旧单环境 | 9.902秒（一次） | 404 |
| 新单环境 | 2.337秒（三次中位） | 1712 |
| 新4环境 | 2.212秒（三次中位） | 1808 |
| 新8环境 | 2.131秒（三次中位） | 1877 |

优化后单环境约4.24倍速度；8环境约4.65倍于旧版、比新单环境快约9.7%。环境数增加不是线性加速，目前共享学习更新占主要剩余开销。

## 正确性与证据边界

原生15项、独立项目7项回归通过；Replay、Agent、训练与评测独立检查通过。headless用无DISPLAY/WAYLAND_DISPLAY且无效SDL视频驱动验证，能真实生成敌机子弹并推进；GUI与headless完整轨迹相同。对改动前原生二进制，种子7/8/209分别104/100/110条快照/步骤结果完全一致。

4000更新单环境新旧checkpoint每个权重张量完全相同，奖励1278、最后loss20.54389190673828一致。新方案每个N的三次重复训练权重一致；所有试验恰好4255决策/4000更新。人工测试覆盖预算不能整除N、各自reset种子、终止前后观察不串局、环形经验池覆盖与抽样同原Replay一致。

这些是性能与运行正确性证据，不是新策略能力评测。并行不能作为与原有N=1策略的严格单因素观察实验；需要固定N后重新建立对照。批量评测已改为headless，GUI回放仍单独启用。日志/权重不提交。

## 下一步

用户可按上面的单预算命令训练，模型能力继续用同任务评测规约判断。无需增加训练步数以证明加速，也不为了速度减少梯度更新。


## 8环境40000更新验证

原生实际完成40255决策、40000更新、205局结束，耗时19.837秒（约2016更新/秒），最终loss4.798976、参数改变。检查点：`exercises/chromium_dqn/runs/train-9fe5f22d1cae4eb9b8128c62ed6d16a7/policy.pt`。训练运行期间无GUI；覆盖经验池填满并多次环形覆盖。尚未对该模型作能力评测，不把训练奖励11199与短训练累计奖励比较。
