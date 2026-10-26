# v2 第三阶段：挂载与多件循环（2026-10-26，开发验证中）

本目录独立于 `hookrack_two_stage_v2_20261022`；不修改 v2/v3 场景资产、不删除旧文件。不连接服务器。
依赖已安装的 raised-parallel v2 场景和原 MuJoCo/robosuite 环境，不包含 Miniconda、模型权重或 RPent。

## 实际完成范围（未完成四类验收）

本包是开发版，不是已完成的四类发布版。obj4 已能抓起，但第三阶段在当前不变的场景中仍未找到通过机械臂全身碰撞检查的可靠挂载路径；会失败中止，不应把它视为可用的 obj4 专家。没有改桌高、碰撞网格、质量、摩擦或力矩上限来规避失败。

- `tests/verify_all_final_a`：seed=123，不传 layout，随机得到 obj1/obj3/obj3/obj1；挂架预置两件自由动态挂件。四次完整抓取、找空位、挂载、松爪、回初始位均成功，重新识别确认桌面清空。代码全程未变。
- 上述连续录像的独立离线接触审计覆盖 10,289 帧（20Hz，包括零力碰撞），最大穿透约 0.427mm，小于 1mm 数值阈值；这不是对每个物理子步的零穿透保证。
- `tests/verify_obj2_final_a`：seed=6、layout=obj4,obj2,obj3,obj2，选择 obj2；单件全流程成功，代码全程未变。
- 因此已验证 obj1、obj2、obj3，但没有 obj4 的第三阶段成功记录，也没有多随机种子的统计成功率。不要用其他 seed 的运行结果替代实际检查。

原两阶段版本以及 v2/v3 资产保持原样；本包不执行安装覆盖或删除。服务器尚未验证。

## 接口

```python
from hang_part import HangingSession
with HangingSession(benchmark_root='/home/zym/rpent_work/hookrack_v2_raised_parallel_20261019',
                    output='/home/zym/results/hanging_cycle_001', seed=123,
                    render_backend='egl') as arm:
    arm.return_to_initial_pose()   # 空夹爪，无业务参数
    parts = arm.perceive()         # 摄像头识别类别、位置；允许重复和缺失类别
    kind = parts[0]['kind']
    arm.grasp_part(kind)
    arm.face_rack(kind)
    arm.hang_part(kind)            # 挂载、松爪、稳定性验证、返回初始视角
```

同一会话调用 `arm.run_all()`：每次从新的桌面图像识别剩余工件，选择一个、抓取、找空位、挂载、回位，再重新识别。不是从初始化类别列表中读出待处理顺序。仿真真值仅用于识别拒绝审计以及第二、三阶段的仿真专家反馈/碰撞规划，不是纯视觉策略，更不是 PI0.5 推理。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rpent
cd ~/rpent_work/hookrack_three_stage_v2_20261026
export LIBERO_TYPE=pro MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python hang_part.py --benchmark-root ~/rpent_work/hookrack_v2_raised_parallel_20261019 --part obj2 --seed 6 --layout obj4,obj2,obj3,obj2 --occupied free-sparse --render-backend egl --output ~/results/hang_obj2_run001
```

连续清空桌面（输出目录必须不存在）：

```bash
python hang_part.py --benchmark-root ~/rpent_work/hookrack_v2_raised_parallel_20261019 --all --seed 123 --occupied free-sparse --render-backend egl --output ~/results/hang_all_run001
```

`--layout` 省略时随机四个实例，类别有放回采样；保留桌内、非重叠、随机偏航和可达性筛选。`--occupied free-sparse` 在对应层预置两件自由动态挂件，使用原 CAD、质量、摩擦和重力；初始化后检查圆头接触支撑，并在执行中检查它们是否掉落/移位。本轮机械臂新挂上的工件同样是自由动态物体，无焊接、无吸附、无中途重定位，会继续留在仿真中，影响后续视觉和碰撞规划。旧的 `--occupied sparse` / `almost-full` 仅为固定 CAD 占位回归模式，不能把该模式当作预置件的重力验证。

## 验收与记录

- 圆头抬过齿尖、向内送入、下降承重；按实际夹持变换计算动作，不回放固定时间轨迹。
- 松开并移开夹爪后连续 5 秒验证：正确层梳齿接触、圆头水平承重面承担至少 90% 重力、杆身承重小于 5% 重力（允许颈部轻微侧向接触，侧向法向力也不得超过 25% 重力）、姿态竖直、漂移小于 3mm、目标位置误差小于 12mm、线速度小于 1cm/s、角速度小于 0.05rad/s；数值穿透必须小于 1mm。
- 不能通过持续夹持、杆身卡住、错层、漂浮、掉落来获得成功。
- 连续循环中，先前已挂载的工件若位移超过 8mm 则中止。
- 空位和剩余工件来自新 RGB-D 图像；无法识别完整时中止，不把漏检当清空。
- `report.json` 为实际结果；代码存在、单元测试通过不等于物理验收通过。
- `main_wrist.mp4` 是实际双视角执行录像；`trace.json` 保存阶段/接触；`cycle_XX/` 保存逐轮挂架识别图；`table_scan_XX/` 保存逐轮桌面识别图。
- `trajectory.npz` 含混合 OSC 和关节控制记录，`train_ready=false`，不能直接当成统一 PI0.5 动作标签。

本地 GLFW 验证不等于服务器 EGL 验证。开发尝试和失败证据保留在 `tests/`。最终通过范围应以实际验收汇总为准。
