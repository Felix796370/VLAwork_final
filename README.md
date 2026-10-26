# 10月26日 v2 三阶段仿真专家：精简运行仓库

只保留今天的“识别 → 抓取 → 面向正确挂架空位 → 挂载 → 松爪验证 → 回初始视角 → 重新识别”的运行依赖。原始 `new` 文件夹没有因本次精简被删除或改写；本仓库在其外部独立整理。

**状态：开发版，obj4第三阶段尚未通过，不能视为四类任务全部完成。**

- obj1、obj2、obj3有当前冻结代码的成功记录。
- 连续验收 seed=123，未指定类别列表，随机得到 `obj1,obj3,obj3,obj1`；挂架预置两件自由动态工件，连续完成四轮挂载并回位，最终视觉确认桌面清空。
- obj2单件验收 seed=6，桌面为 `obj4,obj2,obj3,obj2`，抓取其中一个obj2并完成全流程。
- obj4能抓起，但尚无通过全身碰撞安全检查的可靠挂载路径；脚本可能中止，不会把失败当成功。
- 这是仿真专家，不是PI0.5推理：第一阶段用标定RGB-D与已知CAD，第二、三阶段使用仿真位姿反馈和碰撞规划。它不是纯视觉端到端策略，也不是固定动作回放。
- `train_ready=false`：混合OSC/关节控制记录不能直接作为统一PI0.5动作标签。

## 从原目录中实际保留了什么

| 目录 | 保留内容及原因 |
| --- | --- |
| `hookrack_three_stage_v2_20261026/` | 11个冻结运行代码/参数文件，验收与审查工具，以及两次已通过的精简证据。没有复制失败实验、调试脚本和缓存。 |
| `rpent_hanging_benchmark_v2/extension/` | 工件/挂架注册和桌子、机械臂、挂架、摄像头的场景定义。 |
| `rpent_hanging_benchmark_v2/assets/hookrack/` | 四类工件和三个梳齿挂架的显示、完整CAD分解碰撞网格、墙体与空位识别元数据。网格文件数量较多，但不是历史垃圾，删掉会缺碰撞结构。 |
| `rpent_hanging_benchmark_v2/scripts/collect_expert_hanging.py` | 今天代码仍从它导入 `PROFILES` 抓取参数；仅保留这一实际依赖，不运行其旧CLI。 |
| `rpent_hanging_benchmark_v2/install.py`、`bddl/` | 在Python环境中注册自定义场景和四个任务。 |
| `hookrack_v2_standalone_20261019/base_assets/` | 仅基础桌面XML和三张必要纹理，不是整个10月19日版本。附带资源哈希与上游许可证。 |

`DEPENDENCY_AUDIT.json` 逐文件列出了来源相对路径、字节数、SHA-256及保留原因。复制的场景定义、CAD资产和基础资源还与本机验收时实际安装的版本逐一比对过。

没有上传：10月12日等旧ZIP、旧版专家/两阶段实现、全部开发失败录像、旧轨迹数据集、PI报告、服务器备份、Miniconda/venv、`__pycache__`、模型权重。旧轨迹已经提取成 `hanging_references.json`；其中 `source` 绝对路径仅作来源记录，运行时不会打开这些旧目录。

## 需要另外安装什么

需要Python、MuJoCo、robosuite、LIBERO-Pro、NumPy/SciPy、RGB-D/CAD处理和视频库。Panda及夹爪的通用资源由robosuite提供，不需要复制原 `new` 中的Python环境。

不需要RPent、RLinf推理服务、PI0.5权重或8220/8100端口。这里运行的是独立仿真专家。

下面是推荐的独立环境安装方式，避免改变已有 `rpent` 环境。版本以本机通过验收的Python3.12 / MuJoCo3.3.7 / robosuite1.4.1为依据；服务器EGL与其他版本组合尚未重新验收。

```bash
git clone https://github.com/Felix796370/VLAwork_final.git
cd VLAwork_final
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==1.26.4'
python -m pip install 'torch==2.3.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
export LIBERO_CONFIG_PATH="$PWD/.runtime/libero-config"
python bootstrap_scene.py
python bootstrap_scene.py --apply
```

`bootstrap_scene.py`先预览，`--apply`才安装。它只在激活的独立Python环境中安装场景，覆盖已有目标前保存备份，不删除文件。今天的专家自己生成每次初始化的BDDL，无需旧 `.pruned_init` 或重新生成四任务初始状态。

Windows建议克隆到较短路径（例如 `C:\vla\VLAwork_final`），并使用Python3.12的新虚拟环境。激活后执行同样的pip和bootstrap命令；设置环境变量用 `$env:LIBERO_CONFIG_PATH="$PWD\.runtime\libero-config"`，渲染参数改成 `--render-backend glfw`。LIBERO依赖包本身有超长文件名，必要时启用Windows长路径支持。

## 先检查文件和验收判据

```bash
python tools/verify_subset.py
cd hookrack_three_stage_v2_20261026
python -m unittest test_acceptance -v
cd ..
```

本次打包检查：3468个复制文件的哈希和3418处自定义MJCF文件引用通过检查，14项验收判据单元测试通过。还在一个隔离的LIBERO代码/资产副本中禁止访问原始 `new` 目录，成功初始化场景、读取双摄像头RGB-D并识别出随机的四件工件；见 `PACKAGING_VALIDATION.json`。该检查使用已安装的Python依赖，不是重新安装完整Python环境，也没有重跑一次完整挂载。

## 复现已通过的连续任务

以下命令在仓库根目录运行。`--benchmark-root` 指向本仓库的精简v2目录，而不是旧服务器目录。

```bash
export LIBERO_CONFIG_PATH="$PWD/.runtime/libero-config"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python hookrack_three_stage_v2_20261026/hang_part.py \
  --benchmark-root "$PWD/rpent_hanging_benchmark_v2" \
  --all --seed 123 --occupied free-sparse --render-backend egl \
  --output "$HOME/results/hookrack_1026_all_run001"
```

输出目录必须尚不存在。服务器EGL需要可用的显卡驱动；Python安装成功不代表EGL驱动正常。Windows将 `egl` 改为 `glfw`。

单件obj2复现：

```bash
python hookrack_three_stage_v2_20261026/hang_part.py \
  --benchmark-root "$PWD/rpent_hanging_benchmark_v2" \
  --part obj2 --seed 6 --layout obj4,obj2,obj3,obj2 \
  --occupied free-sparse --render-backend egl \
  --output "$HOME/results/hookrack_1026_obj2_run001"
```

换seed会随机改变类型、数量组合、平移与偏航；不保证所有seed都通过。当前每次仍为四个桌面实例，允许重复和缺失类别，不是任意数量场景。

## 成功证据与输出

- [连续四件录像](hookrack_three_stage_v2_20261026/tests/verify_all_final_a/main_wrist.mp4)
- [连续四件报告](hookrack_three_stage_v2_20261026/tests/verify_all_final_a/report.json)
- [独立接触审计](hookrack_three_stage_v2_20261026/tests/verify_all_final_a/contact_audit.json)：20Hz的10,289帧，包含零力接触，最大数值穿透约0.427mm；不等于每个物理子步的数学零穿透保证。
- [obj2报告](hookrack_three_stage_v2_20261026/tests/verify_obj2_final_a/report.json)：独立接触审计最大数值穿透约0.783mm。

重新运行会保存 `report.json`、`main_wrist.mp4`、逐轮双摄像头识别图、`trace.json` 和 `trajectory.npz`。挂载成功需松开并移开夹爪后5秒仍由正确层圆头接触面承重，满足姿态、漂移、速度及穿透阈值。

本次精简不会重新宣称四类完成，也不会把旧固定轨迹的成功等同于当前随机初始化下的泛化能力。

为减少体积，附带证据没有复制完整的历史状态数组和接触轨迹。需要重新计算逐帧接触审计时，请先运行上面的命令产生新的 `trajectory.npz`，再使用 `audit_saved_contacts.py` 审计新输出；本仓库附带的是原已完成审计的报告。
