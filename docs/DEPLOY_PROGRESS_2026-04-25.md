# VINGS-Mono Jetson 部署进展总结（2026-04-25）

## 一、今日目标
- 在已有 Jetson 环境基础上继续推进 VINGS-Mono，从“依赖修复”进入“真实数据运行”。
- 优先打通 SmallCity headless 冒烟链路，定位并消除启动期与前几帧运行期阻塞。

## 二、今日遇到的主要问题

### 1) 资源就位后，进程仍被直接杀死（Exit 137）
- 现象：`run.py` 在打印权重路径后即 `Killed`，退出码 137。
- 根因：`DepthVideo` 初始化阶段的大量 `.share_memory_()` 张量分配触发系统层 kill（Jetson 上共享内存/统一内存压力敏感）。
- 影响：程序在真正进入前端跟踪前中断。

### 2) 数据根目录层级与配置期望不一致
- 现象：`--dataset-root /home/jetson/VINGS-Mono/data/smallcity` 时数据集长度为 0。
- 根因：实际可用目录位于 `.../data/smallcity/small_city`。
- 影响：出现“进程可退出但未实际处理帧”的空跑结果。

### 3) 进入真实帧处理后出现 CUDA Kernel 架构不匹配
- 现象：处理到前几帧时报错 `RuntimeError: CUDA error: no kernel image is available for execution on the device`。
- 触发位置：相关性计算路径（`droid_backends.corr_index_forward`）。
- 根因：`submodules/dbaf/setup.py` 的 CUDA 编译目标缺少 `sm_87`（Orin NX）。
- 影响：即使主流程启动成功，也无法完成前端核心算子执行。

## 三、今日已优化与已完成进度

### 1) 新依赖与扩展补齐
- 新增安装：`torch-scatter==2.1.2`、`scipy`、`matplotlib`、`pytz`。
- `diff-surfel-rasterization` 已完成 Jetson 本地编译安装。
- 针对 CUDA 11.4 兼容性，修复了 `conv.cu` 中 cooperative_groups API（`dim_threads()` -> `group_dim()`）。

### 2) 运行链路按需依赖改造（减少非关键阻塞）
- 已将多处 `gtsam/open3d/wandb` 调整为按需依赖，避免 VO/headless 冒烟阶段被可选模块阻塞。
- 已将 `run.py` 中 `StorageManager`、`LoopModel` 调整为懒加载，配置关闭时不触发相关重依赖导入。

### 3) 权重与数据接入验证完成
- 权重文件已到位：`ckpts/droid.pth`（约 16MB）。
- 数据文件已到位：`data/smallcity/small_city`（文件数 > 3000）。
- 数据集对象验证：
  - `.../data/smallcity` -> `len=0`
  - `.../data/smallcity/small_city` -> `len=877`

### 4) 137 问题已定位并修复
- 已在 `DepthVideo` 中引入共享内存开关（默认关闭），将关键张量分配从“强制共享内存”改为“可选共享内存”。
- 修复后验证：程序已可越过原先 137 阶段并进入真实帧处理。

### 5) 低显存运行参数已支持
- `run.py` 新增前端运行时覆盖参数：
  - `--frontend-buffer`
  - `--frontend-image-size`（`H,W`）
- 作用：在 Jetson 上快速切换低显存配置进行冒烟与排障。
- 验证结果：`--frontend-buffer 24` 对当前前端实现仍然过小，会在 `self.t1` 递增到缓冲上界时触发越界；`--frontend-buffer 80` 可以让冒烟稳定跑过前 65 帧并进入窗口回收阶段。

## 四、当前状态评估（截止今天）
- 状态：**已从“环境/依赖阻塞”推进到“CUDA 算子架构适配阻塞”**。
- 已确认有效进展：
  - 资源、依赖、主流程启动都已打通。
  - 程序可进入第 1 帧之后的真实推理路径。
- 当前验证结果：`droid_backends` 已在 `vings_jetson` 环境中完成重编译并可正常导入；`smallcity` headless 冒烟在 `--frontend-buffer 80`、`--frontend-image-size 256,448`、`--no-vis`、`--disable-metric`、`--skip-save-ply` 下已稳定推进到 67/877 帧，没有新的 traceback。
- 当前唯一主阻塞：
  - `droid_backends` 需按 `sm_87` 重新编译并安装。

## 五、下次继续时的建议执行顺序
1. 在 `submodules/dbaf/setup.py` 保持 `sm_87` 编译目标，重新完整执行 `python setup.py install`（低并发 `MAX_JOBS=1`）。
2. 重装完成后，先做最小导入/算子验证（`droid_backends` 相关函数可调用）。
3. 使用低显存参数重跑冒烟：
   - `--dataset-root /home/jetson/VINGS-Mono/data/smallcity/small_city`
  - `--frontend-buffer 80`
   - `--frontend-image-size 256,448`
   - `--no-vis --disable-metric --skip-save-ply`
4. 若可稳定跑过前若干帧，再逐步恢复更高分辨率和更大 buffer。
5. 最后再评估是否需要恢复 `use_shared_memory=true` 或继续保持 Jetson 安全配置。

## 六、关键产出文件
- 运行入口增强与懒加载：`scripts/run.py`
- 137 修复（共享内存可选化）：`scripts/frontend/depth_video.py`
- 权重加载峰值优化：`scripts/frontend/dbaf.py`
- gtsam 按需化：`scripts/frontend/multi_sensor.py`、`scripts/frontend/dbaf_frontend.py`、`scripts/vings_utils/gtsam_utils.py`、`scripts/vings_utils/middleware_utils.py`
- 可视化与日志依赖按需化：`scripts/gaussian/gaussian_base.py`、`scripts/gaussian/wandb_utils.py`
- CUDA 扩展兼容修复：`submodules/diff-surfel-rasterization/conv.cu`
- droid_backends 架构适配变更：`submodules/dbaf/setup.py`
- 本进度总结：`docs/DEPLOY_PROGRESS_2026-04-25.md`
