# VINGS-Mono Jetson 部署进展总结（2026-04-17）

## 一、今日目标
- 在 Jetson Orin NX（L4T 35.6.3 / CUDA 11.4）上推进 VINGS-Mono 的 headless 部署。
- 先完成可运行链路（不追求完整功能），再逐步补齐重依赖。

## 二、今日遇到的主要问题

### 1) Conda 源配置异常导致环境创建失败
- 现象：`conda create -n vings_jetson python=3.8` 初期报 `UnavailableInvalidChannel: HTTP 404`。
- 根因：历史 TUNA 源配置里存在无效 aarch64 channel。
- 影响：环境创建被阻塞。

### 2) Python 依赖对 Jetson/aarch64 兼容性不完全
- 现象：`pip install -r requirements.txt` 失败。
- 核心报错：`triton==2.0.0` 无可用 aarch64 wheel（No matching distribution）。
- 影响：无法按仓库默认依赖一次性安装。

### 3) 网络超时与大包下载不稳定
- 现象：安装 onnx/opencv/open3d 等大包时多次 `Read timed out`。
- 影响：批量安装中断，环境不稳定。

### 4) numpy 版本/状态导致 torch 导入异常（中间阶段）
- 现象：曾出现 `AttributeError: module 'numpy' has no attribute 'ndarray'`。
- 影响：torch 导入一度失败。
- 备注：后续通过重新安装与依赖回滚，torch + CUDA 已恢复可用。

### 5) 可选重模块在“冒烟验证”阶段阻塞启动
- 现象：metric（依赖 mmcv/triton）与可视化链路（open3d 等）会在早期引入额外阻塞。
- 影响：即使只想先跑通 headless，也可能被非关键依赖卡住。

## 三、今日已优化与已完成进度

### 1) 基线探测完成
- 硬件/系统：Jetson Orin NX，L4T 35.6.3，CUDA 11.4。
- 资源：磁盘剩余约 853GB，内存约 16GB，Swap 约 8GB。

### 2) 独立环境已建立
- 已创建并使用 conda 环境：`vings_jetson`（Python 3.8.20）。
- 已修复 channel 问题，环境可正常激活。

### 3) 核心 GPU 运行时已打通
- 已安装本地 wheel：
  - torch: `2.1.0a0+41361538.nv23.06`（cp38 aarch64）
  - onnxruntime-gpu: `1.15.1`（cp38 aarch64）
- 验证结果：`torch.cuda.is_available() == True`，设备识别为 `Orin`。

### 4) 运行入口已完成 Jetson 化改造
- 已改造 [scripts/run.py](../scripts/run.py)：
  - 支持 CLI 覆盖：dataset root、output dir、frontend weight、device、`--no-vis`、`--disable-metric`。
  - 对 metric 依赖改为按需导入，避免默认强依赖阻塞。
  - 对可视化/导出路径做了更适合 headless 冒烟验证的处理。
- 已新增 [scripts/run_jetson_smallcity.sh](../scripts/run_jetson_smallcity.sh)：
  - 提供 SmallCity 的 Jetson headless 启动封装。
- 已新增 [docs/JETSON_DEPLOY.md](JETSON_DEPLOY.md)：
  - 记录 Jetson 优先路径和部署注意事项。

### 5) 最小依赖安装取得阶段性成功
- 已成功安装一批基础依赖（示例）：`pyyaml`、`opencv-python`、`psutil`、`pillow==10.2.0`、`tqdm==4.66.2`、`plyfile`。
- 仍未完成：`triton/mmcv/open3d` 等重依赖链。

## 四、当前状态评估（截止今天）
- 状态：**已具备“继续推进冒烟验证”的基础条件**。
- 风险：
  - 依赖链未完全闭合（尤其是 triton/mmcv）。
  - 网络超时会显著拖慢安装进度。
- 结论：
  - 当前策略正确：先确保 headless 最小链路，再逐步补齐可选能力。

## 五、下次继续时的建议执行顺序
1. 先做最小导入验证（torch/torchvision/cv2/yaml/psutil）。
2. 用 headless 参数做一次最小启动尝试，优先拿到“第一条真实运行栈”。
3. 若报 metric 相关错误，保持 `--disable-metric` 并继续前进。
4. 若报可视化相关错误，保持 `--no-vis`（必要时继续跳过导出）。
5. 最后再单独处理 mmcv/triton/open3d 的补齐与性能优化。

## 六、关键产出文件
- 运行入口改造：[scripts/run.py](../scripts/run.py)
- Jetson 启动脚本：[scripts/run_jetson_smallcity.sh](../scripts/run_jetson_smallcity.sh)
- Jetson 部署说明：[docs/JETSON_DEPLOY.md](JETSON_DEPLOY.md)
- 本进度总结：[docs/DEPLOY_PROGRESS_2026-04-17.md](DEPLOY_PROGRESS_2026-04-17.md)
