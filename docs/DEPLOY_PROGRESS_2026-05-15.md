# VINGS-Mono Jetson 部署进展总结（2026-05-15）

## 一、当前所处阶段
- **核心依赖已打通**：成功完成了 C++/CUDA 底层扩展在 Jetson Orin NX (sm_87) 架构下的重新编译。
- **纯视觉跑通前期帧**：系统已经越过了此前初始帧的显存、初始化等各种崩溃点，进入了持续的数据帧读取与处理阶段，目前跑通了约 15% 的数据集。

## 二、最新已完成/已验证的事项

### 1) 架构编译修复 (CUDA sm_87)
- `submodules/dbaf/setup.py` 中针对 `sm_87` 的编译已被验证生效。
- `droid_backends` 和 `lietorch` 已可以在 `vings_jetson` conda 环境下成功被导入，前段相关性计算路径的算子调用正常，去除了曾经的 `no kernel image is available` 错误。

### 2) 运行时配置与路径修复
- 在执行冒烟测试时，排除了两处运行时路径异常：
  - **权限异常**：默认 config 尝试写入 `/data` 目录导致 `[Errno 13] Permission denied`，通过指定 `--output-dir /home/jetson/VINGS-Mono/output` 参数解决。
  - **权重丢失**：修复了相对路径引起的 `FileNotFoundError`，显式指定 `--frontend-weight /home/jetson/VINGS-Mono/ckpts/droid.pth`。
- 测试采用了 `configs/hierarchical/smallcity.yaml` 和较低显存参数：`--frontend-buffer 80 --frontend-image-size 256,448 --no-vis --disable-metric --skip-save-ply`。

### 3) 冒烟测试新里程碑
- 成功越过之前的卡点（原最多跑至 67 帧），测试进程能够稳定追踪至 **第 133 帧**。

## 三、当前出现的新阻断点 (Bug)

**现象：**
在推断到约第 133 帧时，前端发生了异常退出。程序触发了前端历史帧窗口回收逻辑 `__rollup` 时报错。

**报错日志摘要：**
```python
Traceback (most recent call last):
  ...
  File "/home/jetson/VINGS-Mono/scripts/frontend/dbaf_frontend.py", line 159, in __rollup
    self.video.state.timestamps           = self.video.state.timestamps         [roll:]
AttributeError: 'NoneType' object has no attribute 'timestamps'
```

**原因分析：**
在 `scripts/frontend/dbaf_frontend.py` 里的 `__rollup` 方法中，系统尝试对旧的历史状态缓存执行剔除操作。但当前运行的情况下（可能由于采用的是纯视觉或无 IMU 配置），`self.video.state` 为 `None`，导致在进行列表切片 `[roll:]` 时抛出了 `AttributeError`。这反映了前端在窗口滚动逻辑中对无 IMU / 无后端状态对象的边缘情况处理存在漏洞。

## 四、下一步建议执行计划

1. **修复前端 `__rollup` 逻辑**：
   - 修改 `scripts/frontend/dbaf_frontend.py`，在执行 `self.video.state.timestamps = ...` 等操作前，增加对 `self.video.state is not None` 或者 `self.video.imu_enabled` 的判断，确保仅在对应模块存在时才去截断 state 中的数据。
2. **继续推进冒烟测试**：
   - 修复完后再次运行 `smallcity` 数据集的 `run.py`，目标是成功将 877 帧的数据全部跑通不崩溃。
3. **性能与完整度验证**：
   - 若能无误跑完，则尝试恢复常规的（更高）分辨率或者 buffer，测试 Jetson 上的显存资源上限并寻找最佳稳定点配置。