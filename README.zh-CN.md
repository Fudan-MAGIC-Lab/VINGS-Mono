<p align="right">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <img src="docs/logo.png" width="80%" alt="VINGS-Mono：面向大规模场景的视觉惯性 Gaussian Splatting 单目 SLAM">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.08286"><img src="https://img.shields.io/badge/arXiv-2501.08286-b31b1b.svg" alt="arXiv 论文"></a>
  &nbsp;
  <a href="https://vings-mono.github.io/"><img src="https://img.shields.io/badge/项目主页-VINGS--Mono-16a085" alt="项目主页"></a>
  &nbsp;
  <a href="https://github.com/Fudan-MAGIC-Lab/VINGS-Mono"><img src="https://img.shields.io/badge/GitHub-代码仓库-6f42c1" alt="GitHub 代码仓库"></a>
  &nbsp;
  <a href="https://www.youtube.com/watch?v=vniTZcNj_tA&t=163s"><img src="https://img.shields.io/badge/YouTube-演示视频-ff0000" alt="YouTube 演示视频"></a>
  &nbsp;
  <a href="http://www.fudanmagiclab.com"><img src="https://img.shields.io/badge/复旦-MAGIC_Lab-1769aa" alt="复旦大学 MAGIC Lab"></a>
</p>

# 项目简介

VINGS-Mono 是一个面向大规模场景的单目视觉惯性 Gaussian Splatting SLAM 框架。系统使用 RGB 图像与可选的低频 IMU 数据进行位姿估计，并增量构建 2D Gaussian 地图，覆盖室内、室外以及公里级城市环境。

项目主页展示了从室内场景到长距离室外轨迹的重建结果：KITTI odom08 轨迹长 3.7 km，对应约 3,200 万个 Gaussian 椭球；KITTI-360 示例轨迹长 8.05 km，对应约 5,173 万个 Gaussian 椭球。更多可视化结果请参阅[项目主页](https://vings-mono.github.io/)和[演示视频](https://www.youtube.com/watch?v=vniTZcNj_tA&t=163s)。

## 核心能力

- **VIO Front End**：通过稠密束调整与不确定性估计，从 RGB 帧中恢复场景几何和相机位姿。
- **2D Gaussian Map**：使用 Sample-based Rasterizer、Score Manager 与 Pose Refinement 增量维护 Gaussian 地图。
- **NVS Loop Closure**：利用 Gaussian Splatting 的新视角合成能力进行回环检测，并通过图优化校正位姿与地图。
- **Dynamic Eraser**：处理真实室外场景中的动态物体，减少其对重建结果的影响。
- **移动端采集**：支持通过手机相机与低频 IMU 采集数据，由 GPU 服务器完成 Gaussian 地图训练。

## 快速导航

- [环境配置](#1-环境配置)
- [数据准备](#2-数据准备)
- [配置与运行](#3-配置与运行)
- [移动端应用](#移动端应用)
- [引用](#引用)

## 1. 环境配置

### 环境要求

- Python `3.9.19`
- CUDA `11.8`
- VIO 模式可使用 [`Promethe-us/gtsam` 的 `vio` 分支](https://github.com/Promethe-us/gtsam/tree/vio)

> 环境安装可能需要约一小时，请预留足够时间。

### 安装

```bash
git clone --recursive https://github.com/Fudan-MAGIC-Lab/VINGS-Mono
cd VINGS-Mono
bash set_env.sh
```

### 下载预训练权重

```bash
mkdir -p ckpts
cd ckpts

wget https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main/droid.pth
wget https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main/metric_depth_vit_small_800k.pth

mkdir -p lightglue
cd lightglue
wget https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main/superpoint.onnx
wget https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main/superpoint_lightglue.onnx

# 可选：FastSAM 权重
cd ..
wget https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main/FastSAM-x.pt
```

## 2. 数据准备

### Demo 1：SmallCity

- 数据来源：[Hierarchical 3DGS Dataset](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/)
- 请先阅读并同意原数据集许可。
- 项目使用 `Calibrations`，并选择俯视相机图像。
- 处理后的数据已上传至 [VINGS-Mono Dataset](https://huggingface.co/datasets/Promethe-us/VINGS-Mono-Dataset)。

### Demo 2：Hotel

- 数据由 [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) 采集，请先阅读并同意其许可。
- 项目使用的数据已上传至 [VINGS-Mono Dataset](https://huggingface.co/datasets/Promethe-us/VINGS-Mono-Dataset)。

### Waymo、KITTI 与 KITTI-360

请按照 [`docs/PREPARE_DATA.md`](docs/PREPARE_DATA.md) 中的说明准备数据。

## 3. 配置与运行

### 修改配置

运行前，请在 `configs/` 下对应的 YAML 文件中检查并修改：

- `dataset:root`：数据集根目录
- `output:save_dir`：输出目录
- `frontend:weight`：前端模型权重路径

在大规模场景中，可视化 BEV 地图与保存检查点比较耗时。如不需要这些功能，可在配置文件中将 `use_vis` 设为 `False`。

### 运行示例

```bash
# Demo 1：SmallCity
python scripts/run.py configs/hierarchical/smallcity.yaml

# Demo 2：Hotel
python scripts/run.py configs/rtg/hotel.yaml

# KITTI
python scripts/run.py configs/kitti/sync/kitti_2011_09_30_drive_0028.yaml

# KITTI-360
python scripts/run.py configs/kitti360/unsync/kitti360_2013_05_28_drive_0002.yaml
```

## 移动端应用

移动端应用仓库：[victkk/3DGS_SLAM_mobile_app](https://github.com/victkk/3DGS_SLAM_mobile_app.git)

Gaussian Splatting 训练需要 GPU 服务器。手机连接服务器后，可通过项目提供的移动端应用采集相机与 IMU 数据；仓库同时提供可直接安装的 Android APK。

## 致谢

本项目基于多项优秀研究与开源项目构建，感谢所有作者的工作与分享：

- [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
- [Taming3DGS](https://github.com/humansensinglab/taming-3dgs)
- [DBAFusion](https://github.com/GREAT-WHU/DBA-Fusion)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
- [OpenVINS](https://github.com/rpng/open_vins)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM)

## 引用

如果 VINGS-Mono 对你的研究有所帮助，请引用：

```bibtex
@article{wu2025vings,
  title={Vings-mono: Visual-inertial gaussian splatting monocular slam in large scenes},
  author={Wu, Ke and Zhang, Zicheng and Tie, Muer and Ai, Ziqing and Gan, Zhongxue and Ding, Wenchao},
  journal={arXiv preprint arXiv:2501.08286},
  year={2025}
}
```

## 联系方式

如遇实现问题或程序错误，可联系：

- Ke Wu：<kewu23@m.fudan.edu.cn>
- Zicheng Zhang：<zhangzc.fdfz@gmail.com>
