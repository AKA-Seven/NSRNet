
# NSRNet: Noise-Robust Stego-Image Restoration Network

## 📘 项目简介

**本科毕设题目**：噪声鲁棒的隐写域图像复原方法研究

本项目旨在构建一个具有噪声鲁棒性的隐写域图像修复网络 —— **NSRNet**，能够在隐写图像中存在高斯噪声干扰的情况下，在隐写域完成去噪、去模糊、超分辨率任务，并有效恢复秘密图像。

### 🧠 NSRNet 模型结构图

![NSRNet](https://raw.githubusercontent.com/AKA-Seven/NSRNet/main/images/NSRNet.png)

---

## 🧠 关键词

- Stego-image (隐写图像)
- Noise-Robust (噪声鲁棒性)
- Image Restoration (图像复原)
- Deep Learning (深度学习)

---

## 🖼️ 可视化结果

### 🔍 隐写图像与覆盖图像对比

![stego](https://raw.githubusercontent.com/AKA-Seven/NSRNet/main/images/stego.jpg)

### 🧩 真实秘密图像与修复后图像对比

![secret](https://raw.githubusercontent.com/AKA-Seven/NSRNet/main/images/secret.jpg)

---

## 🔧 项目安装

1. 克隆项目代码：
   ```bash
   git clone https://github.com/AKA-Seven/NSRNet.git
   cd NSRNet
   ```

2. 创建并激活虚拟环境（推荐使用 Conda）：
   ```bash
   conda create -n your_env_name python=3.8 -y
   conda activate your_env_name
   ```

3. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 数据准备

1. 下载 DIV2K 数据集：
   ```bash
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
   ```

2. 解压到项目根目录下，确保文件结构如下：
   ```
   ├── NSRNet
   │   ├── DIV2K_train_HR
   │   ├── DIV2K_valid_HR
   │   ├── pretrained
   │   ├── ...
   ```

3. 将预训练模型权重放置到 `pretrained/` 文件夹中:

   你可以通过以下链接下载预训练权重（Google Drive）：
   [点击下载预训练模型](https://drive.google.com/drive/folders/1u6FkmfDke0oYWwm41kQHWHWx17wCEQoa?usp=sharing)

---

## 🚀 运行方法

- 训练&测试 LRH（隐写网络）：
  ```bash
  bash train_LRH.sh
  ```

- 联合训练&测试 SPD & LRH：
  ```bash
  bash finetune.sh
  ```

- 训练&测试 NSRNet（完整隐写域复原网络，包括 LRH+SPD+LSR）：
  ```bash
  bash train_LSR.sh
  ```

---
