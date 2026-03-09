# Listening with the Eyes: Benchmarking Egocentric Co-Speech Grounding across Space and Time

![Task Overview](assets/task.png)

这是论文 **"Listening with the Eyes: Benchmarking Egocentric Co-Speech Grounding across Space and Time"** 的官方开源仓库，包含 **EcoG-Bench** 评估基准的代码与数据说明。

---

## 🌟 任务简介

在实际的协作场景中，人类经常使用“故意不具体化”的指示语（例如：“把**那个**给我”）。这些指示语的真实意图只有通过将语音与视频中的短暂指向动作（Pointing Stroke）进行对齐才能被识别。

**EcoG (Egocentric Co-Speech Grounding)** 任务要求智能体在第一人称视角下，同时预测 **What**（意图解析）、**Where**（空间定位坐标）和 **When**（动作发生的时间戳）。

### 核心特性：
- **高质量数据集**：包含 **811** 个第一人称剪辑，涵盖工业、厨房、办公等多种真实场景。
- **精细化标注**：提供毫米级的时间轴动作监督和密集的空间 Mask 标注。
- **渐进式认知评估 (Progressive Cognitive Evaluation)**：按照任务复杂度（L1–L4）逐步评估模型的推理能力。
- **双语支持**：完整支持中 (ZH) 英 (EN) 双语指令。

> 📄 **详细文档**：关于任务定义、指令类型和数据标注格式的详细说明，请阅读 [任务说明文档 (task_info.md)](task_info.md)。

---

## 目录
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [1. 安装依赖](#1-安装依赖)
  - [2. 配置项目](#2-配置项目)
  - [3. 运行推理评估](#3-运行推理评估)
- [核心功能](#核心功能)
  - [Web 管理界面](#web-管理界面)
  - [消融实验](#消融实验)
- [实验结果](#实验结果)

---

## 项目结构

```text
.
├── data/                       # EcoG 数据集 (包含 data_en 和 data_zn)
├── src/                        # 核心源代码
│   ├── data_loader.py          # 数据加载器
│   ├── eval_engine.py          # 评估引擎 (核心逻辑)
│   ├── gt_formatter.py         # Ground Truth 格式化工具
│   ├── models/                 # 模型接口 (支持 OpenAI, Gemini, DashScope 等)
│   ├── prompts/                # Prompt 模板设计
│   ├── utils/                  # 视频处理与日志工具
│   └── eval/                   # 评估指标 (Accuracy, Distance 等)
├── webui/                      # Web 界面可视化系统
│   ├── frontend/               # 基于 React 的前端代码
│   └── backend/                # 基于 FastAPI 的后端服务
├── results/                    # 评估结果存放目录
├── config.py                   # 全局配置文件
├── main.py                     # 标准推理评估入口
├── run_temporal_anchor_ablation.py # 时间锚点消融实验入口
├── requirements.txt            # Python 依赖包列表
└── task_info.md                # 任务定义与标注详情
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.9+ 环境：

```bash
pip install -r requirements.txt
```

对于 WebUI 前端，需要安装 Node.js 并在 `webui/frontend` 下执行：

```bash
cd webui/frontend && npm install
```

### 2. 配置项目

项目配置主要通过 `config.py` 或 `.env` 文件管理。

1.  **API Keys**: 在 `.env` 中配置您的 API 密钥：
    ```bash
    OPENAI_API_KEY=your_key_here
    GEMINI_API_KEY=your_key_here
    # 其他配置如 MODEL_NAME, DATA_ROOT_DIR 等
    ```
2.  **详细配置**: 打开 `config.py` 可以修改 FPS、模型参数、并行线程数等。

### 3. 运行推理评估

运行标准的推理流程：

```bash
python main.py
```

评估结果将自动保存至 `results/` 目录下。

---

## 核心功能

### Web 管理界面

项目提供了一个直观的 Web 界面，用于管理评估任务、查看模型输出结果及可视化标注。

1.  **启动后端**:
    ```bash
    cd webui/backend && python app.py
    ```
2.  **启动前端 (开发模式)**:
    ```bash
    cd webui/frontend && npm run dev
    ```
3.  **访问**: 打开浏览器访问 `http://localhost:5173` (或后端指定的端口)。

### 消融实验

针对论文中的 **Temporal Anchor (时间锚点)** 进行消融实验：

```bash
python run_temporal_anchor_ablation.py
```
该脚本会对比在有无帧时间戳、有无 ASR 时间词对齐情况下的模型表现。

---

## 实验结果

### EcoG-Bench 主榜单结果

| 模型 | L1 (eco/seq/cls) | L2 (eco/seq/cls) | L3 (eco/seq/cls) | L4 (eco/seq/cls) | Overall (eco/seq/cls) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Native Omni Models** | | | | | |
| Gemini-3-Pro | 30.2 / 30.2 / 79.9 | 29.2 / 29.2 / 71.5 | 10.6 / 1.8 / 51.9 | 10.2 / 0.4 / 64.5 | 17.0 / 10.9 / 63.9 |
| Gemini-3-Flash | 12.2 / 12.2 / 71.2 | 10.2 / 10.2 / 69.3 | 3.2 / 0.7 / 65.3 | 6.6 / 0.0 / 79.5 | 7.0 / 4.1 / 71.4 |
| Qwen3-Omni-30B-A3B | 3.6 / 3.6 / 40.3 | 0.7 / 0.7 / 48.9 | 0.0 / 0.0 / 23.9 | 0.0 / 0.0 / 19.1 | 0.7 / 0.7 / 29.5 |
| Qwen3-Omni-Flash | 2.9 / 2.9 / 59.7 | 0.7 / 0.7 / 66.4 | 0.2 / 0.0 / 40.5 | 0.0 / 0.0 / 37.2 | 0.7 / 0.6 / 47.1 |
| **Vision-Language Models** | | | | | |
| Qwen3-VL-30B | 18.0 / 18.0 / 64.0 | 19.7 / 19.7 / 60.6 | 8.5 / 0.7 / 34.3 | 6.6 / 0.0 / 25.6 | 11.4 / 6.7 / 41.2 |
| Qwen3-VL-8B | 21.6 / 21.6 / 66.9 | 16.1 / 16.1 / 59.9 | 4.4 / 0.4 / 32.7 | 2.7 / 0.0 / 30.5 | 8.8 / 6.5 / 42.5 |
| GPT-5-mini | 5.0 / 5.0 / 74.8 | 2.9 / 2.9 / 56.9 | 2.8 / 0.4 / 38.4 | 3.2 / 0.0 / 47.3 | 3.3 / 1.5 / 50.5 |

> **指标说明**：
> *   **eco (Eco-Accuracy)**：当且仅当 "What"（意图）、"Where"（空间定位）和 "When"（时间对齐）全部正确时，该样本才判定为正确。
> *   **seq (Sequence Success)**：针对多步骤任务的序列成功率。
> *   **cls (Classification)**：物体或空间类别的分类准确率。
> *   **L1-L4**：代表任务的复杂程度级别。

详细的对比数据和图表建议通过 WebUI 或 `results/` 下的日志文件进行分析。
