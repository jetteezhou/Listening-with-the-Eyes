# Visual-Speech Intent Grounding (VSIG) 任务说明

本文档详细介绍了 VSIG Benchmark 的任务定义、指令类型以及数据标注格式。

## 任务介绍

VSIG 要求智能体作为积极的倾听者和观察者，根据第一人称视频和用户语音指令完成以下任务：

1.  **意图解析 (Intent Grounding)**：将模糊的潜在意图（如“把这个放到那里”）解码为明确的可执行指令（Explicit Command，如“把苹果放在桌子上”），并从给定的选项列表中选择对应的物体（Uppercase A-Z）或空间位置（Lowercase a-z）。
2.  **空间定位 (Spatial Grounding)**：为每个被提及的物体或空间位置输出精确的像素坐标 `(x, y)`。
3.  **时间定位 (Temporal Grounding)**：为每个被提及的物体或空间位置输出其在视频中被指向或提及的峰值时间戳（单位：毫秒）。

## 指令类型与评测详述

VSIG 包含 6 种指令类型，复杂度逐级提升：

| 指令 | 动作描述 | 说话内容 | 意图解析 (Explicit Command) | 评测项 (Sequence of Referents) |
| :--- | :--- | :--- | :--- | :--- |
| **指令 1** | 手指向一个物体 | (无语音) | "[物体名称]" | `[Object]` |
| **指令 2** | 手指向一个物体 | "把这个拿起来" | "把[物体名称]拿起来" | `[Object]` |
| **指令 3** | 指向物体 A，随后指向空闲区域 a | "把这个放到这里" | "把[物体A]放到[区域a]" | `[Object, Space]` |
| **指令 4** | 指向物体 A，随后指向参考物 B 的某方位区域 b | "把这个放到它的[方位]" | "把[物体A]放到[物体B]的[方位]" | `[Object, Space]` |
| **指令 5** | 指向物体 A，指向参考物 B 的某方位区域 b，再指向物体 C | "把这个放到它的[方位]，然后把它拿起来" | "把[物体A]放到[物体B]的[方位]，然后把[物体C]拿起来" | `[Object, Space, Object]` |
| **指令 6** | 指向 A -> B 的方位区域 b -> C -> D 的方位区域 d | "把这个放到它的[方位]，然后再把这个放到它的[方位]" | "把[物体A]放到[物体B]的[方位]，然后把[物体C]放到[物体D]的[方位]" | `[Object, Space, Object, Space]` |

### 评测指标 (Evaluation Metrics)

*   **Acc_cls (Classification Accuracy)**: 预测的选项（A, B, a, b...）序列是否与 GT 完全一致。
*   **Acc_s (Spatial Accuracy)**: 对于每个 Referent，预测点是否落在 Object Mask 内，或与 Space GT Points 的平均距离小于 100 像素。
*   **Acc_t (Temporal Accuracy)**: 对于每个 Referent，预测的时间戳是否落在对应的 `stroke_begin_time` 和 `stroke_end_time` 范围内。
*   **Acc_eco (Referent Accuracy)**: 单个 Referent 的分类、空间、时间定位均正确（cls ∧ s ∧ t）。
*   **Acc_seq (Sequence Accuracy)**: 整个样本的所有 Referent 均正确（Acc_eco = 1.0）。

## 标注格式详解

VSIG 包含两类核心标注文件：`annotations.json`（原始标注）和 `eval_gt.json`（评估真值）。

### 1. annotations.json (原始标注)

记录了视频中的物体/空间属性、手势指向的像素坐标以及 ASR 结果。

```json
{
  "id": "1769185743034",
  "folder": "data/env1/指令1",
  "video_name": "2025_12_30_14_10_IMG_6431.mp4",
  "task_template": "指令1",
  "scene": "办公场景",
  "object_space": [
    {
      "name": "桌上最左边的黑色鼠标",
      "points": [[290, 324]],  // 归一化坐标 [y, x] (1-1000)
      "type": "object",
      "mask": {
        "bbox": [573, 293, 677, 350],
        "mask_base64": "...",
        "score": 0.89
      }
    }
  ],
  "asr_result": {
    "text": "哦",
    "words": [
      { "text": "哦", "begin_time": 280, "end_time": 480 }
    ]
  }
}
```

### 2. eval_gt.json (评估真值 - 核心)

这是评测系统（Evaluator）直接读取的文件，将 `annotations.json` 中的信息映射为模型需要输出的选项序列。

```json
{
  "2025_12_30_14_10_IMG_6431.mp4": {
    "object_choices": [
      "A. 白色保护器",
      "F. 黑色鼠标"
    ],
    "space_choices": [
      "a. 黑色鼠标左边的空闲区域"
    ],
    "answer": [
      {
        "choice": "F",
        "stroke_begin_time": 2428,  // 手势/语音提及物体的开始时间
        "stroke_end_time": 3904,    // 结束时间
        "points": [[290, 324]],     // 对应的标注坐标
        "mask": { ... }             // 若为物体，包含 Mask 用于空间评估
      },
      {
        "choice": "a",
        "stroke_begin_time": 4500,
        "stroke_end_time": 5000,
        "points": [[100, 200]]      // 若为空间，使用 points 进行距离评测
      }
    ]
  }
}
```

## 注意事项

1.  **坐标系**：所有 `points` 和 `mask.bbox` 在 `eval_gt.json` 和 `annotations.json` 中内部存储为 `[y, x]` 归一化格式（1-1000），但在**模型输出**和**评测逻辑**中统一使用 `[x, y]` 像素坐标或 `[x, y]` 归一化坐标。评测时会自动处理这些转换。
2.  **时间戳**：模型预测的时间戳应为毫秒。评测时会检查该值是否在 `[stroke_begin_time, stroke_end_time]` 闭区间内。
3.  **多轮交互 (指令 3-6)**：模型必须输出一个长度对应的 `selected_options` 和 `point_list`。例如，指令 6 需要输出 4 个 referents。
4.  **选项列表**：每个视频的 `object_choices` 和 `space_choices` 可能不同，模型需要根据 `user_prompt` 中提供的选项进行选择。
