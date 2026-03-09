# Visual-Speech Intent Grounding (VSIG) Task Description

This document details the task definition, instruction types, and data annotation format for the VSIG Benchmark.

## Task Overview

VSIG requires an agent to act as an active listener and observer, completing the following tasks based on first-person video and user voice instructions:

1.  **Intent Grounding**: Decode ambiguous latent intents (e.g., "Put this there") into explicit executable commands (e.g., "Put the apple on the table") and select the corresponding object (Uppercase A-Z) or spatial location (Lowercase a-z) from a given list of options.
2.  **Spatial Grounding**: Output precise pixel coordinates `(x, y)` for each mentioned object or spatial location.
3.  **Temporal Grounding**: Output the peak timestamp (in milliseconds) when each mentioned object or spatial location is pointed to or mentioned in the video.

## Instruction Types and Evaluation

VSIG includes 6 types of instructions with increasing complexity:

| Instruction | Action Description | Speech Content | Intent Grounding (Explicit Command) | Evaluation Items (Sequence of Referents) |
| :--- | :--- | :--- | :--- | :--- |
| **Inst 1** | Point to an object | (No speech) | "[Object Name]" | `[Object]` |
| **Inst 2** | Point to an object | "Pick this up" | "Pick up the [Object Name]" | `[Object]` |
| **Inst 3** | Point to Object A, then to empty area a | "Put this here" | "Put the [Object A] in [Area a]" | `[Object, Space]` |
| **Inst 4** | Point to Object A, then to a relative area b near reference Object B | "Put this to its [Direction]" | "Put the [Object A] to the [Direction] of [Object B]" | `[Object, Space]` |
| **Inst 5** | Point to A, then to area b near B, then to Object C | "Put this to its [Direction], then pick it up" | "Put [Object A] to the [Direction] of [Object B], then pick up [Object C]" | `[Object, Space, Object]` |
| **Inst 6** | Point to A -> area b near B -> C -> area d near D | "Put this to its [Direction], then put this to its [Direction]" | "Put [Object A] to the [Direction] of [Object B], then put [Object C] to the [Direction] of [Object D]" | `[Object, Space, Object, Space]` |

### Evaluation Metrics

*   **Acc_cls (Classification Accuracy)**: Whether the predicted sequence of options (A, B, a, b...) perfectly matches the GT.
*   **Acc_s (Spatial Accuracy)**: For each Referent, whether the predicted point falls within the Object Mask or if the distance to the average of Space GT Points is less than 100 pixels.
*   **Acc_t (Temporal Accuracy)**: For each Referent, whether the predicted timestamp falls within the corresponding `stroke_begin_time` and `stroke_end_time` range.
*   **Acc_eco (Referent Accuracy)**: Accuracy of a single Referent where classification, spatial, and temporal grounding are all correct (cls ∧ s ∧ t).
*   **Acc_seq (Sequence Accuracy)**: Accuracy where all Referents in the entire sample are correct (Acc_eco = 1.0).

## Annotation Format Details

VSIG consists of two core annotation files: `annotations.json` (raw annotations) and `eval_gt.json` (evaluation ground truth).

### 1. annotations.json (Raw Annotations)

Records object/space attributes in the video, pixel coordinates of hand gestures, and ASR results.

```json
{
  "id": "1769185743034",
  "folder": "data/env1/Inst1",
  "video_name": "2025_12_30_14_10_IMG_6431.mp4",
  "task_template": "指令1",
  "scene": "Office Scene",
  "object_space": [
    {
      "name": "The leftmost black mouse on the table",
      "points": [[290, 324]],  // Normalized coordinates [y, x] (1-1000)
      "type": "object",
      "mask": {
        "bbox": [573, 293, 677, 350],
        "mask_base64": "...",
        "score": 0.89
      }
    }
  ],
  "asr_result": {
    "text": "Oh",
    "words": [
      { "text": "Oh", "begin_time": 280, "end_time": 480 }
    ]
  }
}
```

### 2. eval_gt.json (Evaluation Ground Truth - Core)

This is the file directly read by the evaluation system (Evaluator), mapping information from `annotations.json` into the sequence of options the model needs to output.

```json
{
  "2025_12_30_14_10_IMG_6431.mp4": {
    "object_choices": [
      "A. White protector",
      "F. Black mouse"
    ],
    "space_choices": [
      "a. The empty area to the left of the black mouse"
    ],
    "answer": [
      {
        "choice": "F",
        "stroke_begin_time": 2428,  // Start time of gesture/speech mentioning the object
        "stroke_end_time": 3904,    // End time
        "points": [[290, 324]],     // Corresponding annotated coordinates
        "mask": { ... }             // If it's an object, contains Mask for spatial evaluation
      },
      {
        "choice": "a",
        "stroke_begin_time": 4500,
        "stroke_end_time": 5000,
        "points": [[100, 200]]      // If it's a space, points are used for distance evaluation
      }
    ]
  }
}
```

## Notes

1.  **Coordinate System**: All `points` and `mask.bbox` are stored internally in `eval_gt.json` and `annotations.json` as `[y, x]` normalized format (1-1000). However, the **model output** and **evaluation logic** consistently use `[x, y]` pixel coordinates or `[x, y]` normalized coordinates. These conversions are handled automatically during evaluation.
2.  **Timestamps**: Timestamps predicted by the model should be in milliseconds. Evaluation checks if this value is within the closed interval `[stroke_begin_time, stroke_end_time]`.
3.  **Multi-turn Interactions (Inst 3-6)**: The model must output a `selected_options` and `point_list` of the corresponding length. For example, Inst 6 requires outputting 4 referents.
4.  **Option Lists**: The `object_choices` and `space_choices` for each video may vary. The model needs to make selections based on the options provided in the `user_prompt`.
