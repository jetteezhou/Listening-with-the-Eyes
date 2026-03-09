# src/eval/metrics.py
import math
import numpy as np
import base64
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


class Evaluator:
    """
    EcoG Evaluation Metrics Calculator.
    EcoG 评估指标计算器。

    Metrics definition (consistent with paper):
    指标定义（论文一致）：
      Acc_cls : Mean classification accuracy per-referent (sequence position matching).
                per-referent 分类正确率的均值 (序列位置对应匹配)
      Acc_s   : Mean spatial localization accuracy per-referent.
                per-referent 空间定位正确率的均值
      Acc_t   : Mean temporal localization accuracy per-referent (based on stroke_begin_time to stroke_end_time window).
                per-referent 时间定位正确率的均值（基于 stroke_begin_time-stroke_end_time 窗口）
      Acc_eco : Mean eco_k = cls_k ∧ s_k ∧ t_k per-referent.
                per-referent eco_k = cls_k ∧ s_k ∧ t_k 的均值
      Acc_seq : All-or-Nothing, 1 if sample_acc_eco == 1.0, else 0.
                All-or-Nothing，sample_acc_eco == 1.0 时为1，否则为0
    """

    # ------------------------------------------------------------------ #
    #  Basic Utility Methods / 基础工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_single_point(pt):
        """Ensure input is a single point [x, y]; if multiple points, take the mean.
           确保输入为单点 [x, y]，若为多点则取均值。"""
        if not pt or not isinstance(pt, list):
            return None
        if isinstance(pt[0], list):
            return [sum(p[0] for p in pt) / len(pt),
                    sum(p[1] for p in pt) / len(pt)]
        return pt

    @staticmethod
    def _normalize_pred_to_pixel(pred_pt, width: int, height: int):
        """Convert normalized coordinates (0-1000) to pixel coordinates.
           将归一化坐标（0-1000）转换为像素坐标。"""
        pt = Evaluator._ensure_single_point(pred_pt)
        if pt is None:
            return None
        return [int(pt[0] * width / 1000), int(pt[1] * height / 1000)]

    @staticmethod
    def calculate_distance(pred_pt, gt_pt):
        """Calculate Euclidean distance between two points (pixel coordinates).
           计算两点欧氏距离（像素坐标）。"""
        p = Evaluator._ensure_single_point(pred_pt)
        g = Evaluator._ensure_single_point(gt_pt)
        if p is None or g is None:
            return float('inf')
        return math.sqrt((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2)

    @staticmethod
    def is_point_in_mask(point, mask_base64, bbox, width=1920, height=1080):
        """
        Check if pixel coordinate point falls within Base64 mask.
        检查像素坐标点是否落在 Base64 mask 内。
        bbox: [x1, y1, x2, y2] (pixel coordinates / 像素坐标)
        """
        if not mask_base64 or not bbox:
            return False
        p = Evaluator._ensure_single_point(point)
        if p is None:
            return False
        try:
            px, py = int(p[0]), int(p[1])
            x1, y1, x2, y2 = bbox
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                return False
            mask = cv2.imdecode(
                np.frombuffer(base64.b64decode(mask_base64), np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                return False
            # Full image size mask / 全图尺寸 mask
            if mask.shape[0] == height and mask.shape[1] == width:
                return mask[py, px] > 128
            # bbox cropped size mask / bbox 裁剪尺寸 mask
            mh, mw = mask.shape
            bh, bw = y2 - y1, x2 - x1
            ly = min(max(int((py - y1) * mh / bh) if bh > 0 else 0, 0), mh - 1)
            lx = min(max(int((px - x1) * mw / bw) if bw > 0 else 0, 0), mw - 1)
            return mask[ly, lx] > 128
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Single Referent Evaluation / 单 referent 评估
    # ------------------------------------------------------------------ #

    @staticmethod
    def _eval_spatial(pred_item, gt_item, width, height):
        """
        Evaluate spatial localization for a single referent.
        评估单个 referent 的空间定位。
        Returns 1.0 (correct) or 0.0 (incorrect).
        """
        pred_pt = pred_item.get("point")
        if not pred_pt:
            return 0.0
        pred_pixel = Evaluator._normalize_pred_to_pixel(pred_pt, width, height)
        if pred_pixel is None:
            return 0.0

        mask_info = gt_item.get("mask") or {}
        if mask_info.get("mask_base64"):
            hit = Evaluator.is_point_in_mask(
                pred_pixel,
                mask_info["mask_base64"],
                mask_info.get("bbox"),
                width=width, height=height
            )
            return 1.0 if hit else 0.0
        else:
            gt_points = gt_item.get("points", [])
            if not gt_points:
                return 0.0
            if isinstance(gt_points[0], list):
                gt_avg = [int(sum(p[0] for p in gt_points) / len(gt_points)),
                          int(sum(p[1] for p in gt_points) / len(gt_points))]
            else:
                gt_avg = [int(gt_points[0]), int(gt_points[1])]
            # Distance threshold 100 pixels / 距离阈值 100 像素
            return 1.0 if Evaluator.calculate_distance(pred_pixel, gt_avg) < 100 else 0.0

    @staticmethod
    def _eval_temporal(pred_item, gt_temporal_item):
        """
        Evaluate temporal localization for a single referent.
        评估单个 referent 的时间定位。
        Returns 1.0 (within stroke window) or 0.0.
        返回 1.0（在 stroke 窗口内）或 0.0。
        """
        peak_ms = pred_item.get("timestamp")
        if peak_ms is None or not isinstance(peak_ms, (int, float)):
            return 0.0
        start = gt_temporal_item.get("stroke_begin_time")
        end = gt_temporal_item.get("stroke_end_time")
        if start is None or end is None:
            return 0.0
        return 1.0 if start <= int(peak_ms) <= end else 0.0

    # ------------------------------------------------------------------ #
    #  Sample-level Evaluation / 样本级评估
    # ------------------------------------------------------------------ #

    @staticmethod
    def evaluate_sample(pred, formatted_gt):
        """
        Evaluate a single sample and return sample-level metric values.
        评估单个样本，返回所有指标的 sample-level 值。

        Args:
            pred: Model prediction (normalized 0-1000) / 模型预测（坐标为归一化 0-1000）
            formatted_gt: Formatted ground truth / GTFormatter 格式化后的 GT

        Returns:
            dict:
                acc_cls  (float): Mean classification accuracy per-referent.
                acc_s    (float): Mean spatial localization accuracy per-referent.
                acc_t    (float): Mean temporal localization accuracy per-referent.
                acc_eco  (float): Mean eco_k = cls∧s∧t accuracy per-referent.
                acc_seq  (float): 1.0 if and only if acc_eco == 1.0.
                pred_options (list): Predicted options / 预测选项
                gt_options   (list): Correct options / 正确选项
        """
        task_template = formatted_gt.get("task_template", "")
        has_temporal = True

        width = formatted_gt.get("_video_width", 1920)
        height = formatted_gt.get("_video_height", 1080)

        correct_options = formatted_gt.get("_correct_options", [])
        pred_selected = pred.get("selected_options", [])
        if isinstance(pred_selected, str):
            pred_selected = [pred_selected]
        elif not isinstance(pred_selected, list):
            pred_selected = []

        correct_norm = [str(a).strip() for a in correct_options]
        pred_norm    = [str(s).strip() for s in pred_selected]
        K = len(correct_norm)

        if K == 0:
            return {
                "acc_cls": 0.0, "acc_s": 0.0, "acc_t": 0.0,
                "acc_eco": 0.0, "acc_seq": 0.0,
                "pred_options": pred_norm, "gt_options": correct_norm,
            }

        pred_points      = pred.get("point_list", [])
        gt_items         = formatted_gt.get("_processed_gt", {}).get("items", [])
        gt_speech_temps  = formatted_gt.get("_gt_speech_temporal", [])

        cls_scores = []
        s_scores   = []
        t_scores   = []

        for k in range(K):
            # cls_k
            cls_scores.append(
                1.0 if (k < len(pred_norm) and pred_norm[k] == correct_norm[k]) else 0.0
            )
            # s_k
            if k < len(pred_points) and k < len(gt_items):
                s_scores.append(
                    Evaluator._eval_spatial(pred_points[k], gt_items[k], width, height)
                )
            else:
                s_scores.append(0.0)
            # t_k
            if has_temporal:
                if k < len(pred_points) and k < len(gt_speech_temps):
                    t_scores.append(
                        Evaluator._eval_temporal(pred_points[k], gt_speech_temps[k])
                    )
                else:
                    t_scores.append(0.0)

        acc_cls = sum(cls_scores) / K
        acc_s   = sum(s_scores)   / K
        acc_t   = sum(t_scores)   / K if has_temporal else 0.0

        # eco_k = cls_k ∧ s_k ∧ t_k
        eco_scores = [
            1.0 if (cls_scores[k] == 1.0 and s_scores[k] == 1.0 and
                    (not has_temporal or t_scores[k] == 1.0)) else 0.0
            for k in range(K)
        ]
        acc_eco = sum(eco_scores) / K
        acc_seq = 1.0 if all(e == 1.0 for e in eco_scores) else 0.0

        return {
            "acc_cls": acc_cls,
            "acc_s":   acc_s,
            "acc_t":   acc_t,
            "acc_eco": acc_eco,
            "acc_seq": acc_seq,
            "pred_options": pred_norm,
            "gt_options":   correct_norm,
        }

    # ------------------------------------------------------------------ #
    #  Batch Evaluation / 批量评估
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_score():
        return {"acc_cls": 0.0, "acc_s": 0.0, "acc_t": 0.0,
                "acc_eco": 0.0, "acc_seq": 0.0,
                "pred_options": [], "gt_options": []}

    @staticmethod
    def evaluate_batch(predictions, ground_truths, num_workers=None):
        """
        Batch evaluation, supports multi-threading.
        批量评估，支持多线程并行。

        Returns:
            dict containing overall mean metrics, per-instruction breakdown, and detailed_results.
            dict 包含整体均值指标、per-instruction breakdown 和 detailed_results。
        """
        if not predictions or not ground_truths:
            return {}

        eval_args = list(zip(predictions, ground_truths))
        results_with_gt = []

        def _run(pred, gt):
            try:
                return Evaluator.evaluate_sample(pred, gt)
            except Exception as e:
                logging.error(f"Error evaluating sample: {e}")
                return Evaluator._default_score()

        if num_workers and num_workers > 1 and len(eval_args) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(eval_args))) as ex:
                # Maintain original order / 保持原始顺序
                futures = [(ex.submit(_run, p, g), p, g) for p, g in eval_args]
                for future, pred, gt in futures:
                    results_with_gt.append((future.result(), gt, pred))
        else:
            for pred, gt in eval_args:
                results_with_gt.append((_run(pred, gt), gt, pred))

        if not results_with_gt:
            return {}

        # ---- Aggregation / 汇总 ----
        all_scores = [x[0] for x in results_with_gt]

        def _mean(lst): return sum(lst) / len(lst) if lst else 0.0

        avg_cls = _mean([s["acc_cls"] for s in all_scores])
        avg_s   = _mean([s["acc_s"]   for s in all_scores])
        avg_t   = _mean([s["acc_t"]   for s in all_scores])
        avg_eco = _mean([s["acc_eco"] for s in all_scores])
        avg_seq = _mean([s["acc_seq"] for s in all_scores])

        # ---- Per-instruction breakdown / 按指令分组汇总 ----
        breakdown_raw = {}
        for score, gt, _ in results_with_gt:
            instr = gt.get("task_template", "Unknown")
            if instr not in breakdown_raw:
                breakdown_raw[instr] = {k: [] for k in
                                        ["acc_cls", "acc_s", "acc_t", "acc_eco", "acc_seq"]}
            for k in ["acc_cls", "acc_s", "acc_t", "acc_eco", "acc_seq"]:
                breakdown_raw[instr][k].append(score[k])

        instruction_breakdown = {
            instr: {k: _mean(v) for k, v in data.items()}
            | {"count": len(data["acc_cls"])}
            for instr, data in breakdown_raw.items()
        }

        return {
            "acc_cls": avg_cls,
            "acc_s":   avg_s,
            "acc_t":   avg_t,
            "acc_eco": avg_eco,
            "acc_seq": avg_seq,
            "instruction_breakdown": instruction_breakdown,
            "detailed_results": [
                {
                    "video_name":    gt.get("video_name"),
                    "instruction":   gt.get("task_template"),
                    "pred_options":  score["pred_options"],
                    "gt_options":    score["gt_options"],
                    "acc_cls":       score["acc_cls"],
                    "acc_s":         score["acc_s"],
                    "acc_t":         score["acc_t"],
                    "acc_eco":       score["acc_eco"],
                    "acc_seq":       score["acc_seq"],
                    # Debug/Visualization fields / 调试/可视化字段
                    "visualization_rel_path": pred.get("visualization_rel_path"),
                    "explicit_command":       pred.get("explicit_command"),
                    "reasoning":              pred.get("reasoning"),
                    "_model_input":           pred.get("_model_input"),
                    "_model_output":          pred.get("_model_output"),
                }
                for score, gt, pred in results_with_gt
            ],
        }
