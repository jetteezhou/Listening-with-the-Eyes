"""
GT格式化模块：从annotation和description中提取评估所需的GT格式
统一预处理所有后续逻辑需要的信息：视频分辨率、最后一帧、坐标转换等
"""
import os
import json
import cv2
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.video_processor import VideoProcessor


class GTFormatter:
    """GT格式化器：统一处理GT数据的格式化和提取"""

    # GT 处理缓存
    _gt_cache = {}

    @staticmethod
    def _normalize_to_pixel_coords(points, width: int, height: int):
        """
        将归一化坐标（0-1000）转换为像素坐标。

        Args:
            points: 归一化坐标 [x, y] 或 [[x, y], ...]
            width: 视频宽度
            height: 视频高度

        Returns:
            像素坐标 [x, y] 或 [[x, y], ...]
        """
        if not points:
            return points

        if isinstance(points[0], list):
            # 多个点
            return [[int(p[0] * width / 1000), int(p[1] * height / 1000)] for p in points]
        else:
            # 单个点
            return [int(points[0] * width / 1000), int(points[1] * height / 1000)]

    @staticmethod
    def _extract_video_info(video_path: str, extract_last_frame: bool = False) -> Dict:
        """
        提取视频信息：分辨率、最后一帧路径等。

        Args:
            video_path: 视频文件路径
            extract_last_frame: 是否提取最后一帧（默认False，因为很慢，延迟到需要时再提取）

        Returns:
            包含 width, height, last_frame_path 的字典
        """
        info = {
            "width": 1920,
            "height": 1080,
            "last_frame_path": None
        }

        if not os.path.exists(video_path):
            return info

        try:
            # 提取视频分辨率（快速操作）
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

            # 提取最后一帧（慢操作，默认不提取，延迟到需要时）
            if extract_last_frame:
                try:
                    _, last_frame_path = VideoProcessor.extract_frame(
                        video_path, timestamp_sec=None)
                    info["last_frame_path"] = last_frame_path
                except Exception:
                    pass  # 如果提取失败，保持 None
        except Exception:
            pass  # 如果出错，使用默认值

        return info

    @staticmethod
    def load_eval_gt(file_path: str) -> Tuple[str, Dict]:
        """
        加载 eval_gt.json 文件，提取选项定义和全量答案信息。
        """
        if not os.path.exists(file_path):
            return "", {}

        with open(file_path, 'r', encoding='utf-8') as f:
            video_eval_data = json.load(f)

        if not video_eval_data:
            return "", {}

        # 取任意一个视频的 choices 生成全局 options_text（仅供选项全局共享的数据集兼容使用）
        # 注意：若每个视频有独立选项集（如 max_choices 数据集），process_single_sample 中会
        # 从 formatted_gt["_object_choices"] / ["_space_choices"] 动态覆盖此值
        first_video = next(iter(video_eval_data.values()))
        obj_choices = first_video.get("object_choices", [])
        space_choices = first_video.get("space_choices", [])

        definitions = GTFormatter.build_options_text(obj_choices, space_choices)

        return definitions, video_eval_data

    @staticmethod
    def build_options_text(obj_choices: list, space_choices: list) -> str:
        """
        从 object_choices 和 space_choices 列表构建发送给模型的选项文本。
        用于在 process_single_sample 中为每个视频动态生成专属选项文本。
        """
        definitions_parts = []
        if obj_choices:
            definitions_parts.append("物体 (Object):")
            definitions_parts.extend(obj_choices)
        if space_choices:
            if definitions_parts:
                definitions_parts.append("")
            definitions_parts.append("位置/空间 (Space):")
            definitions_parts.extend(space_choices)
        return "\n".join(definitions_parts).strip()

    @staticmethod
    def process_gt_by_template(gt: Dict, video_eval_info: Dict = None) -> Dict:
        """
        根据任务模板对 GT 数据进行预处理。

        重要：完全从 eval_gt.json 读取数据
        - 如果提供了 video_eval_info (来自 eval_gt.json)，则完全从其 answer 列表中提取信息
        - 每个 answer 项包含 choice, stroke_begin_time, stroke_end_time
        - 大写字母选项（物体）包含 mask 字段
        - 小写字母选项（空间位置）包含 points 字段（归一化 1-1000 [y, x] 格式）
        """
        # 使用 video_eval_info 的 JSON 字符串作为缓存键（如果存在）
        cache_key = json.dumps(video_eval_info, sort_keys=True, ensure_ascii=False) if video_eval_info else json.dumps(
            gt, sort_keys=True, ensure_ascii=False)

        if cache_key in GTFormatter._gt_cache:
            return GTFormatter._gt_cache[cache_key]

        template = gt.get("task_template")
        processed_gt = {
            "template": template,
            "items": []
        }

        # 获取分辨率用于反归一化
        width = gt.get("_video_width", 1920)
        height = gt.get("_video_height", 1080)

        # 必须从 eval_gt.json 的 answer 中提取数据，否则退出
        if not video_eval_info or "answer" not in video_eval_info:
            video_name = gt.get("video_name", "Unknown")
            raise ValueError(
                f"无法从 eval_gt.json 读取评估数据。视频: {video_name}。"
                f"请确保 eval_gt.json 文件存在且包含该视频的 answer 字段。"
            )

        answers = video_eval_info.get("answer", [])
        if not answers:
            video_name = gt.get("video_name", "Unknown")
            raise ValueError(
                f"eval_gt.json 中视频 {video_name} 的 answer 列表为空。"
                f"请检查 eval_gt.json 文件。"
            )

        for ans in answers:
            if not isinstance(ans, dict):
                continue

            choice = ans.get("choice", "")
            if not choice:
                continue

            # 判断是大写字母（物体）还是小写字母（空间位置）
            is_uppercase = choice.isupper()

            item = {
                "choice": choice,
                "stroke_begin_time": ans.get("stroke_begin_time", 0),
                "stroke_end_time": ans.get("stroke_end_time", 0),
                "type": "object" if is_uppercase else "space"
            }

            # 提取并转换 points (不管是物体还是空间)
            # eval_gt.json 中的 points 是 [y, x] 归一化格式 (1-1000)
            if "points" in ans and ans.get("points"):
                ans_points = ans["points"]
                # 1. 转换 xy 顺序 [y, x] -> [x, y]
                if isinstance(ans_points[0], list):
                    # [[y1, x1], [y2, x2]] -> [[x1, y1], [x2, y2]]
                    xy_points = [[p[1], p[0]] for p in ans_points]
                else:
                    # [y, x] -> [x, y]
                    xy_points = [ans_points[1], ans_points[0]]

                # 2. 反归一化为像素坐标
                item["points"] = GTFormatter._normalize_to_pixel_coords(
                    xy_points, width, height)

            # 大写字母选项：提取 mask（评估时只需要 mask，不需要 points）
            if is_uppercase and "mask" in ans:
                item["mask"] = ans["mask"]

            processed_gt["items"].append(item)

        if not processed_gt["items"]:
            video_name = gt.get("video_name", "Unknown")
            raise ValueError(
                f"无法从 eval_gt.json 提取任何有效的评估项。视频: {video_name}。"
                f"请检查 answer 列表中的数据结构。"
            )

        GTFormatter._gt_cache[cache_key] = processed_gt
        return processed_gt

    @staticmethod
    def format_gt_for_evaluation(
        annotation_item: Dict,
        video_dir: str,
        video_eval_info: Dict = None,
        video_width: int = None,
        video_height: int = None
    ) -> Dict:
        """
        格式化GT数据用于评估，统一预处理所有后续逻辑需要的信息。
        """
        # 确保有_video_dir字段
        formatted_gt = annotation_item.copy()
        formatted_gt["_video_dir"] = video_dir

        # 提取视频信息（如果未提供分辨率，则提取）
        video_name = annotation_item.get("video_name")
        video_path = os.path.join(
            video_dir, video_name) if video_name else None

        if video_width is not None and video_height is not None:
            # 使用提供的分辨率（批量提取时已缓存）
            width, height = video_width, video_height
        elif video_path:
            # 单独提取分辨率（慢，仅在单独调用时使用）
            video_info = GTFormatter._extract_video_info(
                video_path, extract_last_frame=False)
            width, height = video_info["width"], video_info["height"]
        else:
            width, height = 1920, 1080

        formatted_gt["_video_width"] = width
        formatted_gt["_video_height"] = height
        formatted_gt["_video_path"] = video_path  # 保存视频路径，需要时再提取最后一帧
        formatted_gt["_last_frame_path"] = None  # 延迟提取

        # 必须提供 video_eval_info，否则无法评估
        if not video_eval_info:
            video_name = annotation_item.get("video_name", "Unknown")
            raise ValueError(
                f"无法评估：视频 {video_name} 缺少 eval_gt.json 数据。"
                f"请确保 eval_gt.json 文件存在且包含该视频的评估信息。"
            )

        # 处理GT items（用于空间/语音时间评估）
        # process_gt_by_template 会从 eval_gt.json 的 answer 中提取数据
        processed_gt = GTFormatter.process_gt_by_template(
            formatted_gt, video_eval_info)

        # 保存处理后的 GT
        formatted_gt["_processed_gt"] = processed_gt

        # 提取正确答案和语音时间 GT
        answers = video_eval_info.get("answer", [])
        if not answers:
            video_name = annotation_item.get("video_name", "Unknown")
            raise ValueError(
                f"eval_gt.json 中视频 {video_name} 的 answer 列表为空。"
            )

        correct_options = []
        gt_speech_temporal = []
        for ans in answers:
            if isinstance(ans, dict):
                choice = ans.get("choice")
                if choice:
                    correct_options.append(choice)
                    gt_speech_temporal.append(ans)
            else:
                # 兼容旧格式（如果是字符串列表）
                correct_options.append(ans)

        # 保存 choices 信息，用于可视化时获取名称
        formatted_gt["_object_choices"] = video_eval_info.get(
            "object_choices", [])
        formatted_gt["_space_choices"] = video_eval_info.get(
            "space_choices", [])

        formatted_gt["_correct_options"] = correct_options
        formatted_gt["_gt_speech_temporal"] = gt_speech_temporal

        return formatted_gt

    @staticmethod
    def _extract_video_resolution_batch(video_paths: List[str], num_workers: int = 4) -> Dict[str, Tuple[int, int]]:
        """
        批量并行提取视频分辨率。

        Args:
            video_paths: 视频路径列表
            num_workers: 并行线程数

        Returns:
            {video_path: (width, height)} 映射
        """
        resolution_cache = {}

        if not video_paths:
            return resolution_cache

        def extract_single(video_path):
            if not os.path.exists(video_path):
                return video_path, (1920, 1080)
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    return video_path, (width, height)
                cap.release()
            except Exception:
                pass
            return video_path, (1920, 1080)

        # 并行提取分辨率
        if num_workers > 1 and len(video_paths) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(video_paths))) as executor:
                futures = {executor.submit(
                    extract_single, vp): vp for vp in video_paths}
                for future in as_completed(futures):
                    try:
                        video_path, resolution = future.result()
                        resolution_cache[video_path] = resolution
                    except Exception:
                        video_path = futures[future]
                        resolution_cache[video_path] = (1920, 1080)
        else:
            # 单线程模式
            for video_path in video_paths:
                _, resolution = extract_single(video_path)
                resolution_cache[video_path] = resolution

        return resolution_cache

    @staticmethod
    def format_batch_gt_for_evaluation(
        annotations: List[Dict],
        video_dir: str,
        video_eval_data: Dict = None,
        num_workers: int = 4
    ) -> List[Dict]:
        """
        批量格式化GT数据用于评估（并行提取视频分辨率以加快速度）
        """
        # 1. 批量并行提取所有视频的分辨率
        video_paths = []
        video_name_to_path = {}
        for item in annotations:
            video_name = item.get("video_name")
            if video_name:
                video_path = os.path.join(video_dir, video_name)
                video_paths.append(video_path)
                video_name_to_path[video_name] = video_path

        resolution_cache = GTFormatter._extract_video_resolution_batch(
            list(set(video_paths)), num_workers=num_workers
        )

        # 必须提供 video_eval_data，否则无法评估
        if not video_eval_data or len(video_eval_data) == 0:
            raise ValueError(
                "无法评估：缺少 eval_gt.json 数据。"
                "请确保 eval_gt.json 文件存在且包含评估信息。"
            )

        # 2. 格式化每个样本（使用缓存的分辨率）
        formatted_list = []
        for item in annotations:
            video_name = item.get("video_name")

            if not video_name:
                raise ValueError(
                    f"标注数据中缺少 video_name 字段。"
                )

            # 检查视频是否在评估列表中
            if video_eval_data and video_name not in video_eval_data:
                continue

            video_path = video_name_to_path.get(video_name)

            # 从缓存中获取分辨率
            if video_path and video_path in resolution_cache:
                width, height = resolution_cache[video_path]
            else:
                width, height = 1920, 1080

            video_eval_info = video_eval_data.get(video_name)
            try:
                formatted_gt = GTFormatter.format_gt_for_evaluation(
                    item, video_dir, video_eval_info, video_width=width, video_height=height
                )
                formatted_list.append(formatted_gt)
            except ValueError as e:
                # 如果单个样本格式化失败，跳过并记录（或者在 main.py 中通过长度差异发现）
                continue

        return formatted_list
