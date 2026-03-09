"""
数据加载模块：统一处理视频、annotation、description的读取
"""
import os
import json
from typing import List, Dict, Tuple, Optional


class DataLoader:
    """数据加载器：负责读取视频、annotation、description"""

    @staticmethod
    def load_annotations(annotation_path: str) -> List[Dict]:
        """
        加载annotation文件

        Args:
            annotation_path: annotations.json文件路径

        Returns:
            标注数据列表
        """
        if not os.path.exists(annotation_path):
            return []

        with open(annotation_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def list_videos(video_dir: str, extensions: List[str] = None) -> List[str]:
        """
        列出视频目录中的所有视频文件

        Args:
            video_dir: 视频目录路径
            extensions: 视频文件扩展名列表，默认为 ['.mp4', '.MOV', '.mov']

        Returns:
            视频文件名列表
        """
        if extensions is None:
            extensions = ['.mp4', '.MOV', '.mov']

        if not os.path.exists(video_dir):
            return []

        video_files = []
        for filename in os.listdir(video_dir):
            if any(filename.endswith(ext) for ext in extensions):
                video_files.append(filename)

        return sorted(video_files)

    @staticmethod
    def filter_annotations_by_videos(
        annotations: List[Dict],
        video_files: List[str]
    ) -> List[Dict]:
        """
        根据视频文件列表过滤annotation数据

        Args:
            annotations: 标注数据列表
            video_files: 视频文件名列表

        Returns:
            过滤后的标注数据列表
        """
        video_set = set(video_files)
        filtered = []

        for item in annotations:
            if item.get("video_name") in video_set:
                filtered.append(item)

        return filtered

    @staticmethod
    def prepare_dataset(
        video_dir: str,
        annotation_path: str
    ) -> Tuple[List[Dict], str, Dict]:
        """
        准备数据集：读取视频、annotation、eval_gt.json

        Args:
            video_dir: 视频目录路径
            annotation_path: annotations.json文件路径

        Returns:
            (数据集列表, 选项定义文本, 视频评估数据映射)
            数据集列表中的每个item都包含：
            - 原始annotation数据
            - _video_dir字段（用于后续评估时定位视频）
        """
        # 1. 读取annotation
        annotations = DataLoader.load_annotations(annotation_path)

        # 2. 列出视频文件
        video_files = DataLoader.list_videos(video_dir)

        # 3. 过滤annotation，只保留有对应视频的项
        dataset = DataLoader.filter_annotations_by_videos(
            annotations, video_files)

        # 3.5. 去重：根据id字段去除重复的标注条目（保留第一次出现）
        seen_ids = set()
        deduplicated_dataset = []
        for item in dataset:
            item_id = item.get("id")
            if item_id is None:
                # 如果没有id字段，保留该项（可能是旧数据格式）
                deduplicated_dataset.append(item)
            elif item_id not in seen_ids:
                seen_ids.add(item_id)
                deduplicated_dataset.append(item)
            # 如果id已存在，跳过该项（去重）
        dataset = deduplicated_dataset

        # 3.6. 根据文件夹名称强制覆盖 task_template 字段
        # annotations.json 中的 task_template 可能不正确（如全部标为"指令1"），
        # 以实际文件夹名称（如 "指令2"）作为唯一权威来源
        folder_name = os.path.basename(video_dir.rstrip(os.sep))
        if folder_name.startswith("指令"):
            for item in dataset:
                item["task_template"] = folder_name

        # 4. 为每个item添加_video_dir字段
        for item in dataset:
            item["_video_dir"] = video_dir

        # 5. 必须读取 eval_gt.json，否则无法评估
        options_text = ""
        video_eval_data = {}
        eval_gt_path = os.path.join(video_dir, "eval_gt.json")

        if not os.path.exists(eval_gt_path):
            raise FileNotFoundError(
                f"无法找到 eval_gt.json 文件: {eval_gt_path}。"
                f"请确保已生成 eval_gt.json 文件。"
            )

        from src.gt_formatter import GTFormatter
        options_text, video_eval_data = GTFormatter.load_eval_gt(eval_gt_path)

        if not video_eval_data:
            raise ValueError(
                f"eval_gt.json 文件为空或格式错误: {eval_gt_path}。"
                f"请检查文件内容。"
            )

        return dataset, options_text, video_eval_data

    @staticmethod
    def scan_data_root(data_root: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        扫描数据根目录，找到所有指令文件夹。
        支持两种结构：
        1. data_root/指令1/
        2. data_root/数据集/指令1/
        3. 甚至更深层次的结构，只要文件夹名为 '指令1-6' 且包含 annotations.json

        Args:
            data_root: 数据根目录路径

        Returns:
            {指令名: [(数据集名, 数据集路径, 指令路径), ...]}
        """
        instruction_dirs = {}
        valid_instructions = ["指令1", "指令2", "指令3", "指令4", "指令5", "指令6"]

        if not os.path.exists(data_root):
            return instruction_dirs

        # 使用 os.walk 进行递归扫描，提高鲁棒性
        for dirpath, dirnames, filenames in os.walk(data_root):
            dirname = os.path.basename(dirpath)
            
            # 检查是否是指令文件夹
            if dirname in valid_instructions and "annotations.json" in filenames:
                instruction_name = dirname
                item_path = dirpath
                
                # 向上取一级作为数据集路径和名称
                dataset_path = os.path.dirname(dirpath)
                dataset_name = os.path.basename(dataset_path)
                
                # 如果 dataset_name 为空（可能是根目录），使用 "default"
                if not dataset_name:
                    dataset_name = "default"
                
                if instruction_name not in instruction_dirs:
                    instruction_dirs[instruction_name] = []
                
                # 添加到结果中
                instruction_dirs[instruction_name].append(
                    (dataset_name, dataset_path, item_path)
                )

        # 对每个指令下的数据集进行排序，确保处理顺序一致
        for inst in instruction_dirs:
            instruction_dirs[inst].sort()

        return instruction_dirs
