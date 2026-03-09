# src/utils/video_processor.py
import cv2
import os
import numpy as np
import logging
import subprocess
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("EcoG_Logger")


class VideoProcessor:
    """
    Process video input and extract frames for model inference.
    处理视频输入，提取帧用于模型推理。
    """

    @staticmethod
    def extract_frames(video_path, num_frames=8, end_timestamp_sec=None, fps=None):
        """
        Extract multiple frames from video. Supports uniform sampling by count or sampling by FPS.
        从视频中提取多帧。支持按数量均匀提取或按FPS提取。

        Args:
            video_path (str): Path to video file / 视频文件路径。
            num_frames (int): Number of frames to extract (effective when fps is None).
                               提取帧的数量（当 fps 为 None 时生效）。
            end_timestamp_sec (float, optional): End timestamp. If provided, samples between 0 and end_timestamp_sec.
                                                 结束时间戳。如果提供，则在 0 到 end_timestamp_sec 之间抽样。
            fps (float, optional): Sampling frame rate. If provided, num_frames is ignored and sampling follows this rate.
                                   采样帧率。如果提供，将忽略 num_frames，按此帧率采样。

        Returns:
            list: List of paths to temporary images / 保存的临时图像路径列表。
            str: Path to the last frame (used for Grounding) / 最后一帧（用于 Grounding）的路径。
            list: List of timestamps (ms) for each frame / 每帧的时间戳（毫秒）列表。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Try ffmpeg as a fallback / 尝试使用 ffmpeg 作为备选方案
            logger.warning(f"OpenCV cannot open video: {video_path}, trying ffmpeg...")
            try:
                return VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames, end_timestamp_sec, fps)
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg also failed: {ffmpeg_error}")
                raise ValueError(f"Cannot open video: {video_path} (Both OpenCV and ffmpeg failed)")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0 or total_frames <= 0:
            cap.release()
            logger.warning(
                f"Invalid video parameters (fps={video_fps}, frames={total_frames}): {video_path}, trying ffmpeg...")
            try:
                return VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames, end_timestamp_sec, fps)
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg also failed: {ffmpeg_error}")
                raise ValueError(
                    f"Invalid video parameters: {video_path} (fps={video_fps}, frames={total_frames})")

        start_frame = 0
        if end_timestamp_sec is None:
            end_frame = total_frames - 1
        else:
            end_frame = min(int(end_timestamp_sec * video_fps), total_frames - 1)

        # Determine frame indices / 确定帧索引
        if fps is not None and fps > 0:
            # Sample by FPS / 按 FPS 采样
            duration_sec = (end_frame - start_frame) / video_fps
            target_num_frames = int(duration_sec * fps) + 1
            frame_indices = []
            for i in range(target_num_frames):
                t = i / fps
                idx = int(t * video_fps) + start_frame
                if idx <= end_frame:
                    frame_indices.append(idx)
            frame_indices = np.array(frame_indices, dtype=int)
            if len(frame_indices) == 0:
                frame_indices = np.array([start_frame], dtype=int)
        else:
            # Sample uniformly by count / 按数量均匀采样
            if end_frame < num_frames:
                frame_indices = np.linspace(
                    start_frame, end_frame, end_frame - start_frame + 1, dtype=int)
            else:
                frame_indices = np.linspace(
                    start_frame, end_frame, num_frames, dtype=int)

        frame_paths = []
        frame_timestamps_ms = []  # Store timestamp (ms) for each frame / 存储每帧的时间戳（毫秒）
        last_frame_path = None

        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        video_name = os.path.basename(video_path).split('.')[0]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                filename = f"{video_name}_frame_{idx}.jpg"
                save_path = os.path.join(temp_dir, filename)
                # Convert BGR to RGB and save with PIL to maintain correct color space
                # 将 BGR 转换为 RGB，并使用 PIL 保存以保持正确的色彩空间和曝光度
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil.save(save_path, 'JPEG', quality=95)
                frame_paths.append(save_path)
                # Calculate timestamp (ms): frame index / video_fps * 1000
                timestamp_ms = int((idx / video_fps) * 1000)
                frame_timestamps_ms.append(timestamp_ms)

        # Extract the last frame of the video for visualization / 提取视频的最后一帧用于可视化
        last_frame_idx = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
        ret, last_frame = cap.read()
        if ret:
            filename = f"{video_name}_frame_{last_frame_idx}.jpg"
            last_frame_path = os.path.join(temp_dir, filename)
            last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(last_frame_rgb)
            img_pil.save(last_frame_path, 'JPEG', quality=95)
        elif frame_paths:
            # Fallback to the last sampled frame / 如果无法读取最后一帧，使用采样帧中的最后一帧作为备选
            last_frame_path = frame_paths[-1]

        cap.release()

        if not frame_paths:
            # Try ffmpeg if OpenCV fails / 如果 OpenCV 失败，尝试使用 ffmpeg
            logger.warning(f"OpenCV failed to extract any frames: {video_path}, trying ffmpeg...")
            try:
                return VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames, end_timestamp_sec, fps)
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg also failed: {ffmpeg_error}")
                raise ValueError(f"Failed to extract any frames: {video_path}")

        logger.debug(f"Extracted {len(frame_paths)} frames, visualization frame: {last_frame_path}")
        return frame_paths, last_frame_path, frame_timestamps_ms

    @staticmethod
    def extract_frame(video_path, timestamp_sec=None):
        """
        Extract a frame at a specific timestamp. If not specified, extract the last frame.
        从视频中提取指定时间点的帧。如果未指定时间，提取最后一帧。

        Args:
            video_path (str): Path to video file / 视频文件路径。
            timestamp_sec (float, optional): Timestamp (seconds). If None, extracts the last frame.
                                             提取帧的时间戳（秒）。如果为 None，提取最后一帧。

        Returns:
            numpy.ndarray: Extracted image frame (BGR format) / 提取的图像帧 (BGR格式)。
            str: Path to saved temporary image / 保存的临时图像路径。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"OpenCV cannot open video: {video_path}, trying ffmpeg...")
            try:
                frame_paths, last_path, _ = VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames=1, end_timestamp_sec=timestamp_sec, fps=None)
                frame = cv2.imread(last_path)
                return frame, last_path
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg also failed: {ffmpeg_error}")
                raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            logger.warning(
                f"Invalid video parameters (fps={fps}, frames={total_frames}): {video_path}, trying ffmpeg...")
            try:
                frame_paths, last_path, _ = VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames=1, end_timestamp_sec=timestamp_sec, fps=None)
                frame = cv2.imread(last_path)
                return frame, last_path
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg also failed: {ffmpeg_error}")
                raise ValueError(f"Invalid video parameters: {video_path}")

        if timestamp_sec is None:
            target_frame_idx = total_frames - 1
        else:
            target_frame_idx = int(timestamp_sec * fps)

        if target_frame_idx >= total_frames:
            target_frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            logger.warning(f"OpenCV cannot read video frame: {video_path}, trying ffmpeg...")
            try:
                frame_paths, last_path, _ = VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames=1, end_timestamp_sec=timestamp_sec, fps=None)
                frame = cv2.imread(last_path)
                return frame, last_path
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg also failed: {ffmpeg_error}")
                raise ValueError("Cannot read video frame")

        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(video_path).split('.')[0] + f"_frame_{target_frame_idx}.jpg"
        save_path = os.path.join(temp_dir, filename)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil.save(save_path, 'JPEG', quality=95)

        logger.debug(f"Extracted frame: {save_path} (Index: {target_frame_idx})")
        return frame, save_path

    @staticmethod
    def visualize_points(image_path, result_json, output_path, gt_json=None, gt_items=None):
        """
        Draw predicted points on image for result visualization.
        在图像上绘制预测点，用于可视化结果。

        Args:
            image_path (str): Path to original image / 原始图像路径。
            result_json (dict): JSON result from model output / 模型输出的 JSON 结果。
            output_path (str): Path to save visualization / 保存可视化结果的路径。
            gt_json (dict, optional): Original Ground Truth data / 原始 Ground Truth 数据。
            gt_items (list, optional): Processed GT items list / 处理后的 GT items 列表。
        """
        if not os.path.exists(image_path):
            logger.warning(f"Warning: Image {image_path} not found for visualization.")
            return

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # Predicted points and GT points are unified to [x, y] format
        pred_point_list = result_json.get("point_list", [])

        gt_point_list = []
        gt_cmd = ""

        if gt_items is None or len(gt_items) == 0:
            raise ValueError("Visualization failed: Missing GT items data.")
        
        source_items = gt_items
        logger.debug(f"Using gt_items, total {len(source_items)} GT items")

        object_choices = []
        space_choices = []
        if gt_json:
            object_choices = gt_json.get("_object_choices", [])
            space_choices = gt_json.get("_space_choices", [])
        
        # Build mapping from choice to name / 构建 choice 到名称的映射
        choice_to_name = {}
        for choice_str in object_choices + space_choices:
            if ". " in choice_str:
                parts = choice_str.split(". ", 1)
                if len(parts) == 2:
                    choice_to_name[parts[0]] = parts[1]

        # Extract GT info / 提取 GT 信息
        for point in source_items:
            mask_data = point.get("mask")
            point_name = point.get("name", "")
            if not point_name:
                choice = point.get("choice", "")
                if choice and choice in choice_to_name:
                    point_name = choice_to_name[choice]
                elif choice:
                    point_name = f"Option {choice}"
            
            if point_name:
                gt_cmd = gt_cmd + point_name + " -> "

            points = point.get("points")
            if points or mask_data:
                gt_point_list.append({
                    "type": point.get("type"),
                    "description": point_name,
                    "point": points if points else [],
                    "mask": mask_data
                })

        def norm_to_pixel(pt):
            """Convert normalized [x, y] (0-1000) to pixel coordinates (x_pixel, y_pixel)"""
            if not pt: return None
            if isinstance(pt[0], list):
                result = []
                for point in pt:
                    if len(point) >= 2:
                        result.append((int(point[0] / 1000 * w), int(point[1] / 1000 * h)))
                return result if result else None
            if len(pt) < 2: return None
            return (int(pt[0] / 1000 * w), int(pt[1] / 1000 * h))

        def get_color_for_type(point_type, is_pred=True):
            """Return color based on type / 根据类型返回颜色"""
            if is_pred:
                if point_type == "target_object": return (0, 0, 255)  # Red
                elif point_type == "spatial_affordance": return (0, 255, 0)  # Green
            else:
                if point_type == "object": return (0, 255, 255)  # Yellow
                elif point_type == "space": return (255, 255, 0)  # Cyan
            return (255, 255, 255)

        # 1. Draw GT points/Mask / 1. 绘制 GT 点/Mask
        overlay = img.copy()
        for point_item in gt_point_list:
            point_type = point_item.get("type", "")
            point_coord = point_item.get("point", [])
            mask_info = point_item.get("mask")
            color = get_color_for_type(point_type, is_pred=False)

            # Draw Mask if available / 绘制 Mask（如果有）
            if mask_info and "mask_base64" in mask_info:
                try:
                    mask_data = base64.b64decode(mask_info["mask_base64"])
                    nparr = np.frombuffer(mask_data, np.uint8)
                    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        bbox = mask_info.get("bbox")
                        if bbox:
                            bx1, by1, bx2, by2 = bbox
                            bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                            bx2, by2 = min(w, int(bx2)), min(h, int(by2))
                            if mask_img.shape[0] != h or mask_img.shape[1] != w:
                                target_h, target_w = by2 - by1, bx2 - bx1
                                if target_h > 0 and target_w > 0:
                                    patch = cv2.resize(mask_img, (target_w, target_h))
                                    resized_mask = np.zeros((h, w), dtype=np.uint8)
                                    resized_mask[by1:by2, bx1:bx2] = patch
                                else: resized_mask = None
                            else: resized_mask = mask_img
                        else:
                            resized_mask = cv2.resize(mask_img, (w, h)) if (
                                mask_img.shape[0] != h or mask_img.shape[1] != w) else mask_img
                        if resized_mask is not None:
                            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, color, -1)
                except Exception as e: logger.error(f"Failed to draw mask: {e}")

            # Draw GT points / 绘制 GT 点
            if point_coord:
                coords = point_coord if isinstance(point_coord[0], list) else [point_coord]
                for pt in coords:
                    pixel_pt = (int(pt[0]), int(pt[1]))
                    cv2.circle(img, pixel_pt, 10, color, 3)
                    cv2.circle(img, pixel_pt, 3, color, -1)
                    cv2.putText(img, "GT", (pixel_pt[0]+12, pixel_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 2. Draw Predicted points / 2. 按顺序绘制预测点
        for point_item in pred_point_list:
            point_type = point_item.get("type", "")
            point_coord = point_item.get("point", [])
            if not point_coord: continue
            color = get_color_for_type(point_type, is_pred=True)
            coords = point_coord if isinstance(point_coord[0], list) else [point_coord]
            for i, pt in enumerate(coords):
                pixel_pt = (int(pt[0] / 1000 * w), int(pt[1] / 1000 * h))
                cv2.circle(img, pixel_pt, 10, color, -1)
                if i == 0:
                    cv2.putText(img, "Pred", (pixel_pt[0]+10, pixel_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. Draw command text and labels / 3. 绘制指令文本和标签
        pred_cmd = result_json.get("explicit_command", "")
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_file_dir))
            project_font_path = os.path.join(project_root, "fonts", "SourceHanSansCN-Regular.otf")
            font_paths = [project_font_path, "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "simhei.ttf"]
            font = None
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, 30)
                        break
                    except: continue
            if font is None: font = ImageFont.load_default()

            draw.text((20, 40), f"Cmd (Pred): {pred_cmd}", font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
            if gt_cmd:
                draw.text((20, 80), f"Cmd (GT): {gt_cmd}", font=font, fill=(0, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

            for point_item in gt_point_list:
                description = point_item.get("description", "")
                point_type = point_item.get("type", "")
                if not description: continue
                point_coord = point_item.get("point", [])
                pt = point_coord[0] if point_coord and isinstance(point_coord[0], list) else point_coord
                if pt and len(pt) >= 2:
                    bgr_color = get_color_for_type(point_type, is_pred=False)
                    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                    draw.text((int(pt[0]) + 15, int(pt[1]) - 25), f"GT: {description}", font=font, fill=rgb_color, stroke_width=2, stroke_fill=(0, 0, 0))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to draw text: {e}")

        cv2.imwrite(output_path, img)
        logger.info(f"Visualization saved: {output_path}")

    @staticmethod
    def _extract_frames_with_ffmpeg(video_path, num_frames=8, end_timestamp_sec=None, fps=None):
        """Use ffmpeg to extract video frames (fallback for OpenCV) / 使用 ffmpeg 提取视频帧（备选方案）"""
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0: raise ValueError(f"Failed to get video duration: {result.stderr}")
            duration = float(result.stdout.strip())
            if end_timestamp_sec is not None: duration = min(duration, end_timestamp_sec)
            safe_duration = max(0, duration - 0.2)
            if fps is not None and fps > 0:
                time_points = np.arange(0, safe_duration, 1.0/fps)
                if len(time_points) == 0: time_points = [0.0]
            elif num_frames > 1: time_points = np.linspace(0, safe_duration, num_frames)
            else: time_points = [safe_duration]

            frame_paths, frame_timestamps_ms = [], []
            temp_dir = "temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            video_name = os.path.basename(video_path).split('.')[0]

            for i, t in enumerate(time_points):
                output_file = os.path.join(temp_dir, f"{video_name}_frame_ffmpeg_{i}.jpg")
                cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-vframes", "1", "-q:v", "2", "-pix_fmt", "yuvj420p", "-strict", "-2", "-f", "image2", output_file]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if (result.returncode != 0 or not os.path.exists(output_file)) and i == len(time_points) - 1:
                    cmd = ["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path, "-vframes", "1", "-q:v", "2", "-pix_fmt", "yuvj420p", "-f", "image2", output_file]
                    subprocess.run(cmd, capture_output=True, timeout=30)
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    frame_paths.append(output_file)
                    frame_timestamps_ms.append(int(t * 1000))
            if not frame_paths: raise ValueError("ffmpeg failed to extract any frames")
            return frame_paths, frame_paths[-1], frame_timestamps_ms
        except Exception as e: raise ValueError(f"ffmpeg extraction failed: {e}")
