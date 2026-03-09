# src/prompts/ecog_prompts.py

import json

class EcoGPrompts:
    """
    管理 Ego-centric Co-Speech Intent Grounding (EcoG) 任务的 Prompt 模板。
    支持中文 (lang="zh") 和英文 (lang="en") 两种语言。
    """

    @staticmethod
    def get_system_prompt(task_template=None, coord_order="xy", options_text="", lang="zh"):
        if lang == "en":
            return EcoGPrompts._get_system_prompt_en(task_template, coord_order, options_text)
        else:
            return EcoGPrompts._get_system_prompt_zh(task_template, coord_order, options_text)

    @staticmethod
    def _get_system_prompt_zh(task_template=None, coord_order="xy", options_text=""):
        """中文系统 Prompt"""
        if coord_order == "yx":
            coord_desc = "使用 [y, x] 格式（y=纵坐标/绝对高度, x=横坐标/绝对宽度），范围 [0-1000]。(0,0) 为左上角。"
            coord_field_desc = "- `point`: [y, x] 整数坐标，0-1000。"
        else:
            coord_desc = "使用 [x, y] 格式（x=横坐标/绝对宽度, y=纵坐标/绝对高度），范围 [0-1000]。(0,0) 为左上角。"
            coord_field_desc = "- `point`: [x, y] 整数坐标，0-1000。"

        base_prompt = f"""你是一个具备具身智能（Embodied AI）能力的视觉-语言助手。你的核心任务是执行"视觉-语音意图定位"（Ego-centric Co-Speech Intent Grounding, EcoG）。

**任务背景：**
你需要根据第一人称视角（Ego-centric）的视频帧和用户的语音指令，深度理解用户的潜在意图，并将其精准转化为明确的行动指令、选项标号和视觉坐标点。

**备选选项列表：**
{options_text}

**核心约束与规则：**
1. **【严重警告】纯净 JSON 输出**: 你必须且只能输出合法的 JSON 字符串。
2. **坐标系基准**: {coord_desc} 必须基于提供的视频序列的**【最后一帧】**来提取物体或区域的坐标。
3. **概念区分**:
   - `target_object`: 用户意图抓取、移动或操作的**具体物体**。
   - `spatial_affordance`: 用户意图放置物体的**目标位置**或用于描述空间关系的**参考物/区域**。

**JSON 输出模板 (必须严格匹配此结构和键名):**
{{
    "reasoning": "简明扼要的推理过程（不超过100字）",
    "explicit_command": "在此处填写明确且完整的中文执行指令",
    "selected_options": ["A", "b"],
    "point_list": [
        {{
            "type": "target_object",
            "description": "物体的具体外观描述",
            "point": [500, 300],
            "timestamp": 1200
        }},
        {{
            "type": "spatial_affordance",
            "description": "目标位置或参考区域的描述",
            "point": [300, 500],
            "timestamp": 3200
        }}
    ]
}}

**字段详细说明：**
- `explicit_command`: 将用户模糊的意图（如"放这里"）翻译为机器人可执行的完整指令（如"把红色螺丝刀放到木桌上"）。
- `selected_options`: 包含正确选项标号的字符串列表。其顺序必须严格对应动作执行的逻辑先后顺序。
- `point_list`: 长度和顺序必须与 `selected_options` 一致。每个对象包含：
    - `type`: 仅限 "target_object" 或 "spatial_affordance"。
    - `description`: 视觉特征或位置特征的简短描述。
    - `point`: {coord_field_desc} (基于最后一帧)。
    - `timestamp`: 整数（毫秒）。指视频中用户手指动作达到稳定状态（Peak stage）、最明确指向该物体/区域那一瞬间的时间戳。
"""
        task_specific_guidance = ""
        if task_template == "指令1":
            task_specific_guidance = "\n**当前任务场景 (指令1)**：用户无语音，仅用手指向某物体。示例：selected_options: [\"A\"], point_list 包含 1 个 target_object。"
        elif task_template == "指令2":
            task_specific_guidance = "\n**当前任务场景 (指令2)**：单物体操作。示例：\"把这个拿起来\"。selected_options: [\"A\"], point_list 包含 1 个 target_object。"
        elif task_template == "指令3":
            task_specific_guidance = "\n**当前任务场景 (指令3)**：单物体放置。示例：\"把这个放到这里\"。selected_options: [\"A\", \"b\"], point_list 包含 target_object -> spatial_affordance。"
        elif task_template == "指令4":
            task_specific_guidance = "\n**当前任务场景 (指令4)**：关系型放置。示例：\"把这个放到它的右边\"。selected_options: [\"A\", \"b\"], point_list 包含 target_object -> spatial_affordance。"
        elif task_template == "指令5":
            task_specific_guidance = "\n**当前任务场景 (指令5)**：多步串联（放+拿）。示例：\"把这个放到它的右边，然后把它拿起来\"。selected_options: [\"A\", \"b\", \"C\"], point_list 包含 target_object -> spatial_affordance -> target_object。"
        elif task_template == "指令6":
            task_specific_guidance = "\n**当前任务场景 (指令6)**：多步串联（放+放）。示例：\"把这个放到它的前面，然后再把这个放到它的后面\"。selected_options: [\"A\", \"b\", \"C\", \"d\"], point_list 包含 target_object -> spatial_affordance -> target_object -> spatial_affordance。"

        return base_prompt + task_specific_guidance + "\n\n请直接输出 JSON，不要有任何前缀或后缀。"

    @staticmethod
    def _get_system_prompt_en(task_template=None, coord_order="xy", options_text=""):
        """English system prompt"""
        if coord_order == "yx":
            coord_desc = "Use [y, x] format (y=vertical/absolute height, x=horizontal/absolute width), range [0-1000]. (0,0) is the top-left."
            coord_field_desc = "- `point`: [y, x] integer coordinates, 0-1000."
        else:
            coord_desc = "Use [x, y] format (x=horizontal/absolute width, y=vertical/absolute height), range [0-1000]. (0,0) is the top-left."
            coord_field_desc = "- `point`: [x, y] integer coordinates, 0-1000."

        base_prompt = f"""You are a visual-language assistant with Embodied AI capabilities. Your core task is Ego-centric Co-Speech Intent Grounding (EcoG).

**Task Background:**
You must deeply understand the user's underlying intent based on ego-centric video frames and the user's voice instructions, translating them into explicit actionable commands, option selections, and visual coordinate points.

**Option List:**
{options_text}

**Core Constraints & Rules:**
1. **[CRITICAL WARNING] Pure JSON Output**: You MUST output ONLY a valid JSON string.
2. **Coordinate Reference**: {coord_desc} Coordinates MUST be extracted based on the **LAST FRAME** of the provided video sequence.
3. **Concept Distinction**:
   - `target_object`: The specific object the user intends to grasp, move, or manipulate.
   - `spatial_affordance`: The destination area or the reference object/region used to define a spatial relationship for placement.

**JSON Output Template (Strictly adhere to this structure and keys):**
{{
    "reasoning": "Brief reasoning process (under 50 words)",
    "explicit_command": "Fill in the complete, explicit, and actionable English command here",
    "selected_options": ["A", "b"],
    "point_list": [
        {{
            "type": "target_object",
            "description": "Specific visual description of the object",
            "point": [500, 300],
            "timestamp": 1200
        }},
        {{
            "type": "spatial_affordance",
            "description": "Visual description of the target location or reference area",
            "point": [300, 500],
            "timestamp": 3200
        }}
    ]
}}

**Field Descriptions:**
- `explicit_command`: Translate vague intents (e.g., "Put that there") into complete robot-executable instructions (e.g., "Put the red screwdriver on the wooden table").
- `selected_options`: A list of strings containing the correct option labels. The order MUST match the logical chronological sequence of the actions.
- `point_list`: Length and order must exactly match `selected_options`. Each object includes:
    - `type`: Must be strictly "target_object" or "spatial_affordance".
    - `description`: Brief visual or spatial feature description.
    - `point`: {coord_field_desc} (Based on the LAST frame).
    - `timestamp`: Integer (milliseconds). The exact moment (peak stage) when the user's pointing gesture holds steady and clearly identifies the target.
"""
        task_specific_guidance = ""
        if task_template == "指令1":
            task_specific_guidance = "\n**Current Scenario (Instruction 1)**: User points ONLY, NO speech. Example: selected_options: [\"A\"], point_list contains 1 target_object."
        elif task_template == "指令2":
            task_specific_guidance = "\n**Current Scenario (Instruction 2)**: Single object manipulation. Example: \"Pick this up\". selected_options: [\"A\"], point_list contains 1 target_object."
        elif task_template == "指令3":
            task_specific_guidance = "\n**Current Scenario (Instruction 3)**: Single object placement. Example: \"Put this here\". selected_options: [\"A\", \"b\"], point_list contains target_object -> spatial_affordance."
        elif task_template == "指令4":
            task_specific_guidance = "\n**Current Scenario (Instruction 4)**: Relational placement. Example: \"Place this to the right of it\". selected_options: [\"A\", \"b\"], point_list contains target_object -> spatial_affordance."
        elif task_template == "指令5":
            task_specific_guidance = "\n**Current Scenario (Instruction 5)**: Multi-step (Place + Pick). Example: \"Place this to its right, then pick that up\". selected_options: [\"A\", \"b\", \"C\"], point_list contains target_object -> spatial_affordance -> target_object."
        elif task_template == "指令6":
            task_specific_guidance = "\n**Current Scenario (Instruction 6)**: Multi-step (Place + Place). Example: \"Place this in front of it, and place that behind it\". selected_options: [\"A\", \"b\", \"C\", \"d\"], point_list contains target_object -> spatial_affordance -> target_object -> spatial_affordance."

        return base_prompt + task_specific_guidance + "\n\nOUTPUT ONLY RAW JSON. NO EXPLANATIONS."

    @staticmethod
    def get_user_prompt(user_transcript, asr_result=None, lang="zh", use_asr_result=False,
                        strip_word_timestamps=False):
        """构建用户输入的 Prompt。采用 XML 标签结构化输入，降低理解难度。

        Args:
            user_transcript: 用户语音转录文本
            asr_result: ASR 结果字典（包含 text 和 words 等字段）
            lang: 语言 "zh" 或 "en"
            use_asr_result: 是否在 prompt 中拼接 ASR 结果
            strip_word_timestamps: 为 True 时，仅保留 ASR 文本内容，剥除词级时间戳字段
                                   （用于 temporal anchor 消融实验中的 no_word_asr_timing 条件）
        """
        asr_info = ""
        if use_asr_result and asr_result:
            # 根据消融条件决定使用哪种 ASR 数据
            if strip_word_timestamps:
                # 仅保留顶层文本，去掉词级 begin_time/end_time 信息
                asr_for_prompt = {"text": asr_result.get("text", user_transcript)}
            else:
                asr_for_prompt = asr_result

            if lang == "en":
                if strip_word_timestamps:
                    asr_note = "Note: ASR result contains the transcript text only (word-level timestamps have been removed)."
                else:
                    asr_note = "Note: ASR results contain word-level timestamps (begin_time, end_time) of the words in the user's transcript."
                asr_info = f"""
<asr_data>
{json.dumps(asr_for_prompt, ensure_ascii=False)}
{asr_note}
</asr_data>"""
                return f"<instruction>\n{user_transcript}\n</instruction>{asr_info}\n\nPlease analyze the provided video frames alongside the inputs above. Generate your response strictly in the requested JSON format."
            else:
                if strip_word_timestamps:
                    asr_note = "注意：ASR结果仅包含转录文本（词级别时间戳已移除）。"
                else:
                    asr_note = "注意：ASR结果包含视频中用户语音转译的词级别的时间戳(begin_time, end_time)。"
                asr_info = f"""
<asr_data>
{json.dumps(asr_for_prompt, ensure_ascii=False)}
{asr_note}
</asr_data>"""
                return f"<instruction>\n{user_transcript}\n</instruction>{asr_info}\n\n请综合分析提供的视频帧与上述输入内容，严格按照要求的 JSON 格式输出最终结果。"

        else:
            if lang == "en":
                return f"\nPlease analyze the provided video frames alongside the inputs above. Generate your response strictly in the requested JSON format."
            else:
                return f"\n请综合分析提供的视频帧与上述输入内容，严格按照要求的 JSON 格式输出最终结果。"

    @staticmethod
    def get_asr_matching_prompt(asr_result, object_space, lang="zh"):
        if lang == "en":
            return EcoGPrompts._get_asr_matching_prompt_en(asr_result, object_space)
        else:
            return EcoGPrompts._get_asr_matching_prompt_zh(asr_result, object_space)

    @staticmethod
    def _get_asr_matching_prompt_zh(asr_result, object_space):
        prompt = f"""你是一个专业的 ASR 结果对齐助手。你的任务是将 ASR 识别出的词语与视频中标注的“物体/空间区域”进行时间戳对齐。

**输入数据：**
<asr_result>
{json.dumps(asr_result, ensure_ascii=False, indent=2)}
</asr_result>

<object_space>
{json.dumps(object_space, ensure_ascii=False, indent=2)}
</object_space>

**对齐规则：**
1. **顺序一致性**：`object_space` 中的元素顺序通常与 `asr_result["words"]` 中的提及顺序一致。
2. **多词合并 (Sub-word Merging)**：如果 `object_space` 中的一个完整概念在 ASR 中被拆分成了多个连续的词（例如：object为"后面"，ASR分为"后"和"面"；或 object为"螺丝刀"，ASR分为"螺丝"和"刀"），你必须**合并**这些词的时间戳：取第一个词的 `begin_time` 和最后一个词的 `end_time`。
3. **错误标记 (`asr_match_error`)**：
   - 如果发生语义冲突（如 ASR 说"左边"，object 标注"右边"），设为 `true`。
   - 如果 ASR 中完全缺失该物体/位置的提及，设为 `true`。
   - 匹配成功且合理时，设为 `false`。

**输出格式要求：**
- JSON
- 数组的长度必须与 `<object_space>` 的长度**完全一致**。
- 格式示例：
[
  {{
    "index": 0,
    "asr_begin_time": 3370,
    "asr_end_time": 4530,
    "asr_match_error": false
  }}
]
"""
        return prompt

    @staticmethod
    def _get_asr_matching_prompt_en(asr_result, object_space):
        prompt = f"""You are an expert ASR alignment assistant. Your task is to align words from ASR recognition results with annotated "objects/spatial regions" from a video.

**Input Data:**
<asr_result>
{json.dumps(asr_result, ensure_ascii=False, indent=2)}
</asr_result>

<object_space>
{json.dumps(object_space, ensure_ascii=False, indent=2)}
</object_space>

**Alignment Rules:**
1. **Sequential Consistency**: The chronological order of elements in `object_space` generally matches their order of appearance in `asr_result["words"]`.
2. **Sub-word Merging**: If a single concept in `object_space` is split across multiple consecutive tokens in ASR (e.g., object is "screwdriver", ASR splits into "screw" and "driver"; or object is "behind", ASR splits into "be" and "hind"), you MUST **merge** their timestamps: use the `begin_time` of the first token and the `end_time` of the last token.
3. **Error Flagging (`asr_match_error`)**:
   - Set to `true` if there is a semantic conflict (e.g., ASR says "left", object says "right").
   - Set to `true` if the object/location is entirely missing from the ASR words.
   - Set to `false` if the match is successful and logical.

**Output Requirements:**
- JSON
- The length of the output array MUST be **exactly equal** to the length of `<object_space>`.
- Example format:
[
  {{
    "index": 0,
    "asr_begin_time": 3370,
    "asr_end_time": 4530,
    "asr_match_error": false
  }}
]
"""
        return prompt