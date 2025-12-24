# poem_vision_extractor.py
# 从“五朝御制诗二.json”中抽取视觉意象与眺望对象（同时利用 title 与 poem）
#如果你希望把你给的那些“意象短语”进一步做成几条真正的 few‑shot 样本
# （比如先给一首短诗的“输入 → 输出 JSON”示例），我也可以帮你把 prompt 再升级一版，更适合做细致标注。

import os
import json
import argparse

from utils import setup_logging       # 复用你现有的工具函数
from LLM_Client import LLM_Client     # 复用你现有的 LLM 客户端


def parse_args():
    """
    命令行参数：
    --input      输入 JSON 文件路径
    --output     输出 JSON 文件路径
    --max_poems  只跑前 N 首诗（调试时可以用）
    --temperature  LLM 温度
    """
    parser = argparse.ArgumentParser(
        description="从五朝御制诗中抽取视觉意象和眺望对象（读 title + poem）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="E:\Qianlong\LLM_poem_viewer\五朝御制诗二.json",
        help="输入 JSON 文件路径，默认：E:\Qianlong\LLM_poem_viewer\五朝御制诗二.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="五朝御制诗二_视觉意象抽取结果.json",
        help="输出结果 JSON 文件路径",
    )
    parser.add_argument(
        "--max_poems",
        type=int,
        default=None,
        help="只处理前 N 条数据（调试用，默认全部处理）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM 采样温度（默认 0.2）", #检索类任务
    )
    return parser.parse_args()


def safe_json_load(maybe_json: str):
    """
    尝试从 LLM 返回的字符串中解析 JSON。
    兼容以下几种情况：
    - 直接是 JSON
    - 包在 ```json ... ``` 代码块里
    - 包在 ``` ... ``` 代码块里
    """
    maybe_json = maybe_json.strip()

    # 1. 直接解析
    try:
        return json.loads(maybe_json)
    except Exception:
        pass

    # 2. 处理 ```json ... ``` 或 ``` ... ``` 包裹
    if "```" in maybe_json:
        first = maybe_json.find("```")
        last = maybe_json.rfind("```")
        inner = maybe_json[first + 3:last].strip()  # 去掉左右 ``` 后的内容

        # 去掉可能的 'json' 标记
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()

        return json.loads(inner)

    # 如果还是失败，就抛异常，让上层处理
    raise ValueError("无法从模型输出中解析合法 JSON：\n" + maybe_json)


def build_prompt(record: dict) -> str:
    """
    根据一条诗歌记录构造 prompt。

    record 结构示例：
    {
      "point": "天然图画",
      "emperor": "雍正朝",
      "year": "",
      "title": "秋日登朗吟阁寓目（350）",
      "poem": "..."
    }
    """
    point = record.get("point", "")
    emperor = record.get("emperor", "")
    year = record.get("year", "")
    title = record.get("title", "")
    poem = record.get("poem", "")

    # 类型体系和任务说明（已按你的新要求改：读 title + poem，增加 place，删除 verb_type）
    few_shot_desc = """
【任务说明】
你是一个精通中国古文献学、训诂学以及中国古典园林史的专家，
同时也是一位熟悉圆明园等皇家园林景观结构的研究者。

现在需要你对【题目 + 诗文】整体进行分析，完成两个子任务：
1. 从题目中提取圆明园内的具体地点名称，作为 place 字段。
   - 例如题目为“春雨轩小坐因而成咏”，则应抽取“春雨轩”作为 place。
   - 如果题目中出现多个地名，只保留本首诗重点描写的核心景点名称（通常是带“轩、阁、堂、馆、楼、舫、庄、园”等字的专名）。
   - 如果确实无法确定具体地点，则 place 置为 ""（空字符串）。

2. 在【题目 + 诗文】中，一并抽取具有“视线方向”的视觉动词及其眺望对象：
   - 包括“看、望、瞻、眺、极目、俯视、俯瞰、远挹、远眺、凭栏极目、极望、俯凭”等，
     以及其他明显表示“向外看/向某处看”的动词或短语。
   - 不需要区分该动词出现在题目还是诗文，统一作为 visual_events 列出即可。

【对象类别体系】
1. object_category（大类：视线在园内还是园外）
   - "境外眺望对象"：圆明园之外的景观意象（如远山、田畴、城郭、云天等）
   - "境内眺望对象"：圆明园内部的景观意象（如园内山石、湖池、楼阁、树木、禽鸟等）

2. object_subcategory（小类，需与大类匹配）
   - 若 object_category = "境外眺望对象"：
        * "山林"：远岑、远峤、西山、翠微、青螺、雪峰、岭岫等
        * "田村"：稻塍千顷、畎亩、平田、连阡麦穗香等
        * "云天"：云、霞、月、天光、碧霄空、黄云、轻霞等
   - 若 object_category = "境内眺望对象"：
        * "山石"：奇石、怪石、嶙峋、假山、湖石等
        * "湖溪"：池、渚、溪、泉、潺潺水流、碧流、潀泉等
        * "田圃"：塍、田、畦、稻町、菜圃等
        * "建筑"：轩、阁、堂、楼、桥、廊、书屋、亭等
        * "植物"：松、竹、荷花、古树、柳、梅、桃花、草、芳茵等
        * "动物"：鸟、鱼、蝶、禽、鸦等

【输出要求】
1. 只根据本首诗的题目与诗文进行分析，不要编造原文中没有出现的意象。
2. 需要完成：
   (1) place：从题目中提取圆明园具体景点名；若不确定则为 ""。
   (2) visual_events：列出所有具有“视线方向”的视觉动作，以及每个动作所看到/指向的对象。
3. 对于每一个眺望/观看事件，请给出：
   - verb: 动词原词（如 "望"、"极目"）
   - verb_span: 诗题或诗句中包含该动词的完整短语（原文片段，用于回溯）
   - object_phrase: 被看到/眺望的对象在题目或诗文中的原文短语
   - object_category: "境外眺望对象" 或 "境内眺望对象"
   - object_subcategory: 在上述小类中选择最合适的一类；若实在无法判断，可用 "其他"
   - comment: 简短文字说明“视线从哪里看到什么”，便于后续人工校对

4. 如果该首诗中完全没有这类视觉动作，请返回 "visual_events": []，但仍需给出 place 字段。

【输出 JSON 模板】
请严格按照下列 JSON 模板输出，保证是合法 JSON，不要在 JSON 外输出任何解释：

{
  "point": "...",
  "place": "春雨轩",
  "emperor": "...",
  "year": "...",
  "title": "...",
  "poem": "...",
  "visual_events": [
    {
      "verb": "望",
      "verb_span": "凭栏极目",
      "object_phrase": "西山浓翠屏风展",
      "object_category": "境外眺望对象",
      "object_subcategory": "山林",
      "comment": "在园内凭栏远望园外西山"
    }
  ]
}
    """.strip()

    prompt = f"""
你是一位中国古文献学和中国古典园林专家，请对下面这首御制诗进行“视觉意象与眺望对象”抽取，同时从题目中提取圆明园具体地点。

【基本信息】
题咏点（point）：{point}
朝代/皇帝（emperor）：{emperor}
年份（year，如有）：{year}
题目（title）：{title}

【诗文原文（poem）】
{poem}

{few_shot_desc}
"""
    return prompt


def extract_visual_imagery_for_record(client: LLM_Client, record: dict, logger):
    """
    对单条诗歌记录调用 LLM，返回带 visual_events 的结构。
    同时包含 place 字段，并确保输出字段顺序：
    point -> place -> emperor -> year -> title -> poem -> visual_events
    """
    prompt = build_prompt(record)
    messages = [{"role": "user", "content": prompt}]

    logger.info(f"正在处理：{record.get('title', '')} | {record.get('point', '')}")

    # 调用 LLM
    res_content = client.get_chat_completion(messages=messages)

    # 解析 JSON（兼容 ```json ... ``` 等形式）
    try:
        parsed = safe_json_load(res_content)
    except Exception as e:
        logger.error(f"解析 LLM 输出失败，将该首诗标记为错误：{e}")
        return {
            "point": record.get("point", ""),
            "place": "",
            "emperor": record.get("emperor", ""),
            "year": record.get("year", ""),
            "title": record.get("title", ""),
            "poem": record.get("poem", ""),
            "visual_events": [],
            "error": f"parse_failed: {str(e)}",
            "raw_response": res_content,
        }

    # 补充 / 纠正关键字段（避免模型漏填）
    point = parsed.get("point", record.get("point", ""))
    place = parsed.get("place", "")
    emperor = parsed.get("emperor", record.get("emperor", ""))
    year = parsed.get("year", record.get("year", ""))
    title = parsed.get("title", record.get("title", ""))
    poem = parsed.get("poem", record.get("poem", ""))
    visual_events = parsed.get("visual_events", [])

    # 为了保证 key 顺序，重新构建一个 dict
    normalized = {
        "point": point,
        "place": place,
        "emperor": emperor,
        "year": year,
        "title": title,
        "poem": poem,
        "visual_events": visual_events,
    }

    # 如果 LLM 另外加了一些字段，也可以按需挂到 normalized 上（可选）
    # for k, v in parsed.items():
    #     if k not in normalized:
    #         normalized[k] = v

    return normalized


def main():
    args = parse_args()

    # 日志文件名用输出名衍生，便于对应
    log_path = os.path.splitext(args.output)[0] + ".log"
    logger = setup_logging(log_path)

    # 初始化 LLM 客户端
    client = LLM_Client(
        logging=logger,
        temperature=args.temperature,
        max_tokens=4096,
        model_name="DeepSeek-V3.1",   # 对应 LLM_Client 中的 MODEL_CONFIG key
    )

    # 读取输入 JSON
    logger.info(f"读取输入文件：{args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 期望是一个 list，每个元素是一首诗的字典。")

    if args.max_poems is not None:
        data = data[: args.max_poems]
        logger.info(f"仅处理前 {args.max_poems} 首诗（调试模式）。")

    results = []
    total = len(data)

    for idx, record in enumerate(data):
        logger.info(f"===== 处理第 {idx + 1}/{total} 首 =====")
        result = extract_visual_imagery_for_record(client, record, logger)
        results.append(result)

    # 写出结果
    logger.info(f"写出结果到：{args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印 / 记录成本信息（沿用 LLM_Client 的计费统计）
    try:
        cost_info = client.calculate_cost()
        logger.info(f"总 Token 统计与费用：{cost_info}")
    except Exception as e:
        logger.error(f"cost 计算失败（可忽略）：{e}")


if __name__ == "__main__":
    main()
