#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_ai_rename.py  ·  并发流水线版
=====================================
并发三阶段流水线架构：

   ┌──────────────────────────────────────────────────────┐
   │  扫描文件  →  同时提交到两条独立起跑线                │
   │                                                       │
   │  [GPU Worker x1]   串行  →  Whisper 转写              │
   │  [Frame Worker xN] 并行  →  FFmpeg 截图               │
   │                                                       │
   │  某文件的两路结果同时就绪 → [AI Worker xM] 并发请求   │
   │  AI 完成 → [Finalize Worker] 写元数据 + 重命名         │
   └──────────────────────────────────────────────────────┘

工具依赖 (置于 ffmpeg/ 目录):
  ffmpeg.exe / ffprobe.exe / exiftool.exe

Python 依赖:
  pip install faster-whisper google-genai tqdm
"""

import os
import sys

# ─────────────────────────────────────────────
# 修复 Windows 环境下 HuggingFace 缺少开发者权限导致的创建软链接报错 (WinError 1314)
# ─────────────────────────────────────────────
if os.name == 'nt':
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import re
import json
import shutil
import logging
import argparse
import subprocess
import tempfile
import datetime
import ctypes
import threading
import queue
from ctypes import wintypes
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# ─────────────────────────────────────────────
# 修复 Windows 下 Faster-Whisper (CTranslate2) 找不到 CUDA DLL 的问题
# ─────────────────────────────────────────────
if os.name == 'nt':
    try:
        import site
        all_site_pkgs = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site not in all_site_pkgs:
            all_site_pkgs.append(user_site)
        for site_pkg in all_site_pkgs:
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.exists(nvidia_dir):
                for folder in os.listdir(nvidia_dir):
                    bin_path = os.path.join(nvidia_dir, folder, "bin")
                    if os.path.exists(bin_path):
                        os.add_dll_directory(bin_path)
                        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# ─────────────────────────────────────────────
#  可在此处修改的全局配置
# ─────────────────────────────────────────────
CONFIG = {
    # ── 公共配置 ───────────────────────────────────────────────────────────────────
    # AI 供应商选择: "gemini" 或 "openai" (优先读取 .env)
    "ai_provider": os.environ.get("AI_PROVIDER", "openai"),

    # ── Gemini 配置 ───────────────────────────────────────────────────────────────
    # API Key (推荐通过 .env 文件或环境变量 GEMINI_API_KEY 传入)
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    # Gemini 模型名称
    "gemini_model": "gemini-3-flash-preview",

    # ── OpenAI 层 (支持任意第三方 OpenAI 居容接口) ───────────────────────────
    # API Key (推荐通过 .env 文件或环境变量 OPENAI_API_KEY 传入)
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    # OpenAI 兼容的请求地址 (例: https://api.openai.com/v1  或第三方接口)
    "openai_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    # 请求的模型名 (如 gpt-4o / qwen-vl-max / glm-4v 等)
    "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o"),

    # ── 转写配置 ───────────────────────────────────────────────────────────────────
    # Faster-Whisper 模型大小
    "whisper_model": "large-v3-turbo",
    # Faster-Whisper 计算精度
    "whisper_compute_type": "int8_float16",

    # ── 关键帧配置 ──────────────────────────────────────────────────────────────────
    # 最多提取关键帧数量
    "max_keyframes": 3,
    # 关键帧缩放宽度 (px)
    "keyframe_width": 720,

    # ── 其他配置 ──────────────────────────────────────────────────────────────────
    # 静音判断阈值 (dBFS), 低于此值认为静音
    "silence_threshold_db": -60.0,
    # 视频文件扩展名白名单
    "video_extensions": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv",
                         ".flv", ".webm", ".ts", ".mts", ".m2ts"},
    # ffmpeg / ffprobe / exiftool 所在目录 (相对于本脚本)
    "tools_dir": "ffmpeg",
    # 并发截图线程数
    "keyframe_workers": 2,
    # AI/收尾并发线程数
    "ai_workers": 2,
    # 是否将处理失败的文件移入 _failed 子目录
    "move_failed": True,
    # 干跑模式
    "dry_run": False,
}

# ─────────────────────────────────────────────
#  日志配置 (线程安全 + 强制 UTF-8 输出防止 Windows GBK 乱码)
# ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VideoAIRename")

# 用于统计结果的全局计数器（线程安全）
_stats_lock = threading.Lock()
_success = 0
_fail    = 0
_skipped = 0


# ══════════════════════════════════════════════════════════════════════════════
#  工具路径解析
# ══════════════════════════════════════════════════════════════════════════════
def _tool(name: str) -> str:
    script_dir = Path(__file__).parent
    local = script_dir / CONFIG["tools_dir"] / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"找不到工具: {name}。请将其放入 {CONFIG['tools_dir']}/ 目录。")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: 媒体信息检测 (Probe)
# ══════════════════════════════════════════════════════════════════════════════
def probe_video(video_path: str) -> dict:
    cmd = [
        _tool("ffprobe.exe"),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr[:400]}")

    data    = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt     = data.get("format", {})

    has_audio    = any(s.get("codec_type") == "audio" for s in streams)
    duration     = float(fmt.get("duration", 0))
    
    tags = fmt.get("tags", {})
    creation_time = tags.get("creation_time", "")
    
    # 检测特有的防重复标记 (ExifTool -Software)
    # 因为 ffprobe 对 MP4 文件默认不解析 Software 字段，我们必须用 exiftool 来读取它
    software_tag = ""
    try:
        cmd_exif = [_tool("exiftool.exe"), "-Software", "-S", "-s", video_path]
        r_exif = subprocess.run(cmd_exif, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if r_exif.returncode == 0:
            software_tag = r_exif.stdout.strip()
    except Exception:
        pass

    is_processed = ("AIVideoRenameV1" in software_tag)

    mean_volume = None
    if has_audio:
        mean_volume = _detect_volume(video_path)

    return {
        "has_audio":    has_audio,
        "mean_volume":  mean_volume,
        "duration":     duration,
        "creation_time": creation_time,
        "is_processed":  is_processed,
    }


def _detect_volume(video_path: str) -> float:
    cmd = [
        _tool("ffmpeg.exe"),
        "-i", video_path,
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="ignore")
    match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", result.stderr)
    if match:
        return float(match.group(1))
    return CONFIG["silence_threshold_db"] - 1


def is_silent(mean_volume: Optional[float]) -> bool:
    if mean_volume is None:
        return True
    return mean_volume < CONFIG["silence_threshold_db"]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2a: 音频转写 (Faster-Whisper) — GPU 串行，全局复用模型实例
# ══════════════════════════════════════════════════════════════════════════════
_WHISPER_MODEL_CACHE = None


def transcribe_audio(video_path: str, tmp_dir: str) -> str:
    global _WHISPER_MODEL_CACHE
    audio_path = os.path.join(tmp_dir, "audio.wav")

    cmd_extract = [
        _tool("ffmpeg.exe"),
        "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    r = subprocess.run(cmd_extract, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"音频提取失败: {r.stderr[-400:]}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper 未安装，跳过语音转写。")
        return ""

    if _WHISPER_MODEL_CACHE is None:
        log.info("  [Whisper] 加载模型 %s (%s)...",
                 CONFIG["whisper_model"], CONFIG["whisper_compute_type"])
        try:
            _WHISPER_MODEL_CACHE = WhisperModel(
                CONFIG["whisper_model"],
                device="auto",
                compute_type=CONFIG["whisper_compute_type"],
            )
        except Exception as e:
            if any(k in str(e).lower() for k in ("cuda","cublas","cudnn","loaded")):
                log.warning("  [Whisper] GPU 不可用，回退 CPU 模式: %s",
                            str(e).split('\n')[0])
                _WHISPER_MODEL_CACHE = WhisperModel(
                    CONFIG["whisper_model"],
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise
    else:
        log.info("  [Whisper] 复用已加载的模型")

    segments, info = _WHISPER_MODEL_CACHE.transcribe(audio_path, beam_size=5)
    log.info("  [Whisper] 语言=%s (%.2f), 时长=%.1fs",
             info.language, info.language_probability, info.duration)

    from tqdm import tqdm
    text_segs = []
    with tqdm(total=round(info.duration, 2), unit="s",
              desc="  [转写进度]",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]") as pbar:
        for seg in segments:
            text_segs.append(seg.text.strip())
            pbar.update(seg.end - pbar.n)
        pbar.update(info.duration - pbar.n)

    return " ".join(text_segs).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2b: 关键帧截图 — 纯 CPU/IO，可并行
# ══════════════════════════════════════════════════════════════════════════════
def extract_keyframes(video_path: str, tmp_dir: str, duration: float = 0.0) -> list[str]:
    log.info("  [Keyframe] 等间距极速截图...")
    frames = _extract_uniform_frames(video_path, tmp_dir,
                                     CONFIG["max_keyframes"], duration)
    frames = frames[:CONFIG["max_keyframes"]]
    log.info("  [Keyframe] 共提取 %d 张", len(frames))
    return [str(f) for f in frames]


def _extract_uniform_frames(video_path: str, tmp_dir: str,
                              count: int, duration: float) -> list[Path]:
    for old in Path(tmp_dir).glob("frame_*.jpg"):
        old.unlink(missing_ok=True)

    if duration <= 0:
        duration = 30.0

    interval = max(duration / (count + 1), 1)
    found = []
    for i in range(1, count + 1):
        ts      = interval * i
        out     = os.path.join(tmp_dir, f"frame_uniform_{i:03d}.jpg")
        cmd = [
            _tool("ffmpeg.exe"),
            "-y",
            "-ss", f"{ts:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            "-vf", f"scale={CONFIG['keyframe_width']}:-1",
            out,
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out):
            found.append(Path(out))
    return found


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: AI 多模态推理 (Gemini / OpenAI 兼容双后端)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你现在是一个专业的视频内容分析专家。
我会为你提供：
1. 视频音频的转写文本（若有）
2. 若干关键帧截图

请结合两者完成以下两项任务：
1. 提取出一个极其简练精锐的短词汇组合作为视频【标题】。绝对不要在标题中输出类似于日期、时间的数字前缀！
2. 撰写一段详细生动的【内容描述】，概括视频中的场景、核心主体、人物动作以及故事脉络。"""


# Pydantic 结构化输出模型（两个后端共用）
from pydantic import BaseModel, Field

class VideoMetadata(BaseModel):
    title: str = Field(description="12字以内的简洁标题。请只返回概括性文字，【绝对不要】包含任何形式的日期时间或数字前缀，不要标点符号！")
    description: str = Field(description="100字以内的详细中文描述，概括视频主要内容")


def _build_text_prompt(transcript: str, creation_time: str) -> str:
    text = ""
    if transcript:
        text += f"【音频转写文本】\n{transcript}\n\n"
    else:
        text += "【音频转写文本】\n（无音频或无法识别的语音内容）\n\n"
    if creation_time:
        text += f"【原始录制时间参考】\n{creation_time}\n\n"
    text += "【请结合以上文本并参考附带的关键帧图片进行判定】"
    return text


def _query_gemini_backend(transcript: str, frame_paths: list[str], creation_time: str) -> dict:
    """Gemini 多模态后端，使用 Pydantic response_schema 强制结构化输出。"""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("请安装: pip install google-genai")

    api_key = CONFIG["gemini_api_key"] or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    contents = [_build_text_prompt(transcript, creation_time)]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            img_bytes = f.read()
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=CONFIG["gemini_model"],
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=VideoMetadata,
        )
    )
    finish = (response.candidates[0].finish_reason.name
              if response.candidates and response.candidates[0].finish_reason
              else "UNKNOWN")
    raw = response.text.strip() if response.text else "{}"
    log.info("  [Gemini] 结束原因: %s | 原始响应: %s", finish, raw[:200])
    try:
        return json.loads(raw)
    except Exception:
        raise ValueError(f"JSON 解析失败: {raw[:300]}")


def _query_openai_backend(transcript: str, frame_paths: list[str], creation_time: str) -> dict:
    """
    OpenAI 兼容层后端，支持任意第三方接口（含 Gemini OpenAI 兼容层）。
    使用 json_object 模式 + system prompt 约束，兼容性最广。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装: pip install openai")
    import base64

    api_key = CONFIG["openai_api_key"] or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=CONFIG["openai_base_url"])

    # 将 JSON 结构约束嵌入 system prompt，避免使用 json_schema（Gemini 兼容层不支持）
    system_with_schema = SYSTEM_PROMPT + """

请严格以如下 JSON 格式输出，不要输出任何其他内容：
{
  "title": "12字以内的简洁中文标题，不含日期时间数字前缀，不含标点",
  "description": "100字以内的详细中文描述，概括视频主要内容"
}"""

    user_content: list = [{"type": "text", "text": _build_text_prompt(transcript, creation_time)}]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    response = client.chat.completions.create(
        model=CONFIG["openai_model"],
        messages=[
            {"role": "system", "content": system_with_schema},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},   # 兼容所有 OpenAI 兼容层
    )
    choice = response.choices[0]
    finish = choice.finish_reason or "UNKNOWN"
    raw = (choice.message.content or "{}").strip()
    log.info("  [OpenAI] 结束原因: %s | 原始响应: %s", finish, raw[:200])
    try:
        data = json.loads(raw)
        # 健壮性校验：确保必须字段存在
        if "title" not in data or "description" not in data:
            raise ValueError(f"返回 JSON 缺少必要字段: {list(data.keys())}")
        return data
    except json.JSONDecodeError:
        raise ValueError(f"JSON 解析失败: {raw[:300]}")



def query_ai(transcript: str, frame_paths: list[str], creation_time: str) -> dict:
    """
    公共 AI 调度函数，统一管理重试/退避逻辑。
    根据 CONFIG['ai_provider'] 自动路由到 Gemini 或 OpenAI 兼容后端。
    """
    import time
    provider = CONFIG.get("ai_provider", "gemini").lower()
    backend  = _query_gemini_backend if provider == "gemini" else _query_openai_backend
    label    = "Gemini" if provider == "gemini" else f"OpenAI({CONFIG['openai_model']})"

    max_retries = 3
    base_delay  = 5
    for attempt in range(max_retries + 1):
        try:
            return backend(transcript, frame_paths, creation_time)
        except Exception as e:
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
                log.warning(
                    "  [⚠ %s] 请求失败，第 %d 次重试 (等待 %ds)... 错误: %s",
                    label, attempt + 1, sleep_time, str(e).split('\n')[0]
                )
                time.sleep(sleep_time)
            else:
                raise e



# ══════════════════════════════════════════════════════════════════════════════
#  Step 4a: ExifTool 写入元数据 (argfile 模式，根治 GBK 乱码)
# ══════════════════════════════════════════════════════════════════════════════
def write_metadata(video_path: str, title: str, description: str,
                   creation_time: str) -> str:
    title       = title.replace('\n', ' ').strip()
    description = description.replace('\n', ' ').strip()

    args = ["-m", "-q", "-overwrite_original", "-charset", "utf8"]
    if title:
        args += [f"-Title={title}", f"-ItemList:Title={title}"]
    if description:
        args += [f"-Description={description}", f"-Comment={description}",
                 f"-ItemList:Comment={description}"]
    if creation_time:
        args.append(f"-CreateDate={creation_time}")
        
    # 打入特殊防重复扫描的标记水印
    args += ["-Software=AIVideoRenameV1", "-ItemList:Software=AIVideoRenameV1"]
    
    args.append(video_path)

    arg_file = video_path + ".exifargs"
    try:
        with open(arg_file, "w", encoding="utf-8") as f:
            for a in args:
                f.write(a + "\n")
        cmd = [_tool("exiftool.exe"), "-@", arg_file]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore")
        if r.returncode != 0:
            log.warning("[\u26A0ExifTool] 写入元数据异常: %s", r.stderr[-300:])
    finally:
        if os.path.exists(arg_file):
            os.remove(arg_file)
    return video_path


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4b: 文件重命名
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def rename_video(video_path: str, new_stem: str) -> str:
    p = Path(video_path)
    clean = sanitize_filename(new_stem)
    new_path = p.parent / (clean + p.suffix.lower())
    counter = 1
    while new_path.exists() and new_path != p:
        new_path = p.parent / f"{clean}_{counter}{p.suffix.lower()}"
        counter += 1
    if not CONFIG["dry_run"]:
        p.rename(new_path)
        log.info("  [Rename] %s  →  %s", p.name, new_path.name)
    else:
        log.info("  [DryRun] %s  →  %s", p.name, new_path.name)
    return str(new_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4c: 恢复文件时间戳 (Windows kernel32)
# ══════════════════════════════════════════════════════════════════════════════
def set_file_times_windows(file_path: str, ctime: float,
                            atime: float, mtime: float):
    if os.name != 'nt':
        os.utime(file_path, (atime, mtime))
        return

    def to_filetime(t: float):
        v = int((t + 11644473600.0) * 10_000_000)
        return wintypes.FILETIME(v & 0xFFFFFFFF, v >> 32)

    kernel32 = ctypes.windll.kernel32
    CreateFile = kernel32.CreateFileW
    CreateFile.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE]
    CreateFile.restype  = wintypes.HANDLE

    SetFileTime = kernel32.SetFileTime
    SetFileTime.argtypes = [wintypes.HANDLE,
                             ctypes.POINTER(wintypes.FILETIME),
                             ctypes.POINTER(wintypes.FILETIME),
                             ctypes.POINTER(wintypes.FILETIME)]
    SetFileTime.restype  = wintypes.BOOL

    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype  = wintypes.BOOL

    handle = CreateFile(
        str(file_path),
        0x0100,          # FILE_WRITE_ATTRIBUTES
        0x01 | 0x02,    # FILE_SHARE_READ | FILE_SHARE_WRITE
        None, 3,         # OPEN_EXISTING
        0x02000000,      # FILE_FLAG_BACKUP_SEMANTICS
        None,
    )
    if handle in (-1, 0):
        os.utime(file_path, (atime, mtime))
        return

    SetFileTime(handle,
                ctypes.byref(to_filetime(ctime)),
                ctypes.byref(to_filetime(atime)),
                ctypes.byref(to_filetime(mtime)))
    CloseHandle(handle)


def extract_date_str(video_path: str, creation_time_str: str) -> str:
    if creation_time_str:
        try:
            ct = creation_time_str.replace('Z', '+00:00')
            dt = datetime.datetime.fromisoformat(ct).astimezone()
            return dt.strftime("%Y%m%d_%H%M")
        except Exception:
            pass
    mtime = os.path.getmtime(video_path)
    return datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M")


# ══════════════════════════════════════════════════════════════════════════════
#  错误处理：移动失败文件
# ══════════════════════════════════════════════════════════════════════════════
def _move_to_failed(video_path: str):
    if CONFIG["move_failed"] and not CONFIG["dry_run"]:
        failed_dir = Path(video_path).parent / "_failed"
        failed_dir.mkdir(exist_ok=True)
        dest = failed_dir / Path(video_path).name
        try:
            shutil.move(video_path, dest)
            log.info("  已移至 _failed/: %s", dest.name)
        except Exception as e:
            log.warning("  移动到 _failed/ 失败: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  并发流水线核心
# ══════════════════════════════════════════════════════════════════════════════
# 哨兵对象，用于优雅结束所有 worker 线程
_SENTINEL = object()


class _VideoJob:
    """贯穿整条流水线的"作业上下文"，在各阶段逐步填充数据。"""
    __slots__ = ("video_path", "probe", "orig_stat",
                 "tmp_dir", "transcript", "frame_paths",
                 "ai_result")

    def __init__(self, video_path: str):
        self.video_path:  str             = video_path
        self.probe:       Optional[dict]  = None
        self.orig_stat:   tuple           = (0.0, 0.0, 0.0)  # (ctime, atime, mtime)
        self.tmp_dir:     str             = ""
        self.transcript:  str             = ""
        self.frame_paths: list[str]       = []
        self.ai_result:   Optional[dict]  = None


def _run_pipeline(video_paths: list[str]):
    """
    启动完整三阶段并发流水线并等待所有视频处理完毕。
    """
    total = len(video_paths)

    # ── 队列定义 ──────────────────────────────────────────────────────────────
    # 转写队列：由单个 GPU worker 串行消费
    whisper_q: queue.Queue = queue.Queue()
    # 截图队列：由多个 CPU worker 并行消费
    frame_q:   queue.Queue = queue.Queue()
    # 合并队列：等两路结果都就绪后触发 AI 调用
    # key = video_path, value = {"transcript": ..., "frames": ..., "ready": ...}
    merge_lock = threading.Lock()
    merge_map: dict = {}
    # AI 队列：并发请求 Gemini
    ai_q:      queue.Queue = queue.Queue()
    # 收尾队列：写元数据+重命名
    finalize_q: queue.Queue = queue.Queue()
    # 完成计数器（线程安全），用于判断全部结束
    done_event = threading.Event()
    done_counter = {"n": 0}
    done_lock = threading.Lock()

    global _success, _fail

    def _mark_done():
        with done_lock:
            done_counter["n"] += 1
            if done_counter["n"] >= total:
                done_event.set()

    # ── 辅助：初始化合并表条目 ────────────────────────────────────────────────
    def _init_merge(vp: str):
        with merge_lock:
            if vp not in merge_map:
                merge_map[vp] = {"transcript": None, "frames": None}

    def _try_merge(vp: str, key: str, value):
        """将 transcript 或 frames 写入合并表；若两者都到齐则推入 ai_q。"""
        with merge_lock:
            merge_map[vp][key] = value
            entry = merge_map[vp]
            if entry["transcript"] is not None and entry["frames"] is not None:
                return True, entry["transcript"], entry["frames"]
        return False, None, None

    # ── 阶段：初始化（Probe + 拆分到两队列）─────────────────────────────────
    # 在主线程串行完成 probe（很快），然后同时扔进 whisper_q 和 frame_q
    def _init_all():
        log.info("=" * 60)
        log.info("并发流水线启动，共 %d 个视频", total)
        log.info("=" * 60)
        for vp in video_paths:
            job = _VideoJob(vp)
            try:
                stat = os.stat(vp)
                job.orig_stat = (stat.st_ctime, stat.st_atime, stat.st_mtime)
                job.probe = probe_video(vp)
                
                # 如果发现特殊防重标记，直接跳过处理
                if job.probe.get("is_processed"):
                    log.info("[Probe] \u23ED\uFE0F 跳过已处理视频: %s", Path(vp).name)
                    with _stats_lock:
                        global _skipped
                        _skipped += 1
                    _mark_done()
                    continue

                log.info("\n[Probe] %s", Path(vp).name)
                log.info("  时长=%.1fs  音量=%s dBFS  类型=%s",
                    job.probe["duration"],
                    f"{job.probe['mean_volume']:.1f}" if job.probe["mean_volume"] is not None else "N/A",
                    "双模态" if job.probe["has_audio"] and not is_silent(job.probe["mean_volume"]) else "纯视觉")
                # 创建各自独立的临时目录（生命周期手动管理）
                job.tmp_dir = tempfile.mkdtemp(prefix="vai_")
                _init_merge(vp)
                whisper_q.put(job)
                frame_q.put(job)
            except Exception as e:
                log.error("[FAIL] [Probe] %s 失败: %s", Path(vp).name, e)
                _move_to_failed(vp)
                with _stats_lock:
                    global _fail
                    _fail += 1
                _mark_done()

        # 发送结束哨兵
        whisper_q.put(_SENTINEL)
        for _ in range(CONFIG["keyframe_workers"]):
            frame_q.put(_SENTINEL)

    # ── Worker: 转写 (GPU 独占，单线程) ──────────────────────────────────────
    def _whisper_worker():
        while True:
            job = whisper_q.get()
            if job is _SENTINEL:
                break
            vp = job.video_path
            log.info("[Whisper] 开始转写: %s", Path(vp).name)
            transcript = ""
            try:
                if job.probe["has_audio"] and not is_silent(job.probe["mean_volume"]):
                    transcript = transcribe_audio(vp, job.tmp_dir)
                    log.info("  转写结果 (前150字): %s", transcript[:150])
                else:
                    log.info("  静音或无音频，跳过转写。")
            except Exception as e:
                log.warning("  [Whisper] 转写失败 (将使用空文本): %s", e)

            ready, tr, fr = _try_merge(vp, "transcript", transcript)
            if ready:
                ai_q.put((job, tr, fr))

    # ── Worker: 截图 (CPU 并行) ───────────────────────────────────────────────
    def _frame_worker():
        while True:
            job = frame_q.get()
            if job is _SENTINEL:
                break
            vp = job.video_path
            log.info("[Keyframe] 截图: %s", Path(vp).name)
            frames = []
            try:
                frames = extract_keyframes(vp, job.tmp_dir,
                                           duration=job.probe["duration"])
            except Exception as e:
                log.warning("  [Keyframe] 截图失败: %s", e)

            ready, tr, fr = _try_merge(vp, "frames", frames)
            if ready:
                ai_q.put((job, tr, fr))

    # ── Worker: AI 调度 (并发 IO, 自动路由到所选供应商) ─────────────────────────
    def _ai_worker():
        _provider_label = CONFIG.get("ai_provider", "gemini").upper()
        while True:
            task = ai_q.get()
            if task is _SENTINEL:
                break
            job, transcript, frame_paths = task
            vp = job.video_path
            log.info("[%s] 调用 AI: %s", _provider_label, Path(vp).name)
            try:
                result = query_ai(transcript, frame_paths,
                                  job.probe["creation_time"])
                job.ai_result = result
                log.info("  AI 标题: %s", result.get("title", ""))
                log.info("  AI 描述: %s", result.get("description", "")[:80])
                finalize_q.put(job)
            except Exception as e:
                log.error("[FAIL] [%s] %s 失败: %s", _provider_label, Path(vp).name, e)
                _cleanup_job(job, failed=True)
                _mark_done()

    # ── Worker: 收尾（写元数据+重命名）──────────────────────────────────────
    def _finalize_worker():
        while True:
            job = finalize_q.get()
            if job is _SENTINEL:
                break
            vp = job.video_path
            try:
                result   = job.ai_result
                ai_title = result.get("title", "").strip()
                ai_desc  = result.get("description", "").strip()
                if not ai_title:
                    raise ValueError("AI 返回空标题")

                log.info("[Finalize] 写入元数据+重命名: %s", Path(vp).name)
                if not CONFIG["dry_run"]:
                    write_metadata(vp, ai_title, ai_desc,
                                   job.probe["creation_time"])

                date_prefix = extract_date_str(vp, job.probe["creation_time"])
                final_title = f"{date_prefix}_{ai_title}"
                new_path    = rename_video(vp, final_title)

                orig_ctime, orig_atime, orig_mtime = job.orig_stat
                try:
                    set_file_times_windows(new_path, orig_ctime, orig_atime, orig_mtime)
                except Exception as e:
                    log.warning("  恢复文件时间失败: %s", e)

                log.info("[OK] 完成: %s", final_title)
                with _stats_lock:
                    global _success
                    _success += 1

            except Exception as e:
                log.error("[FAIL] [Finalize] %s 失败: %s", Path(vp).name, e)
                _cleanup_job(job, failed=True)
                with _stats_lock:
                    global _fail
                    _fail += 1
            finally:
                _cleanup_job(job, failed=False)
                _mark_done()

    def _cleanup_job(job: _VideoJob, failed: bool):
        """清理临时目录；若 failed 且配置了移动，则把原视频移入 _failed。"""
        if failed:
            _move_to_failed(job.video_path)
        try:
            if job.tmp_dir and os.path.exists(job.tmp_dir):
                shutil.rmtree(job.tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # ── 启动所有 worker 线程 ──────────────────────────────────────────────────
    threads = []

    t_init = threading.Thread(target=_init_all, name="Init", daemon=True)
    threads.append(t_init)

    t_whisper = threading.Thread(target=_whisper_worker, name="Whisper", daemon=True)
    threads.append(t_whisper)

    for i in range(CONFIG["keyframe_workers"]):
        t = threading.Thread(target=_frame_worker, name=f"Frame-{i}", daemon=True)
        threads.append(t)

    for i in range(CONFIG["ai_workers"]):
        t = threading.Thread(target=_ai_worker, name=f"AI-{i}", daemon=True)
        threads.append(t)

    # finalize 也可以并行，取 ai_workers 数量
    for i in range(CONFIG["ai_workers"]):
        t = threading.Thread(target=_finalize_worker, name=f"Fin-{i}", daemon=True)
        threads.append(t)

    for t in threads:
        t.start()

    # 等待所有视频处理完毕
    done_event.wait()

    # 广播哨兵给 AI 和 Finalize workers
    for _ in range(CONFIG["ai_workers"]):
        ai_q.put(_SENTINEL)
    for _ in range(CONFIG["ai_workers"]):
        finalize_q.put(_SENTINEL)

    # 等待所有 worker 线程退出
    for t in threads:
        t.join(timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
#  批量处理入口
# ══════════════════════════════════════════════════════════════════════════════
def collect_videos(folder: str) -> list[str]:
    ext_set = CONFIG["video_extensions"]
    videos  = []
    for root, _, files in os.walk(folder):
        if os.path.basename(root) == "_failed":
            continue
        for f in sorted(files):
            if Path(f).suffix.lower() in ext_set:
                videos.append(os.path.join(root, f))
    return videos


def run_batch(folder: str):
    global _success, _fail, _skipped
    _success = _fail = _skipped = 0

    log.info("扫描文件夹: %s", folder)
    videos = collect_videos(folder)
    total  = len(videos)

    if total == 0:
        log.info("未找到视频文件，退出。")
        return

    log.info("共找到 %d 个视频文件，启动并发流水线...\n", total)
    _run_pipeline(videos)

    log.info("\n" + "=" * 60)
    log.info("全部处理完成: [OK] 成功 %d  [\u23ED\uFE0F 跳过] %d  [FAIL] 失败 %d  共计 %d",
             _success, _skipped, _fail, total)
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════════════════════════════════════
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description="视频 AI 自动重命名工具 —— 并发流水线版",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("folder", nargs="?", default=".",
                        help="要处理的视频文件夹路径（默认: 当前目录）")
    parser.add_argument("--api-key", default="",
                        help="Gemini API Key")
    parser.add_argument("--model", default=CONFIG["gemini_model"],
                        help=f"Gemini 模型名称 (默认: {CONFIG['gemini_model']})")
    parser.add_argument("--whisper-model", default=CONFIG["whisper_model"],
                        help=f"Faster-Whisper 模型 (默认: {CONFIG['whisper_model']})")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式：只分析，不写入/重命名")
    parser.add_argument("--keyframe-workers", type=int,
                        default=CONFIG["keyframe_workers"],
                        help=f"截图并发线程数 (默认: {CONFIG['keyframe_workers']})")
    parser.add_argument("--ai-workers", type=int,
                        default=CONFIG["ai_workers"],
                        help=f"AI/收尾并发线程数 (默认: {CONFIG['ai_workers']})")
    parser.add_argument("--max-keyframes", type=int,
                        default=CONFIG["max_keyframes"],
                        help=f"最多提取关键帧数量 (默认: {CONFIG['max_keyframes']})")
    parser.add_argument("--silence-db", type=float,
                        default=CONFIG["silence_threshold_db"],
                        help=f"静音阈值 dBFS (默认: {CONFIG['silence_threshold_db']})")
    parser.add_argument("--no-move-failed", action="store_true",
                        help="失败文件不移入 _failed/")
    parser.add_argument("--log-file", default="",
                        help="将日志同时写入文件（可选）")

    args = parser.parse_args()

    if args.api_key:
        CONFIG["gemini_api_key"] = args.api_key
    CONFIG["gemini_model"]          = args.model
    CONFIG["whisper_model"]         = args.whisper_model
    CONFIG["dry_run"]               = args.dry_run
    CONFIG["keyframe_workers"]      = args.keyframe_workers
    CONFIG["ai_workers"]            = args.ai_workers
    CONFIG["max_keyframes"]         = args.max_keyframes
    CONFIG["silence_threshold_db"]  = args.silence_db
    if args.no_move_failed:
        CONFIG["move_failed"] = False

    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        log.error("路径不存在或不是目录: %s", folder)
        sys.exit(1)

    if args.dry_run:
        log.info("[DryRun] 干跑模式，不修改任何文件。")

    run_batch(folder)


if __name__ == "__main__":
    main()
