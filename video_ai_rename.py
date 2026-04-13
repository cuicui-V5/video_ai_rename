#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_ai_rename.py
==================
视频自动化分类与批量重命名工具

Pipeline:
  Step 1: ffprobe 检测音频流 (静音 / 双模态)
  Step 2: Faster-Whisper 转写 + FFmpeg 场景切换关键帧提取
  Step 3: Gemini 多模态模型生成文件名与描述
  Step 4: FFmpeg 写入元数据 + 文件重命名

工具依赖 (置于 ffmpeg/ 目录):
  ffmpeg.exe / ffprobe.exe / exiftool.exe

Python 依赖:
  pip install faster-whisper google-generativeai tqdm
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
from ctypes import wintypes
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# 修复 Windows 下 Faster-Whisper (CTranslate2) 找不到 CUDA DLL 的问题
# ─────────────────────────────────────────────
if os.name == 'nt':
    try:
        import site
        # 收集所有可能的 site-packages 路径（含用户级安装目录）
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
    # Gemini API Key (也可通过环境变量 GEMINI_API_KEY 传入)
    "gemini_api_key": "AIzaSyBO3n_XGmOazLRT1__MYTePyD4UNqDx7HQ",

    # Gemini 模型名称
    "gemini_model": "gemini-3-flash-preview",

    # Faster-Whisper 模型大小: tiny / base / small / medium / large-v3 / large-v3-turbo
    "whisper_model": "large-v3-turbo",

    # Faster-Whisper 计算精度: float16 / int8_float16 / int8
    "whisper_compute_type": "int8_float16",

    # 场景切换阈值 (0.0~1.0, 越小越灵敏)
    "scene_threshold": 0.4,

    # 关键帧最多提取数量
    "max_keyframes": 3,

    # 关键帧缩放宽度 (px)
    "keyframe_width": 720,

    # 静音判断阈值 (dBFS), 超过此值认为有音频
    "silence_threshold_db": -60.0,

    # 视频文件扩展名白名单
    "video_extensions": {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv",
                         ".flv", ".webm", ".ts", ".mts", ".m2ts"},

    # ffmpeg / ffprobe / exiftool 所在目录 (相对于本脚本)
    "tools_dir": "ffmpeg",

    # 是否将处理失败的文件移入 _failed 子目录
    "move_failed": True,

    # 是否在重命名前备份原始文件 (原文件名追加 .bak)
    "backup_original": False,

    # 干跑模式 (只打印不执行写入/重命名)
    "dry_run": False,
}

# ─────────────────────────────────────────────
#  日志配置
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("VideoAIRename")


# ══════════════════════════════════════════════════════════════════════════════
#  工具路径解析
# ══════════════════════════════════════════════════════════════════════════════
def _tool(name: str) -> str:
    """返回工具的绝对路径。优先使用 ffmpeg/ 子目录中的版本，否则回退到系统 PATH。"""
    script_dir = Path(__file__).parent
    local = script_dir / CONFIG["tools_dir"] / name
    if local.exists():
        return str(local)
    # 回退到系统 PATH
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"找不到工具: {name}。请将其放入 {CONFIG['tools_dir']}/ 目录或确保系统 PATH 中存在。"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: 媒体信息检测
# ══════════════════════════════════════════════════════════════════════════════
def probe_video(video_path: str) -> dict:
    """
    使用 ffprobe 获取视频的基本信息。
    返回 dict 包含: has_audio, mean_volume, duration, creation_time
    """
    cmd = [
        _tool("ffprobe.exe"),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(fmt.get("duration", 0))

    # 提取 creation_time
    creation_time = fmt.get("tags", {}).get("creation_time", "")

    # 检测平均音量
    mean_volume = None
    if has_audio:
        mean_volume = _detect_volume(video_path)

    return {
        "has_audio": has_audio,
        "mean_volume": mean_volume,  # dBFS float or None
        "duration": duration,
        "creation_time": creation_time,
    }


def _detect_volume(video_path: str) -> float:
    """用 ffmpeg volumedetect filter 检测平均音量，返回 mean_volume (dBFS)。"""
    cmd = [
        _tool("ffmpeg.exe"),
        "-i", video_path,
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    stderr = result.stderr
    match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
    if match:
        return float(match.group(1))
    return CONFIG["silence_threshold_db"] - 1  # 无法检测时默认视为静音


def is_silent(mean_volume: Optional[float]) -> bool:
    if mean_volume is None:
        return True
    return mean_volume < CONFIG["silence_threshold_db"]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2a: 音频转写 (Faster-Whisper)
# ══════════════════════════════════════════════════════════════════════════════

# 全局缓存模型实例，防止跨视频重新加载，同时避免 Windows 下 C++ 析构导致进程闪退
_WHISPER_MODEL_CACHE = None

def transcribe_audio(video_path: str, tmp_dir: str) -> str:
    """
    提取音频并用 Faster-Whisper 转写，返回纯文本字符串。
    """
    global _WHISPER_MODEL_CACHE
    audio_path = os.path.join(tmp_dir, "audio.wav")

    # 提取音频为 WAV (16kHz mono)
    cmd_extract = [
        _tool("ffmpeg.exe"),
        "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    r = subprocess.run(cmd_extract, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(f"音频提取失败: {r.stderr[-500:]}")

    # Faster-Whisper 转写
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning("faster-whisper 未安装，跳过语音转写。执行: pip install faster-whisper")
        return ""

    if _WHISPER_MODEL_CACHE is None:
        log.info("  [Whisper] 加载模型 %s (%s)...", CONFIG["whisper_model"], CONFIG["whisper_compute_type"])
        try:
            model = WhisperModel(
                CONFIG["whisper_model"],
                device="auto",
                compute_type=CONFIG["whisper_compute_type"],
            )
            _WHISPER_MODEL_CACHE = model
        except Exception as e:
            if "cuda" in str(e).lower() or "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "loaded" in str(e).lower():
                log.warning("  [Whisper] GPU 运行失败 (CUDA未配置完整)，自动回退至纯 CPU 模式。详细错误: %s", str(e).split('\n')[0])
                model = WhisperModel(
                    CONFIG["whisper_model"],
                    device="cpu",
                    compute_type="int8",
                )
                _WHISPER_MODEL_CACHE = model
            else:
                raise
    else:
        log.info("  [Whisper] 重用已加载的模型")

    segments, info = _WHISPER_MODEL_CACHE.transcribe(audio_path, beam_size=5)
    
    log.info("  [Whisper] 检测语言: %s (概率 %.2f)，视频总长度: %.1f秒", info.language, info.language_probability, info.duration)

    # 接入全动态进度条
    from tqdm import tqdm
    text_segments = []
    
    with tqdm(total=round(info.duration, 2), unit="s", desc="  [转写进度]", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for seg in segments:
            text_segments.append(seg.text.strip())
            # 更新进度条到当前片段的结束时间（相比上次增长的值）
            pbar.update(seg.end - pbar.n)
            
        pbar.update(info.duration - pbar.n) # 补齐最后一点尾巴

    text = " ".join(text_segments)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2b: 智能关键帧提取
# ══════════════════════════════════════════════════════════════════════════════
def extract_keyframes(video_path: str, tmp_dir: str, duration: float = 0.0) -> list[str]:
    """
    极速等间距抽取关键帧（完全废弃消耗极高 CPU 的原味场景检测）。
    """
    log.info("  [Keyframe] 使用等间距法极速截取画面...")
    frames = _extract_uniform_frames(video_path, tmp_dir, CONFIG["max_keyframes"], duration)
    
    frames = frames[: CONFIG["max_keyframes"]]
    log.info("  [Keyframe] 共提取 %d 张关键帧", len(frames))
    return [str(f) for f in frames]


def _extract_uniform_frames(video_path: str, tmp_dir: str, count: int, duration: float) -> list[Path]:
    """使用高阶 -ss 输入跳转指令（Input Seeking）进行纳秒级的等间距截取。"""
    # 先清理旧帧
    for old in Path(tmp_dir).glob("frame_*.jpg"):
        old.unlink(missing_ok=True)

    if duration <= 0:
        duration = 30.0  # 保底
        
    interval = max(duration / (count + 1), 1)
    frames_found = []
    
    for i in range(1, count + 1):
        target_time = interval * i
        out_path = os.path.join(tmp_dir, f"frame_uniform_{i:03d}.jpg")
        cmd = [
            _tool("ffmpeg.exe"),
            "-y", 
            "-ss", f"{target_time:.2f}",   # 利用输入侧极其快速的 Keyframe 跳转
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            "-vf", f"scale={CONFIG['keyframe_width']}:-1",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out_path):
            frames_found.append(Path(out_path))
            
    return frames_found


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Gemini 多模态推理
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你现在是一个专业的视频内容分析专家。
我会为你提供：
1. 视频音频的转写文本（若有）
2. 若干关键帧截图

请结合两者完成以下两项任务：
1. 提取出一个极其简练精锐的短词汇组合作为视频【标题】。绝对不要在标题中输出类似于日期、时间的数字前缀！
2. 撰写一段详细生动的【内容描述】，概括视频中的场景、核心主体、人物动作以及故事脉络。"""


def query_gemini(
    transcript: str,
    frame_paths: list[str],
    creation_time: str,
) -> dict:
    """
    调用最新版的 google-genai 接口进行多模态推理，强制返回 JSON。
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("请安装最新版 SDK: pip install google-genai")

    api_key = CONFIG["gemini_api_key"] or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 Gemini API Key。请在 CONFIG['gemini_api_key'] 或环境变量 GEMINI_API_KEY 中配置。")

    client = genai.Client(api_key=api_key)

    # 构建消息片段
    contents = []

    # 转写文本与时间说明
    text_prompt = ""
    if transcript:
        text_prompt += f"【音频转写文本】\n{transcript}\n\n"
    else:
        text_prompt += "【音频转写文本】\n（无音频或无法识别的语音内容）\n\n"

    if creation_time:
        text_prompt += f"【原始录制时间参考】\n{creation_time}\n\n"
        
    text_prompt += "【请结合以上文本并参考附带的关键帧图片进行判定】"
    contents.append(text_prompt)

    # 附带关键帧图片
    for fp in frame_paths:
        with open(fp, "rb") as f:
            img_bytes = f.read()
        contents.append(
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        )

    from pydantic import BaseModel, Field
    class VideoMetadata(BaseModel):
        title: str = Field(description="12字以内的简洁标题。请只返回概括性文字，【绝对不要】包含任何形式的日期时间或数字前缀，不要标点符号！")
        description: str = Field(description="100字以内的详细中文描述，概括视频主要内容")

    # 新版 SDK 使用 client.models.generate_content
    # 并强制应用 Pydantic 结构化输出保证数据不被意外截断或排版错误
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

    finish_reason = response.candidates[0].finish_reason.name if response.candidates and response.candidates[0].finish_reason else "UNKNOWN"
    raw = response.text.strip() if response.text else "{}"
    log.info("  [Gemini] 结束原因: %s | 原始响应: %s", finish_reason, raw)

    # 直接反序列化，因为 API 确保会返回标准的 JSON 串
    try:
        return json.loads(raw)
    except Exception:
        raise ValueError(f"Gemini 返回的强类型 JSON 解析失败: {raw}")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4a: ExifTool 写入元数据 (零拷贝极速模式)
# ══════════════════════════════════════════════════════════════════════════════
def write_metadata(video_path: str, title: str, description: str, creation_time: str) -> str:
    """
    使用 ExifTool 极速修改 MP4 元数据，避免 FFmpeg 整体重写文件。
    采用 -@ 传参文件模式，彻底根除 Windows 控制台的 GBK 乱码现象。
    """
    # 清理引起 argfile 解析错误的回车符
    title = title.replace('\n', ' ').strip()
    description = description.replace('\n', ' ').strip()

    args = [
        "-m", "-q", "-overwrite_original",
        "-charset", "utf8",
    ]
    
    # argfile 必须一行为一个完整参数，切忌加引号！
    if title:
        args.append(f"-Title={title}")
        args.append(f"-ItemList:Title={title}")
    if description:
        args.append(f"-Description={description}")
        args.append(f"-Comment={description}")
        args.append(f"-ItemList:Comment={description}")
    if creation_time:
        args.append(f"-CreateDate={creation_time}")
        
    args.append(video_path)

    arg_file = video_path + ".exifargs"
    try:
        with open(arg_file, "w", encoding="utf-8") as f:
            for arg in args:
                f.write(arg + "\n")
                
        cmd = [_tool("exiftool.exe"), "-@", arg_file]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if r.returncode != 0:
            log.warning("[\u26A0\uFE0FExifTool] 写入元数据发生异常: %s", r.stderr[-500:])
    finally:
        if os.path.exists(arg_file):
            os.remove(arg_file)
    
    return video_path


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4b: 文件重命名
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_filename(name: str) -> str:
    """去除 Windows 不允许的文件名字符。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def rename_video(video_path: str, new_stem: str) -> str:
    """将视频文件重命名为新名称，若目标已存在则追加序号。返回新路径。"""
    p = Path(video_path)
    new_stem_clean = sanitize_filename(new_stem)
    new_name = new_stem_clean + p.suffix.lower()
    new_path = p.parent / new_name

    # 防止同名冲突
    counter = 1
    while new_path.exists() and new_path != p:
        new_name = f"{new_stem_clean}_{counter}{p.suffix.lower()}"
        new_path = p.parent / new_name
        counter += 1

    if not CONFIG["dry_run"]:
        p.rename(new_path)
        log.info("  [Rename] %s  →  %s", p.name, new_path.name)
    else:
        log.info("  [DryRun] 将重命名:  %s  →  %s", p.name, new_path.name)

    return str(new_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4c: 恢复文件时间戳 (Windows 专属)
# ══════════════════════════════════════════════════════════════════════════════
def set_file_times_windows(file_path: str, ctime: float, atime: float, mtime: float):
    """
    修改文件的系统属性：创建时间(ctime), 访问时间(atime), 修改时间(mtime)。
    必须在 Windows 平台通过 ctypes 调用 kernel32 API，否则 Python 默认的 os.utime 无法修改创建时间。
    """
    if os.name != 'nt':
        os.utime(file_path, (atime, mtime))
        return

    def to_filetime(epoch_time: float):
        intervals = int((epoch_time + 11644473600.0) * 10000000)
        return wintypes.FILETIME(intervals & 0xFFFFFFFF, intervals >> 32)

    FILE_WRITE_ATTRIBUTES = 0x0100
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    
    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    
    set_file_time = ctypes.windll.kernel32.SetFileTime
    set_file_time.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
    set_file_time.restype = wintypes.BOOL
    
    close_handle = ctypes.windll.kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    
    # 强制获取写属性权限，并开启共享读写，以防被防病毒软件等独占锁卡死导致创建失败
    handle = create_file(
        str(file_path), 
        FILE_WRITE_ATTRIBUTES, 
        FILE_SHARE_READ | FILE_SHARE_WRITE, 
        None, 
        OPEN_EXISTING, 
        FILE_FLAG_BACKUP_SEMANTICS, 
        None
    )
    if handle == -1 or handle == 0: # INVALID_HANDLE_VALUE
        os.utime(file_path, (atime, mtime)) # fallback
        return
        
    c_ft = to_filetime(ctime)
    a_ft = to_filetime(atime)
    m_ft = to_filetime(mtime)
    
    set_file_time(handle, ctypes.byref(c_ft), ctypes.byref(a_ft), ctypes.byref(m_ft))
    close_handle(handle)


def extract_date_str(video_path: str, creation_time_str: str) -> str:
    """提取 YYYYMMDD_HHMM 格式的日期字符串，用于文件重命名前缀。"""
    # 1. 尝试解析元数据中的 creation_time (例如 "2024-08-01T12:00:00.000000Z")
    if creation_time_str:
        try:
            # 将 Z 替换为标准时区标识，确保被解析为 UTC 时间，并转换为本地时区(如北京时间)
            creation_time_str = creation_time_str.replace('Z', '+00:00')
            dt = datetime.datetime.fromisoformat(creation_time_str)
            dt = dt.astimezone()  # 自动通过操作系统时差偏移补上那8个小时
            return dt.strftime("%Y%m%d_%H%M")
        except Exception:
            pass
    
    # 2. 如果元数据没有，回退到底层文件系统的修改时间 (mtime)
    mtime = os.path.getmtime(video_path)
    dt = datetime.datetime.fromtimestamp(mtime)
    return dt.strftime("%Y%m%d_%H%M")

# ══════════════════════════════════════════════════════════════════════════════
#  主处理流程：单个视频
# ══════════════════════════════════════════════════════════════════════════════
def process_video(video_path: str) -> bool:
    """
    处理单个视频文件，完成整个 Pipeline。
    返回 True 表示成功，False 表示失败。
    """
    log.info("=" * 60)
    log.info("处理: %s", os.path.basename(video_path))
    log.info("=" * 60)

    with tempfile.TemporaryDirectory(prefix="video_ai_") as tmp_dir:
        try:
            # ── Step 1: Probe ──────────────────────────────────────────
            # 记录文件系统最原始的 timestamps 备后期还原
            orig_stat = os.stat(video_path)
            orig_ctime = orig_stat.st_ctime
            orig_mtime = orig_stat.st_mtime
            orig_atime = orig_stat.st_atime

            log.info("[Step 1] 检测媒体信息...")
            probe = probe_video(video_path)
            log.info(
                "  时长=%.1fs, has_audio=%s, volume=%s dBFS",
                probe["duration"],
                probe["has_audio"],
                f"{probe['mean_volume']:.1f}" if probe["mean_volume"] is not None else "N/A",
            )

            task_type = "纯视觉任务"
            if probe["has_audio"] and not is_silent(probe["mean_volume"]):
                task_type = "双模态任务"
            log.info("  任务类型: %s", task_type)

            # ── Step 2: 特征提取 ───────────────────────────────────────
            transcript = ""
            if task_type == "双模态任务":
                log.info("[Step 2a] 音频转写...")
                try:
                    transcript = transcribe_audio(video_path, tmp_dir)
                    log.info("  转写结果 (前200字): %s", transcript[:200])
                except Exception as e:
                    log.warning("  转写失败 (将跳过): %s", e)

            log.info("[Step 2b] 提取关键帧...")
            frame_paths = extract_keyframes(video_path, tmp_dir, duration=probe["duration"])

            # ── Step 3: Gemini 推理 ────────────────────────────────────
            log.info("[Step 3] 调用 Gemini 多模态模型...")
            result = query_gemini(transcript, frame_paths, probe["creation_time"])
            ai_title = result.get("title", "").strip()
            ai_desc = result.get("description", "").strip()
            log.info("  AI 标题: %s", ai_title)
            log.info("  AI 描述: %s", ai_desc)

            if not ai_title:
                raise ValueError("Gemini 未返回有效标题")

            # ── Step 4: 写入元数据 + 重命名 ───────────────────────────
            log.info("[Step 4] 写入元数据 (ExifTool 极速模式)...")
            if not CONFIG["dry_run"]:
                write_metadata(video_path, ai_title, ai_desc, probe["creation_time"])

            log.info("[Step 4] 重命名并还原时间戳...")
            
            # 读取日期拼接前缀
            date_prefix = extract_date_str(video_path, probe["creation_time"])
            final_title = f"{date_prefix}_{ai_title}"
            
            new_path_str = rename_video(video_path, final_title)

            # 彻底还原被抹杀的文件创建时间和修改时间
            try:
                set_file_times_windows(new_path_str, orig_ctime, orig_atime, orig_mtime)
            except Exception as e:
                log.warning("  恢复文件系统时间失败: %s", e)

            log.info("✅ 处理完成: %s", final_title)
            return True

        except Exception as e:
            log.error("❌ 处理失败: %s", e, exc_info=True)

            # 清理可能残留的临时输出文件
            tmp_out_maybe = video_path + ".meta_tmp.mp4"
            if os.path.exists(tmp_out_maybe):
                try:
                    os.remove(tmp_out_maybe)
                except Exception:
                    pass

            if CONFIG["move_failed"]:
                failed_dir = Path(video_path).parent / "_failed"
                failed_dir.mkdir(exist_ok=True)
                dest = failed_dir / Path(video_path).name
                try:
                    if not CONFIG["dry_run"]:
                        shutil.move(video_path, dest)
                        log.info("  已移至 _failed/: %s", dest.name)
                except Exception as mv_err:
                    log.warning("  移动到 _failed/ 失败: %s", mv_err)

            return False


# ══════════════════════════════════════════════════════════════════════════════
#  批量处理入口
# ══════════════════════════════════════════════════════════════════════════════
def collect_videos(folder: str) -> list[str]:
    """递归收集文件夹内所有视频文件路径。"""
    ext_set = CONFIG["video_extensions"]
    videos = []
    for root, _, files in os.walk(folder):
        # 跳过 _failed 目录
        if os.path.basename(root) == "_failed":
            continue
        for f in sorted(files):
            if Path(f).suffix.lower() in ext_set:
                videos.append(os.path.join(root, f))
    return videos


def run_batch(folder: str):
    log.info("扫描文件夹: %s", folder)
    videos = collect_videos(folder)
    total = len(videos)

    if total == 0:
        log.info("未找到视频文件，退出。")
        return

    log.info("共找到 %d 个视频文件，开始处理...\n", total)

    success_count = 0
    fail_count = 0

    for idx, vp in enumerate(videos, 1):
        log.info("\n[%d/%d] %s", idx, total, vp)
        ok = process_video(vp)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    log.info("\n" + "=" * 60)
    log.info("批量处理完成: 成功 %d / 失败 %d / 共计 %d", success_count, fail_count, total)
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="视频 AI 自动重命名工具 —— 批量转写 + 关键帧分析 + Gemini 命名",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="要处理的视频文件夹路径（默认: 当前目录）",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Gemini API Key (也可通过环境变量 GEMINI_API_KEY 设置)",
    )
    parser.add_argument(
        "--model",
        default=CONFIG["gemini_model"],
        help=f"Gemini 模型名称 (默认: {CONFIG['gemini_model']})",
    )
    parser.add_argument(
        "--whisper-model",
        default=CONFIG["whisper_model"],
        help=f"Faster-Whisper 模型大小 (默认: {CONFIG['whisper_model']})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：只分析，不执行写入和重命名",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=CONFIG["scene_threshold"],
        help=f"场景切换检测阈值 0.0~1.0 (默认: {CONFIG['scene_threshold']})",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=CONFIG["max_keyframes"],
        help=f"最多提取关键帧数量 (默认: {CONFIG['max_keyframes']})",
    )
    parser.add_argument(
        "--silence-db",
        type=float,
        default=CONFIG["silence_threshold_db"],
        help=f"静音判断阈值 dBFS (默认: {CONFIG['silence_threshold_db']})",
    )
    parser.add_argument(
        "--no-move-failed",
        action="store_true",
        help="失败文件不移入 _failed/ 目录",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="将日志同时写入文件（可选）",
    )

    args = parser.parse_args()

    # 应用 CLI 参数到配置
    if args.api_key:
        CONFIG["gemini_api_key"] = args.api_key
    CONFIG["gemini_model"] = args.model
    CONFIG["whisper_model"] = args.whisper_model
    CONFIG["dry_run"] = args.dry_run
    CONFIG["scene_threshold"] = args.scene_threshold
    CONFIG["max_keyframes"] = args.max_keyframes
    CONFIG["silence_threshold_db"] = args.silence_db
    if args.no_move_failed:
        CONFIG["move_failed"] = False

    # 日志文件
    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        log.error("路径不存在或不是目录: %s", folder)
        sys.exit(1)

    if args.dry_run:
        log.info("⚡ 干跑模式已启用，不会修改任何文件。")

    run_batch(folder)


if __name__ == "__main__":
    main()
