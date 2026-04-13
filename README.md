# 视频 AI 自动重命名工具

基于 **Faster-Whisper + Gemini 多模态** 的视频批量分类与重命名工具。

## 目录结构

```
AiVideoRename/
├── video_ai_rename.py      ← 主程序
├── requirements.txt
├── README.md
└── ffmpeg/
    ├── ffmpeg.exe
    ├── ffprobe.exe
    └── exiftool.exe
```

## 依赖安装

```bash
pip install -r requirements.txt
```

> **注意**: `faster-whisper` 首次运行会自动从 HuggingFace 下载模型（large-v3 约 3GB）。
> 如网络受限，可提前手动下载或改用较小模型（`--whisper-model small`）。

## 配置 API Key

方式一（推荐）：设置环境变量
```powershell
$env:GEMINI_API_KEY = "你的API_Key"
```

方式二：直接修改脚本顶部 `CONFIG["gemini_api_key"]`

方式三：命令行参数 `--api-key 你的Key`

> 获取 Gemini API Key: https://aistudio.google.com/app/apikey

## 使用方法

```bash
# 处理指定文件夹内所有视频
python video_ai_rename.py D:\Videos\旅行视频

# 干跑模式（只分析，不修改文件）
python video_ai_rename.py D:\Videos --dry-run

# 指定较小的 Whisper 模型（速度更快，精度稍低）
python video_ai_rename.py D:\Videos --whisper-model small

# 调整场景切换灵敏度（越小越灵敏，提取更多帧）
python video_ai_rename.py D:\Videos --scene-threshold 0.3

# 同时写入日志文件
python video_ai_rename.py D:\Videos --log-file rename_log.txt

# 查看所有参数
python video_ai_rename.py --help
```

## Pipeline 说明

```
Step 1: ffprobe 检测音频
        ├── 有音频 → 双模态任务
        └── 无音频/静音 → 纯视觉任务

Step 2: 特征提取
        ├── 2a: Faster-Whisper 语音转写（仅双模态）
        └── 2b: FFmpeg 场景切换关键帧抽取（最多3张，720P）

Step 3: Gemini 推理
        └── 转写文本 + 关键帧 → 生成标题 + 描述

Step 4: 写入 + 重命名
        ├── FFmpeg 写入 MP4 元数据 (title/comment/creation_time)
        └── 文件重命名
```

## 输出元数据

| Tag | 内容 |
|-----|------|
| `title` | AI 生成的 12 字内简洁标题 |
| `comment` | AI 生成的 100 字内视频描述 |
| `creation_time` | 保留原始录制时间 |

> 写入后，Windows 资源管理器「属性」中可查看 title/comment 字段。

## 常用参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-key` | — | Gemini API Key |
| `--model` | `gemini-3-flash-preview` | Gemini 模型名称 |
| `--whisper-model` | `large-v3-turbo` | Whisper 模型大小 |
| `--dry-run` | False | 干跑，不修改文件 |
| `--scene-threshold` | `0.4` | 场景切换阈值 |
| `--max-keyframes` | `3` | 最大关键帧数 |
| `--silence-db` | `-60.0` | 静音判断阈值(dBFS) |
| `--no-move-failed` | — | 失败不移入 _failed/ |
| `--log-file` | — | 日志输出文件 |

## 注意事项

- 处理失败的视频会自动移入目标文件夹的 `_failed/` 子目录，不影响其余视频处理。
- 脚本会递归扫描子目录，自动跳过 `_failed/` 目录。
- 支持格式：`.mp4 .mov .avi .mkv .m4v .wmv .flv .webm .ts .mts .m2ts`
- GPU 加速：若 CUDA 可用，Faster-Whisper 自动使用 GPU（float16），否则回退 CPU（int8）。
