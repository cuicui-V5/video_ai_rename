import os
import argparse
import time
import math

# --- 自动处理 Windows 端 pip 安装的 CUDA DLL（由于 Python 3.8+ 安全策略导致动态库加载失败） ---
if os.name == 'nt':
    try:
        import site
        # 获取所有 Python 的 site-packages 路径
        packages = site.getsitepackages()
        if hasattr(site, 'getusersitepackages'):
            packages.append(site.getusersitepackages())
        for pkg_dir in packages:
            nvidia_dir = os.path.join(pkg_dir, "nvidia")
            if os.path.exists(nvidia_dir):
                for module_name in os.listdir(nvidia_dir):
                    bin_dir = os.path.join(nvidia_dir, module_name, "bin")
                    if os.path.exists(bin_dir):
                        # 添加到环境变量
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                        # 添加到 DLL 加载路径 (Python >= 3.8 Windows)
                        if hasattr(os, 'add_dll_directory'):
                            os.add_dll_directory(bin_dir)
    except Exception:
        pass
# ----------------------------------------------------------------------------------------------------

from faster_whisper import WhisperModel

def format_timestamp(seconds: float) -> str:
    """
    格式化时间戳为标准的 srt 格式: HH:MM:SS,MMM
    """
    milliseconds = round(seconds * 1000.0)
    
    hours = milliseconds // 3600000
    milliseconds -= hours * 3600000
    
    minutes = milliseconds // 60000
    milliseconds -= minutes * 60000
    
    sec = milliseconds // 1000
    milliseconds -= sec * 1000
    
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{milliseconds:03d}"

def split_words_into_chunks(words, max_chars=30):
    """
    根据词级时间戳和标点符号，将长段落智能切分为更短的字幕行。
    切分规则：
    1. 遇到句号、问号、叹号等强停顿符。
    2. 遇到逗号等弱停顿符，且当前长度已超过 15 个字符。
    3. 极端情况：没有任何标点，强制在达到 max_chars (30字) 时切断。
    """
    chunks = []
    current_words = []
    current_start = None
    last_word_end = None
    
    major_punct = ('。', '！', '？', '!', '?', '；', ';')
    minor_punct = ('，', ',', '、')
    
    for word in words:
        if current_start is None:
            current_start = word.start
            
        current_words.append(word)
        last_word_end = word.end
        
        # 拼接当前文本（去头尾空格计算长度）
        current_text = "".join(w.word for w in current_words).strip()
        clean_len = len(current_text)
        word_str = word.word.strip()
        
        hit_major = word_str.endswith(major_punct)
        hit_minor = word_str.endswith(minor_punct)
        
        # 判断切分逻辑
        if hit_major or clean_len >= max_chars or (clean_len >= 15 and hit_minor):
            chunks.append({
                'start': current_start,
                'end': last_word_end,
                'text': current_text
            })
            current_words = []
            current_start = None

    # 处理最后剩余的部分
    if current_words:
        chunks.append({
            'start': current_start,
            'end': last_word_end,
            'text': "".join(w.word for w in current_words).strip()
        })
        
    return chunks

def convert_video_to_srt(video_path: str, output_srt_path: str = None):
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 '{video_path}'")
        return

    # 确定输出的 srt 文件路径
    if output_srt_path is None:
        base_name, _ = os.path.splitext(video_path)
        output_srt_path = f"{base_name}.srt"
        
    print(f"正在加载 faster-whisper 模型 (large-v3-turbo) ...")
    model_size = "large-v3-turbo"
    
    # 尝试加载模型，优先使用 GPU
    try:
        print("尝试使用 GPU (CUDA) 加载模型...")
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("✅ 成功加载模型到 GPU (CUDA)")
    except Exception as e:
        print(f"⚠️ 加载 GPU 模型失败，尝试退回到 CPU: {e}")
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("✅ 成功加载模型到 CPU")
        except Exception as cpu_e:
            print(f"❌ 加载 CPU 模型失败: {cpu_e}")
            return
            
    print(f"\n开始转写视频: {video_path}")
    start_time = time.time()
    
    # beam_size=5 提升准确度
    # word_timestamps=True 开启词级时间戳以支持精准分句设定的字数
    # vad_filter=True 过滤静音片段，减少大片空白导致的幻觉
    segments, info = model.transcribe(video_path, beam_size=5, word_timestamps=True, vad_filter=True)
    
    print(f"识别到主音频语言: {info.language} (置信度: {info.language_probability:.3f})")
    print("正在生成按语义完美切分的 SRT 字幕文件...\n")

    # 打开文件并写入 SRT 标准格式
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        subtitle_index = 1
        for segment in segments:
            # 拿到这一大段识别出的单词(Token)数据后，交给智能切分函数去分片
            if segment.words:
                chunks = split_words_into_chunks(segment.words, max_chars=30)
            else:
                chunks = [{'start': segment.start, 'end': segment.end, 'text': segment.text.strip()}]
                
            for chunk in chunks:
                text = chunk['text']
                if not text:
                    continue
                    
                start_time_str = format_timestamp(chunk['start'])
                end_time_str = format_timestamp(chunk['end'])
                
                # 写入 SRT
                srt_file.write(f"{subtitle_index}\n")
                srt_file.write(f"{start_time_str} --> {end_time_str}\n")
                srt_file.write(f"{text}\n\n")
                
                print(f"[{start_time_str} --> {end_time_str}] {text}")
                subtitle_index += 1

    end_time = time.time()
    print(f"\n转写完成! 总耗时: {end_time - start_time:.2f} 秒")
    print(f"🎉 智能短句格式字幕文件已成功保存到: {output_srt_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 faster-whisper (large-v3-turbo) 将视频提取并转写为 SRT 字幕")
    parser.add_argument("video", type=str, help="需要转手写字幕的视频文件路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="自定义输出的 .srt 文件路径 (可选，默认与视频同名并保存在同目录)")
    
    args = parser.parse_args()
    
    convert_video_to_srt(args.video, args.output)
