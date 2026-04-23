"""Chuyển đổi audio sang WAV chuẩn cho voice cloning."""

import os
import sys
from pydub import AudioSegment


def convert_to_standard_wav(input_path: str, output_path: str = None):
    if output_path is None:
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join("data", "reference_voices", f"{name}_converted.wav")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"📂 Input  : {input_path}")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
    audio.export(output_path, format="wav")

    duration = len(audio) / 1000.0
    print(f"💾 Output : {output_path}")
    print(f"⏱️  Length : {duration:.1f} giây")

    if duration < 3:
        print("⚠️  File quá ngắn (< 3s). Nên dùng 5-15 giây.")
    elif duration > 30:
        print("⚠️  File quá dài (> 30s). Nên cắt lại 5-15 giây.")

    print("✅ Chuyển đổi thành công!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng:")
        print("  python scripts/convert_audio.py <đường_dẫn_file_audio>")
        print("  Ví dụ: python scripts/convert_audio.py C:\\Users\\me\\recording.mp3")
    else:
        convert_to_standard_wav(sys.argv[1])