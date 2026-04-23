from __future__ import annotations

import os
import sys
import time
import types
import warnings
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from pydub import AudioSegment
from underthesea import sent_tokenize

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

# Đảm bảo import được module khi right-click Run trong PyCharm
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import generate_output_filename, load_config, resolve_path, setup_logger
from text_normalization import detect_language, normalize_text as _vi_normalize_text

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class VietnameseVoiceCloner:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config["paths"]["log_dir"])
        self.device = self.config.get("device", "cpu")
        self.model_dir = (PROJECT_ROOT / self.config["model"]["cache_dir"]).resolve()
        self.model = None
        self._load_model()

    def _download_model_if_needed(self):
        """
        Tải model viXTTS về đúng thư mục models/vixtts nếu chưa có.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)

        required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
        missing = [f for f in required_files if not (self.model_dir / f).exists()]

        if not missing:
            self.logger.info(f"Model đã có sẵn tại: {self.model_dir}")
            return

        self.logger.info("Đang tải model viXTTS từ Hugging Face...")

        snapshot_download(
            repo_id="capleaf/viXTTS",
            local_dir=str(self.model_dir),
        )

        speaker_file = self.model_dir / "speakers_xtts.pth"
        if not speaker_file.exists():
            hf_hub_download(
                repo_id="coqui/XTTS-v2",
                filename="speakers_xtts.pth",
                local_dir=str(self.model_dir),
            )

        self.logger.info("Tải model xong.")

    def _patch_tokenizer_for_vietnamese(self):
        """
        Vá tokenizer XTTS gốc để chấp nhận tiếng Việt ở mức đủ dùng cho inference.
        Đây là workaround vì package TTS gốc không hỗ trợ 'vi' hoàn chỉnh.
        """
        if not hasattr(self.model, "tokenizer"):
            self.logger.warning("Model không có tokenizer để patch.")
            return

        tokenizer = self.model.tokenizer

        # 1) Vá char_limits để tránh KeyError: 'vi'
        if not hasattr(tokenizer, "char_limits") or tokenizer.char_limits is None:
            tokenizer.char_limits = {}

        if "vi" not in tokenizer.char_limits:
            fallback_limit = tokenizer.char_limits.get("en", 250)
            tokenizer.char_limits["vi"] = fallback_limit
            self.logger.info(f"Đã patch tokenizer.char_limits['vi'] = {fallback_limit}")

        # 2) Vá preprocess_text để tránh NotImplementedError: Language 'vi' is not supported
        original_preprocess_text = tokenizer.preprocess_text

        def patched_preprocess_text(this, txt, lang):
            # Dùng nhánh preprocess của en để tokenizer không vỡ.
            # Model vẫn là viXTTS, nên phần acoustic/speaker vẫn chạy theo checkpoint viXTTS.
            mapped_lang = "en" if lang == "vi" else lang
            return original_preprocess_text(txt, mapped_lang)

        tokenizer.preprocess_text = types.MethodType(patched_preprocess_text, tokenizer)
        self.logger.info("Đã patch tokenizer.preprocess_text cho tiếng Việt.")

        # 3) Vá encode nếu cần, để các lời gọi encode(lang='vi') luôn đi qua preprocess đã patch
        original_encode = tokenizer.encode

        def patched_encode(this, txt, lang="en"):
            mapped_lang = "vi" if lang == "vi" else lang
            return original_encode(txt, lang=mapped_lang)

        tokenizer.encode = types.MethodType(patched_encode, tokenizer)
        self.logger.info("Đã patch tokenizer.encode cho tiếng Việt.")

    def _load_model(self):
        self._download_model_if_needed()

        self.logger.info("Đang load viXTTS...")
        start = time.time()

        config = XttsConfig()
        config.load_json(str(self.model_dir / "config.json"))

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=str(self.model_dir), eval=True)

        self._patch_tokenizer_for_vietnamese()

        if self.device == "cuda" and torch.cuda.is_available():
            self.model.cuda()
            self.logger.info("Đang chạy bằng GPU.")
        else:
            self.device = "cpu"
            self.logger.info("Đang chạy bằng CPU.")

        elapsed = time.time() - start
        self.logger.info(f"Load model xong sau {elapsed:.1f}s")

    def _ensure_reference_audio(self, speaker_audio_path: str) -> str:
        """
        Chuẩn hóa file giọng mẫu về WAV mono, sample_rate theo config.
        """
        input_path = resolve_path(speaker_audio_path, PROJECT_ROOT)
        if input_path is None or not Path(input_path).exists():
            raise FileNotFoundError(f"Không tìm thấy file giọng mẫu: {input_path}")

        sample_rate = int(self.config.get("audio", {}).get("sample_rate", 24000))
        converted_dir = Path(self.config["paths"]["reference_voices"]) / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)

        output_wav = converted_dir / f"{Path(input_path).stem}_std.wav"

        try:
            audio = AudioSegment.from_file(str(input_path))
        except Exception as e:
            raise RuntimeError(
                "Không đọc được file audio tham chiếu. "
                "Nếu bạn dùng .m4a hoặc .mp3 thì hãy cài ffmpeg."
            ) from e

        audio = audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
        audio.export(str(output_wav), format="wav")

        duration_sec = len(audio) / 1000.0
        self.logger.info(f"Reference WAV: {output_wav}")
        self.logger.info(f"Duration: {duration_sec:.2f}s")

        if duration_sec < 3:
            self.logger.warning("File giọng mẫu quá ngắn. Nên dùng khoảng 6-15 giây.")
        elif duration_sec > 30:
            self.logger.warning("File giọng mẫu hơi dài. Nên dùng khoảng 6-15 giây.")

        return str(output_wav)

    @staticmethod
    def normalize_vietnamese_text(text: str) -> str:
        """
        Chuẩn hóa văn bản tiếng Việt trước khi đưa vào mô hình TTS.

        Delegate đến text_normalization.normalize_text để thực hiện toàn
        bộ pipeline: bảo vệ code/URL/ID, mở rộng viết tắt, chuyển đổi
        số/ngày/tiền tệ/đơn vị sang chữ tiếng Việt, dọn dấu câu.
        """
        return _vi_normalize_text(text, lang="vi")

    def _split_sentences(self, text: str) -> list[str]:
        """
        Tách câu ở ngoài để không phụ thuộc text splitting nội bộ của XTTS.
        """
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = [text]

        if not sentences:
            sentences = [text]

        sentences = [s.strip() for s in sentences if s and s.strip()]
        return sentences if sentences else [text.strip()]

    def clone(
        self,
        text: str,
        speaker_audio: str,
        language: str = "vi",
        normalize_text: bool = True,
        output_path: str | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Text rỗng.")

        if language != "vi":
            self.logger.warning(f"Model này đang tối ưu cho tiếng Việt, nhưng bạn đang dùng language='{language}'.")

        if output_path is None:
            output_path = generate_output_filename(
                self.config["paths"]["output_dir"],
                prefix="vi_clone",
                ext="wav",
            )
        else:
            output_path = str(resolve_path(output_path, PROJECT_ROOT))

        speaker_wav = self._ensure_reference_audio(speaker_audio)

        norm_cfg = self.config.get("normalization", {})
        if normalize_text and norm_cfg.get("enabled", True):
            if norm_cfg.get("auto_detect_language", False):
                detected = detect_language(text)
                self.logger.info(f"Ngôn ngữ tự động phát hiện: {detected}")
                effective_lang = detected
            else:
                effective_lang = language
            if effective_lang == "vi":
                text = self.normalize_vietnamese_text(text)

        self.logger.info(f"Text: {text}")
        self.logger.info(f"Language: {language}")
        self.logger.info(f"Output: {output_path}")

        self.logger.info("Đang tính speaker embedding...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_wav,
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

        sentences = self._split_sentences(text)
        self.logger.info(f"Tách thành {len(sentences)} câu")

        wav_chunks = []
        start = time.time()

        for i, sentence in enumerate(sentences, start=1):
            self.logger.info(f"[{i}/{len(sentences)}] {sentence}")

            try:
                out = self.model.inference(
                    text=sentence,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.2,
                    length_penalty=1.0,
                    repetition_penalty=5.0,
                    top_k=20,
                    top_p=0.75,
                    enable_text_splitting=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Inference thất bại ở câu {i}: {sentence}\n"
                    f"Nguyên nhân gốc: {e}"
                ) from e

            wav = out.get("wav")
            if wav is None:
                raise RuntimeError(f"Model không trả về 'wav' cho câu: {sentence}")

            wav_chunks.append(torch.tensor(wav, dtype=torch.float32))

        if not wav_chunks:
            raise RuntimeError("Không tạo được audio chunk nào.")

        full_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, full_wav, 24000)

        elapsed = time.time() - start
        self.logger.info(f"Hoàn thành sau {elapsed:.1f}s")
        self.logger.info(f"File output: {output_path}")
        return output_path

def load_text_file(path: str) -> str:
    p = resolve_path(path, PROJECT_ROOT)
    if p is None or not Path(p).exists():
        raise FileNotFoundError(f"Không tìm thấy file text: {p}")

    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    return text.strip()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    SPEAKER_AUDIO = "data/reference_voices/my_voice.wav"

    TEXT_FILE = "data/texts/sample.txt"
    TEXT = load_text_file(TEXT_FILE)

    LANGUAGE = "vi"

    cloner = VietnameseVoiceCloner()
    output = cloner.clone(
        text=TEXT,
        speaker_audio=SPEAKER_AUDIO,
        language=LANGUAGE,
        normalize_text=True,
    )

    print(f"Output file: {output}")