voice-cloning-project/                    ← Working Directory
│
├── .venv/                                ← PyCharm tự tạo (ẩn)
│
├── configs/
│   └── settings.yaml                     ← Cấu hình chung
│
├── src/                                  ← ⭐ Mark as Sources Root
│   ├── __init__.py
│   ├── clone_voice.py                    ← ▶ Run chính
│   ├── batch_clone.py                    ← Clone hàng loạt
│   └── utils.py                          ← Hàm tiện ích
│
├── scripts/
│   ├── check_gpu.py                      ← ▶ Chạy đầu tiên
│   └── convert_audio.py                  ← Chuyển đổi audio
│
├── data/
│   ├── reference_voices/
│   │   └── my_voice.wav                  ← 🎤 File giọng mẫu
│   └── texts/
│       └── sample.txt                    ← Văn bản clone hàng loạt
│
├── outputs/
│   ├── cloned/                           ← 🔊 Kết quả audio
│   │   └── clone_20260411_143050.wav
│   └── logs/                             ← 📋 Log chạy
│       └── clone_20260411_143050.log
│
├── models/                               ← Model cache (tự tải)
├── notebooks/                            ← Jupyter notebook
├── requirements.txt
├── .gitignore
└── README.md