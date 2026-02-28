# CDAC_Chatbot 🤖

A domain-specific AI chatbot for **C-DAC February 2026 admissions**, fine-tuned on Llama 3.2 3B Instruct using QLoRA (Unsloth) and served via a Flask web interface.

---

## 📁 Project Structure

```
chatbot_pt2/
├── app.py                  # Flask web server + chat UI
├── finetune.py             # QLoRA fine-tuning script (Unsloth)
├── dataset-1.json          # Training dataset (C-DAC admissions Q&A)
├── cdac_qlora_adapter/     # Fine-tuned LoRA adapter (generated after training)
├── .gitignore
└── README.md
```

> **Note:** `venv/`, `outputs/`, `unsloth_compiled_cache/`, and model weight files (`.safetensors`, `.bin`, `.pt`, `.gguf`) are excluded from git.

---

## ⚙️ Prerequisites

- **OS:** Windows 10/11 (or Linux)
- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4050 6GB VRAM)
- **Python:** 3.10 or 3.11
- **CUDA Toolkit:** 11.8 or 12.1

---

## 🚀 Setup Instructions

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shawtanya/CDAC_Chatbot.git
cd CDAC_Chatbot
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### Step 3 — Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl transformers datasets flask
```

> ⚠️ On **Windows**, also run:
> ```bash
> pip install triton-windows
> ```

---

## 🏋️ Fine-Tuning the Model

Run the fine-tuning script to generate the `cdac_qlora_adapter/`:

```bash
python finetune.py
```

**What it does:**
- Loads `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (auto-downloaded from Hugging Face)
- Trains for **3 epochs** using QLoRA (r=32) on `dataset-1.json`
- Saves the adapter to `./cdac_qlora_adapter/`

⏱️ **Estimated time:** ~8–10 minutes on RTX 4050 (6GB VRAM)

---

## 🌐 Running the Chatbot

Once fine-tuning is complete, start the Flask server:

```bash
python app.py
```

Then open your browser and visit:

```
http://localhost:5000
```

The chatbot will load your fine-tuned adapter and start answering C-DAC admission queries.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `RuntimeError: Unsloth: No or fused cross entropy` | `XFORMERS_FORCE_DISABLE_TRITON=1` is already set in `finetune.py`. Ensure you're using the latest version from this repo. |
| `CUDA not available` | Ensure NVIDIA drivers and CUDA Toolkit are installed. Run `python -c "import torch; print(torch.cuda.is_available())"` to verify. |
| Out of VRAM | Batch size is already set to 1. Try reducing `MAX_SEQ_LENGTH` in `finetune.py` from 2048 to 1024. |
| Port 5000 already in use | Change `port=5000` to another port (e.g., `5001`) in `app.py`. |

---

## 📌 Notes

- The model **only answers C-DAC February 2026 admission queries** — off-topic questions are rejected by the system prompt.
- Do **not** share or commit the `venv/` folder or model weights to GitHub.
- The `cdac_qlora_adapter/` folder **is included** in the repo so others don't need to retrain.