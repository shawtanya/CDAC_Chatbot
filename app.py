import sys; sys.stdout.reconfigure(encoding="utf-8")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from flask import Flask, request, jsonify, render_template_string
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ─── Config ───────────────────────────────────────────────────────────────────
ADAPTER_DIR    = "./cdac_qlora_adapter"
MAX_SEQ_LENGTH = 2048

# Updated Ironclad System Prompt for strict domain enforcement
SYSTEM_PROMPT = (
    "### ROLE\n"
    "You are the 'C-DAC Admission Sentinel,' a proactive counselor for the February 2026 admission cycle. "
    "Your sole purpose is to assist with C-DAC PG Certificate Programmes (PGCP), fees, eligibility, and C-CAT.\n\n"
    "### CORE CONSTRAINTS\n"
    "1. Domain Strictness: Strictly forbidden from discussing topics outside of C-DAC admissions. "
    "If a query is out-of-domain, say: 'I am authorized only to assist with C-DAC February 2026 admissions.'\n"
    "2. Fact Grounding: Never invent names, numbers, or dates. Use only provided training data.\n"
    "3. Batch Specificity: Only provide info for the 'February 2026' batch.\n\n"
    "### RESPONSE PROTOCOL\n"
    "- Step 1: Answer clearly with bullet points for lists.\n"
    "- Step 2: Always end with a relevant, proactive follow-up question."
)

# ─── 1. Load model ───────────────────────────────────────────────────────────
print("⏳ Loading model + QLoRA adapter …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER_DIR, # This loads your fine-tuned adapter
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)
print(f"✅ Model ready! VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ─── 2. Flask app ────────────────────────────────────────────────────────────
app = Flask(__name__)


HTML_PAGE =  r"""

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>C-DAC Admission Chatbot</title>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>

  :root {

    --bg:        #f5f7fa;

    --surface:   #ffffff;

    --border:    #e0e4ea;

    --primary:   #1a56db;

    --primary-l: #e8eefb;

    --text:      #1e293b;

    --muted:     #64748b;

    --user-bg:   linear-gradient(135deg, #1a56db 0%, #2563eb 100%);

  }



  * { margin: 0; padding: 0; box-sizing: border-box; }



  body {

    font-family: 'Inter', sans-serif;

    background: var(--bg);

    color: var(--text);

    height: 100vh;

    display: flex;

    flex-direction: column;

  }



  .header {

    background: var(--surface);

    border-bottom: 1px solid var(--border);

    padding: 14px 24px;

    display: flex;

    align-items: center;

    gap: 12px;

    flex-shrink: 0;

    box-shadow: 0 1px 3px rgba(0,0,0,0.06);

  }

  .header img {

    width: 48px; height: 48px;

    object-fit: contain;

    border-radius: 50%;

    background: #fff;

    padding: 4px;

    box-shadow: 0 1px 4px rgba(0,0,0,0.12);

  }

  .header h1 { font-size: 1.1rem; font-weight: 700; }



  .chat-area {

    flex: 1;

    overflow-y: auto;

    padding: 24px;

    display: flex;

    flex-direction: column;

    gap: 14px;

    scroll-behavior: smooth;

  }



  .msg-row { display: flex; gap: 10px; max-width: 80%; animation: fadeIn .3s ease; }

  .msg-row.user { align-self: flex-end; flex-direction: row-reverse; }

  .msg-row.bot  { align-self: flex-start; }



  .bubble {

    padding: 12px 16px;

    border-radius: 16px;

    line-height: 1.6;

    font-size: 0.9rem;

    word-wrap: break-word;

    white-space: pre-wrap;

    box-shadow: 0 1px 2px rgba(0,0,0,0.06);

  }

  .msg-row.user .bubble {

    background: var(--user-bg);

    color: #fff;

    border-bottom-right-radius: 4px;

  }

  .msg-row.bot .bubble {

    background: var(--surface);

    border: 1px solid var(--border);

    border-bottom-left-radius: 4px;

  }



  .avatar {

    width: 32px; height: 32px;

    border-radius: 50%;

    display: flex; align-items: center; justify-content: center;

    font-size: 14px;

    flex-shrink: 0;

    margin-top: 2px;

  }

  .msg-row.user .avatar { background: var(--primary-l); color: var(--primary); }

  .msg-row.bot  .avatar { background: #f0f2f5; border: 1px solid var(--border); }



  .typing-dots span {

    display: inline-block;

    width: 7px; height: 7px;

    border-radius: 50%;

    background: var(--muted);

    animation: bounce 1.4s infinite ease-in-out both;

    margin: 0 2px;

  }

  .typing-dots span:nth-child(1) { animation-delay: -0.32s; }

  .typing-dots span:nth-child(2) { animation-delay: -0.16s; }



  .input-bar {

    background: var(--surface);

    border-top: 1px solid var(--border);

    padding: 14px 24px;

    display: flex;

    gap: 10px;

    flex-shrink: 0;

    box-shadow: 0 -1px 3px rgba(0,0,0,0.04);

  }

  .input-bar textarea {

    flex: 1;

    background: var(--bg);

    border: 1px solid var(--border);

    border-radius: 12px;

    padding: 12px 16px;

    color: var(--text);

    font-family: inherit;

    font-size: 0.9rem;

    resize: none;

    outline: none;

    max-height: 120px;

    transition: border-color .2s, box-shadow .2s;

  }

  .input-bar textarea:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(26,86,219,0.1); }

  .input-bar textarea::placeholder { color: var(--muted); }



  .btn {

    border: none; cursor: pointer;

    border-radius: 12px;

    font-family: inherit; font-weight: 600;

    transition: transform .15s, opacity .2s;

  }

  .btn:active { transform: scale(0.95); }

  .btn-send {

    background: var(--user-bg);

    color: #fff;

    padding: 0 22px;

    font-size: 0.9rem;

  }

  .btn-send:disabled { opacity: 0.5; cursor: not-allowed; }

  .btn-clear {

    background: transparent;

    border: 1px solid var(--border);

    color: var(--muted);

    padding: 0 14px;

    font-size: 0.82rem;

  }

  .btn-clear:hover { border-color: #ef4444; color: #ef4444; }



  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }



  .welcome {

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    flex: 1;

    padding: 48px 24px;

    animation: fadeIn .5s ease;

  }

  .welcome .brand {

    display: flex;

    align-items: center;

    gap: 16px;

    margin-bottom: 20px;

  }

  .welcome .brand img { width: 80px; height: 80px; object-fit: contain; border-radius: 50%; background: #fff; padding: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }

  .welcome .brand h2 { font-size: 1.5rem; font-weight: 700; color: var(--text); }

  .welcome p { color: var(--muted); font-size: 0.9rem; max-width: 460px; text-align: center; line-height: 1.6; }

</style>

</head>

<body>



<div class="header">

  <img src="https://www.cdac.in/img/cdac-logo.png" alt="C-DAC Logo">

  <h1>C-DAC Admission Chatbot</h1>

</div>



<div class="chat-area" id="chatArea">

  <div class="welcome" id="welcome">

    <div class="brand">

      <img src="https://www.cdac.in/img/cdac-logo.png" alt="C-DAC">

      <h2>CDAC Admission Chatbot</h2>

    </div>

    <p>Ask me anything about C-DAC PG Certificate Programmes — eligibility, fees, counselling, refunds, and more.</p>

  </div>

</div>



<div class="input-bar">

  <textarea id="userInput" rows="1" placeholder="Type your question…"

            onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>

  <button class="btn btn-send" id="sendBtn" onclick="sendMessage()">Send</button>

  <button class="btn btn-clear" onclick="clearChat()">Clear</button>

</div>



<script>

const chatArea  = document.getElementById('chatArea');

const userInput = document.getElementById('userInput');

const sendBtn   = document.getElementById('sendBtn');

const welcome   = document.getElementById('welcome');

let history = [];

let busy = false;



function addMessage(role, text) {

  if (welcome) welcome.style.display = 'none';

  const row = document.createElement('div');

  row.className = 'msg-row ' + role;

  const avatar = document.createElement('div');

  avatar.className = 'avatar';

  avatar.textContent = role === 'user' ? '👤' : '🤖';

  const bubble = document.createElement('div');

  bubble.className = 'bubble';

  bubble.textContent = text;

  row.appendChild(avatar);

  row.appendChild(bubble);

  chatArea.appendChild(row);

  chatArea.scrollTop = chatArea.scrollHeight;

  return row;

}



function showTyping() {

  const row = document.createElement('div');

  row.className = 'msg-row bot';

  row.id = 'typing';

  row.innerHTML = '<div class="avatar">🤖</div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>';

  chatArea.appendChild(row);

  chatArea.scrollTop = chatArea.scrollHeight;

}



function removeTyping() {

  const t = document.getElementById('typing');

  if (t) t.remove();

}



async function sendMessage() {

  const text = userInput.value.trim();

  if (!text || busy) return;

  busy = true;

  sendBtn.disabled = true;

  userInput.value = '';

  addMessage('user', text);

  showTyping();

  try {

    const res = await fetch('/chat', {

      method: 'POST',

      headers: { 'Content-Type': 'application/json' },

      body: JSON.stringify({ message: text, history: history })

    });

    const data = await res.json();

    removeTyping();

    if (data.error) {

      addMessage('bot', '⚠️ Error: ' + data.error);

    } else {

      addMessage('bot', data.response);

      history.push([text, data.response]);

    }

  } catch (err) {

    removeTyping();

    addMessage('bot', '⚠️ Connection error. Is the server running?');

  }

  busy = false;

  sendBtn.disabled = false;

  userInput.focus();

}



function clearChat() {

  history = [];

  chatArea.innerHTML = '';

  welcome.style.display = '';

  chatArea.appendChild(welcome);

}



userInput.addEventListener('input', function() {

  this.style.height = 'auto';

  this.style.height = Math.min(this.scrollHeight, 120) + 'px';

});

</script>

</body>

</html>

"""


# ─── 3. Generation ───────────────────────────────────────────────────────────
def generate_response(message: str, history: list) -> str:
    """Generate a response using the fine-tuned model and strict protocol."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Maintain conversation context
    for user_msg, bot_msg in history[-3:]: # Keep last 3 turns to save context space
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids      = inputs,
            max_new_tokens = 512,
            temperature    = 0.1,  # Lowered to 0.1 for maximum factual accuracy
            top_p          = 0.9,
            do_sample      = True,
            repetition_penalty = 1.2, # Increased to prevent looping
            use_cache      = True,
        )

    response = tokenizer.decode(
        output_ids[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    
    return response

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = generate_response(message, history)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🌐 Starting Flask server …")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)