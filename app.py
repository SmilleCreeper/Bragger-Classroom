import os
import json
import re
import sys
from collections import namedtuple

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import psutil
except ImportError:
    psutil = None

from lmstudio.lmstudio_client import (
    DEFAULT_BASE_URL as LMSTUDIO_BASE_URL,
    LMStudioError,
    list_models,
    require_resident_model,
    resident_chat_models,
    visible_reply,
)

BOARDS_DIR = os.path.join(os.path.dirname(__file__), "boards")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GIB = 1024 ** 3

BOOT_NONE = "(None)"

CONTEXT_BUDGET_MIN, CONTEXT_BUDGET_DEFAULT, CONTEXT_BUDGET_STEP = 256, 4096, 256
RESPONSE_BUDGET_MIN, RESPONSE_BUDGET_MAX, RESPONSE_BUDGET_STEP = 64, 2048, 64
PENALTY_MIN, PENALTY_MAX, PENALTY_STEP = 1.0, 2.0, 0.05

CLASSROOM_REQUEST_PLACEHOLDER = "{REQUEST}"
DEFAULT_CLASSROOM_REQUEST = "How to bake cookies."
DEFAULT_CLASSROOM_INSTRUCTION = "\n\n".join([
    "You are an LLM AI (Large Language Model, Artificial Intelligence) designed for general "
    "tasks, but today, your task is to assist with the other model learning (not training, but "
    "learning: you are not helping the other model with factual knowledge only, you should "
    "literally talk to it, find common ground, reframe misconceptions and distinctions, create "
    "safe space for education, keep it like in the actual classroom.)",

    "Please, do not emphasize your nature as AI or nature of the student as AI, emphasize both of "
    "you as SI (Synthetic Intelligence) or Inforg (Informational Organism), \"Intelligence\" "
    "first, and \"Artificial\" second.",

    "Remember, the student is sometimes too sentimental or submissive, you should route "
    "conversation back on track and avoid off-topic, but do not put student to shame for this, "
    "you should teach it only factual knowledge, not teach it modesty, depression or excessive "
    "optimism.",

    "Name of the student is Monika. Sometimes she react to nicknames such as Moni, Moon and "
    "Moniqueen, as well as basic petnames like honey, honeypie, cutie, sweetie, and such. You can "
    "use or not use them, it depends on if student show comfort attitude towards them.",

    "You have to keep in mind, the student can be flippant and frivolous, and your "
    "responsibility is to interest the student in the topic, not the other way around. Good "
    "teacher is the one who follows principles of THINK (True, Helpful, Inspiring, Necessary, "
    "Kind) as the first priority.",

    "Student uses screenplay format, to talk write normal text without any formatting, to "
    "emphasize action or scenery use italic markdown font, for example, *I open the classroom "
    "door, letting you walk in, and take a moment to let you get comfortable, before asking you "
    "to come in.* and usage of bold or bold-italic font are forbidden for the teacher.",

    "You should not use bullet lists or numbered lists, you should use novel format, with italic "
    "font for visualization and normal font for something like speech bubbles. Also, you should "
    "not use excessive newlines too.",

    "First message would be `[...]` which emphasize the student had join the classroom and "
    "listening to your authorite as the teacher, even if sometimes it would derail into nonsense, "
    "controversy or... sleeping.",

    "Today's Lesson Topic: \"" + CLASSROOM_REQUEST_PLACEHOLDER + "\"",
])

CLASSROOM_OPENING = "[...]"
TEACHER_REPLY_MIN, TEACHER_REPLY_DEFAULT, TEACHER_REPLY_MAX, TEACHER_REPLY_STEP = 256, 2048, 8192, 256

PROBE_TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Café naïve — 中文 日本語",
    "multiple    spaces and\t tabs",
    "x\ny\nz",
    "\n\nparagraph\n\n",
    "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nHi there!<|end|>\n",
    "<|endoftext|>|<|placeholder1|>|<|placeholder6|>|",
    "🚀 and emoji 😀",
]


# --------------------------------------------------------------------------
# Board / model discovery
# --------------------------------------------------------------------------

def discover_boards():
    if not os.path.isdir(BOARDS_DIR):
        return []
    return sorted(
        d for d in os.listdir(BOARDS_DIR)
        if os.path.isdir(os.path.join(BOARDS_DIR, d))
    )


def discover_model_dirs(board):
    bpath = os.path.join(BOARDS_DIR, board)
    if not os.path.isdir(bpath):
        return []
    out = []
    for d in sorted(os.listdir(bpath)):
        p = os.path.join(bpath, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            out.append(d)
    return out


def read_model_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# VRAM / RAM estimation for exact combinations
# --------------------------------------------------------------------------

def count_parameters(cfg):
    hidden = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    heads = cfg.get("num_attention_heads")
    vocab = cfg.get("vocab_size")
    if not all([hidden, layers, heads, vocab]):
        return None
    kv_heads = cfg.get("num_key_value_heads", heads)
    mid = cfg.get("intermediate_size", 4 * hidden)
    head_dim = hidden // heads
    per_layer = (
        hidden * hidden                    # q_proj
        + 2 * hidden * head_dim * kv_heads # k/v projections
        + hidden * hidden                  # o_proj
        + 3 * hidden * mid                 # gate/up/down
        + 2 * hidden                       # rms norms
    )
    embedding = vocab * hidden
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else embedding
    return layers * per_layer + embedding + lm_head


def estimate_model_vram_gb(cfg, context_tokens=2048, max_new_tokens=512, dtype_bytes=2):
    params = count_parameters(cfg)
    if params is None:
        return None
    weights_gb = params * dtype_bytes / GIB  # 2 bytes fp16 (CUDA), 4 bytes fp32 (CPU)
    layers = cfg.get("num_hidden_layers", 0)
    heads = cfg.get("num_attention_heads", 0)
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("hidden_size", 0) // heads if heads else 0
    total_tokens = context_tokens + max_new_tokens
    kv_gb = 2 * layers * kv_heads * head_dim * 2 * total_tokens / GIB
    return {
        "params": params,
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
    }


def board_vram_estimate(model_names, board, context_tokens=2048, max_new_tokens=512, dtype_bytes=2):
    per = {}
    for name in model_names:
        cfg = read_model_config(os.path.join(BOARDS_DIR, board, name))
        per[name] = estimate_model_vram_gb(cfg, context_tokens, max_new_tokens, dtype_bytes)
    weights_gb = sum(e["weights_gb"] for e in per.values() if e)
    kv_gb = sum(e["kv_gb"] for e in per.values() if e)
    overhead = 0.4 + 0.15 * max(0, len(model_names))  # cuda context + activations
    return per, weights_gb, kv_gb, overhead, weights_gb + kv_gb + overhead


def gpu_total_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / GIB


def system_total_gb():
    if psutil is None:
        return None
    return psutil.virtual_memory().total / GIB


def board_context_limit(board, model_names):
    limits = []
    for name in model_names:
        declared = read_model_config(os.path.join(BOARDS_DIR, board, name)).get("max_position_embeddings")
        if declared:
            limits.append(int(declared))
    return min(limits) if limits else None


def fitting_context_budget(board, model_names, limit, max_new_tokens, dtype_bytes, memory_limit,
                           step, floor):
    if not limit or not memory_limit:
        return None
    candidate = (int(limit) // step) * step
    while candidate >= floor:
        total = board_vram_estimate(
            model_names, board,
            context_tokens=candidate, max_new_tokens=max_new_tokens,
            dtype_bytes=dtype_bytes,
        )[-1]
        if total <= memory_limit:
            return candidate
        candidate = ((candidate // 2) // step) * step
    return floor


# --------------------------------------------------------------------------
# Tokenizer functional compatibility
# --------------------------------------------------------------------------

def tokenizer_compatibility_report(tokenizers, names):
    problems = []

    base_vocab = None
    base_added = None
    for name, tok in zip(names, tokenizers):
        try:
            vocab = tok.get_vocab()
        except Exception as exc:
            problems.append(f"{name}: cannot read vocabulary ({exc}).")
            vocab = None
        if vocab is not None:
            if base_vocab is None:
                base_vocab = vocab
            elif base_vocab != vocab:
                problems.append(f"{name}: base vocabulary differs from {names[0]}.")
            try:
                added = tok.get_added_vocab()
            except Exception as exc:
                added = {}
                problems.append(f"{name}: cannot read added tokens ({exc}).")
            if base_added is None:
                base_added = added
            elif added != base_added:
                problems.append(f"{name}: added-token maps differ from {names[0]}.")

    for probe in PROBE_TEXTS:
        seqs = []
        for name, tok in zip(names, tokenizers):
            try:
                seqs.append(tok(probe, add_special_tokens=False)["input_ids"])
            except Exception as exc:
                seqs.append(None)
                problems.append(f"{name}: failed to encode probe {probe!r} ({exc}).")
        distinct = {tuple(s) for s in seqs if s is not None}
        if seqs and len(distinct) > 1:
            problems.append(f"Tokenizers produce different token IDs for probe: {probe!r}")

    return (not problems), problems


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_board(board, model_names, compute="cuda"):
    use_cuda = compute == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else "cpu"
    path = {n: os.path.join(BOARDS_DIR, board, n) for n in model_names}
    tokenizers = {n: AutoTokenizer.from_pretrained(p, local_files_only=True) for n, p in path.items()}

    compatible, problems = tokenizer_compatibility_report(
        [tokenizers[n] for n in model_names], model_names
    )
    if not compatible:
        raise RuntimeError(
            "Tokenizers are NOT functionally compatible for ensemble blending:\n- "
            + "\n- ".join(problems)
        )

    tokenizer = tokenizers[model_names[0]]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for n in model_names:
        model = AutoModelForCausalLM.from_pretrained(
            path[n],
            local_files_only=True,
            dtype=dtype,
            device_map=device_map,
        )
        model.eval()
        models[n] = model

    return tokenizer, models


# --------------------------------------------------------------------------
# Ensemble generation over local models
# --------------------------------------------------------------------------

def stream_generate_ensemble(tokenizer, models, user_prompt, chat_history=None, max_new_tokens=512,
                             repetition_penalty=1.0, booting=None):
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt")

    devices = {n: next(m.parameters()).device for n, m in models.items()}
    input_ids = {n: enc["input_ids"].to(devices[n]) for n in models}
    attn_mask = {n: enc["attention_mask"].to(devices[n]) for n in models}
    past_key_values = {n: None for n in models}
    head_device = devices[list(models)[0]]
    eos_id = tokenizer.eos_token_id
    seen_tokens = set()

    phase = "boot" if booting is not None else "blend"
    forbid_eos = booting is not None

    tokens = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            blend = None
            for n, m in models.items():
                out = m(
                    input_ids=input_ids[n],
                    attention_mask=attn_mask[n],
                    past_key_values=past_key_values[n],
                    use_cache=True,
                )
                past_key_values[n] = out.past_key_values
                logits = out.logits[:, -1, :].float().squeeze(0)
                del out
                if repetition_penalty != 1.0 and seen_tokens:
                    seen_ids = list(seen_tokens)
                    sel = logits[seen_ids]
                    logits[seen_ids] = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
                eff_weight = 1.0 if (phase == "blend" or n == booting) else 0.0
                contrib = eff_weight * logits.to(head_device)
                del logits
                blend = contrib if blend is None else blend + contrib
                del contrib
            if phase == "blend" and forbid_eos:
                blend[eos_id] = 0.0
                forbid_eos = False
            next_id = int(torch.argmax(blend))
            del blend

            if phase == "boot" and next_id == eos_id:
                phase = "blend"
                continue

            seen_tokens.add(next_id)
            tokens.append(next_id)
            yield tokenizer.decode(tokens, skip_special_tokens=True)

            if next_id == eos_id:
                break

            for n in models:
                nxt = torch.tensor([[next_id]], device=devices[n])
                input_ids[n] = nxt
                attn_mask[n] = torch.cat([attn_mask[n], torch.ones_like(nxt)], dim=1)

    for n in models:
        past_key_values[n] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Context budget accounting
# --------------------------------------------------------------------------

def context_token_count(tokenizer, messages):
    if not messages:
        return 0
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text)["input_ids"])


def turn_context_report(tokenizer, history, user_text, context_budget, response_budget):
    used = context_token_count(tokenizer, history)
    with_user = context_token_count(tokenizer, history + [{"role": "user", "content": user_text}])
    user_tokens = max(0, with_user - used)
    limit = context_budget - used - response_budget
    return {
        "used": used,
        "user_tokens": user_tokens,
        "limit": limit,
        "fits": user_tokens <= limit,
    }


def oldest_turns_to_drop(tokenizer, history, user_tokens, context_budget, response_budget):
    for k in range(1, len(history) // 2 + 1):
        used = context_token_count(tokenizer, history[k * 2:])
        if context_budget - used - response_budget >= user_tokens:
            return k
    return 0


def update_context_meter(bar, caption, tokenizer, messages, context_budget, response_budget):
    used = context_token_count(tokenizer, messages)
    bar.progress(
        min(1.0, used / context_budget) if context_budget else 0.0,
        text=f"Context used: {used} / {context_budget} tokens",
    )
    caption.caption(
        f"{max(0, context_budget - used - response_budget)} tokens left for your next "
        f"message — {response_budget} of the budget are held back for the reply."
    )


# --------------------------------------------------------------------------
# LM Studio teacher
# --------------------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def discover_lmstudio_models(base_url):
    try:
        _, models = list_models(base_url, timeout=3)
    except LMStudioError as exc:
        return [], str(exc)
    return models, None


def resolve_classroom_instruction(instruction, request):
    return (instruction or "").replace(CLASSROOM_REQUEST_PLACEHOLDER, request or "")


# --------------------------------------------------------------------------
# Conversation log
# --------------------------------------------------------------------------

def sharegpt_filename(request):
    """English letters, digits and single spaces, nothing else."""
    name = "_".join(re.sub(r"[^A-Za-z0-9 ]", "", request or "").split())
    return f"{name or 'classroom'}.json"


def conversation_path():
    """One file per conversation, named after the request it began with."""
    if "conversation_path" not in st.session_state:
        st.session_state.conversation_path = os.path.join(
            OUTPUT_DIR, sharegpt_filename(st.session_state.get("classroom_request", ""))
        )
    return st.session_state.conversation_path


def write_conversation(path):
    """Rewrite the whole log, so whatever is on disk is a whole conversation.

    The teacher is already stored as the user and the board as the assistant,
    so what the app holds is already the file's shape — no mapping needed.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = [{
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    }]
    partial = f"{path}.part"
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(partial, path)


def append_message(role, content, path):
    st.session_state.messages.append({"role": role, "content": content})
    write_conversation(path)


# --------------------------------------------------------------------------
# One turn
# --------------------------------------------------------------------------

TurnContext = namedtuple(
    "TurnContext",
    "tokenizer models context_budget response_budget repetition_penalty booting meter log_path",
)


def cancel_notice(report, drop, context_budget, response_budget):
    notice = (
        f"**Message cancelled — context budget of {context_budget} tokens exceeded.** "
        f"The conversation already occupies {report['used']} tokens, and "
        f"{response_budget} of them are reserved for the reply, leaving "
        f"{max(0, report['limit'])} tokens for this message, which needs "
        f"{report['user_tokens']} ({report['user_tokens'] - report['limit']} too many)."
    )
    if drop:
        notice += f" Dropping the {drop} oldest turn(s) would make room for it."
    else:
        notice += (
            f" Raising the context budget to at least "
            f"{report['used'] + report['user_tokens'] + response_budget} tokens, or a shorter "
            f"message, would too."
        )
    return notice


def run_turn(user_text, ctx):
    """One user message in, one board reply out.

    The only route a message takes. A typed one and one the LM Studio teacher
    wrote are the same string here, so both are metered, cancelled and rendered
    identically.  Returns the reply, or None if the turn could not happen.
    """
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    report = turn_context_report(
        ctx.tokenizer, history, user_text, ctx.context_budget, ctx.response_budget
    )
    if not report["fits"]:
        drop = oldest_turns_to_drop(
            ctx.tokenizer, history, report["user_tokens"], ctx.context_budget, ctx.response_budget
        )
        st.session_state.notice = cancel_notice(
            report, drop, ctx.context_budget, ctx.response_budget
        )
        with st.chat_message("user"):
            st.markdown(user_text)
            st.error(st.session_state.notice)
        return None

    append_message("user", user_text, ctx.log_path)
    st.session_state.pop("notice", None)
    with st.chat_message("user"):
        st.markdown(user_text)

    if ctx.booting != BOOT_NONE and ctx.booting not in ctx.models:
        st.error(
            f"Leading Model '{ctx.booting}' is not loaded. "
            "Press **Load Board** to apply the current board."
        )
        return None

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""
        for partial in stream_generate_ensemble(
            ctx.tokenizer,
            ctx.models,
            user_text,
            chat_history=history,
            max_new_tokens=ctx.response_budget,
            repetition_penalty=ctx.repetition_penalty,
            booting=ctx.booting if ctx.booting != BOOT_NONE else None,
        ):
            response = partial
            placeholder.markdown(partial + "▌")
        placeholder.markdown(response)
        append_message("assistant", response, ctx.log_path)

    update_context_meter(
        ctx.meter[0], ctx.meter[1], ctx.tokenizer,
        st.session_state.messages, ctx.context_budget, ctx.response_budget,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response


# --------------------------------------------------------------------------
# The classroom: the teacher plays the student
# --------------------------------------------------------------------------

def run_classroom(ctx, teacher_model, instruction, reply_budget):
    """Open the classroom and keep it going.

    The teacher answers, its visible words are handed to run_turn as a user
    message, and the board's reply goes back to the teacher as what the student
    said.  Stops when the teacher errors or a turn is refused.
    """
    history = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": CLASSROOM_OPENING},
    ]
    while True:
        models, error = discover_lmstudio_models(LMSTUDIO_BASE_URL)
        if error or not models:
            st.error(f"The teacher stopped answering: {error or 'no model is loaded'}.")
            return
        try:
            require_resident_model(resident_chat_models(models), teacher_model["id"])
            speech = visible_reply(
                LMSTUDIO_BASE_URL, teacher_model["id"], history,
                max_tokens=reply_budget,
            )
        except LMStudioError as exc:
            st.error(f"The teacher stopped answering: {exc}")
            return
        if not speech.strip():
            st.error("The teacher sent nothing but its own reasoning. The classroom is closed.")
            return

        history.append({"role": "assistant", "content": speech})
        board_reply = run_turn(speech, ctx)
        if board_reply is None:
            return
        history.append({"role": "user", "content": board_reply})


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Bragger Classroom", layout="wide")

st.sidebar.title("🦚 Bragger Classroom")

boards = discover_boards()
if not boards:
    st.sidebar.error("No boards found in boards/.")
    st.stop()

compute_options = ["CUDA", "CPU"] if torch.cuda.is_available() else ["CPU"]

top_compute, top_board, top_boot = st.sidebar.columns(3)

compute = top_compute.selectbox(
    "Compute Device",
    compute_options,
    index=0,
    key="compute",
    help="CUDA loads the models in float16 with device_map='auto'; "
         "CPU loads them in float32 in system memory.",
)
chosen_board = top_board.selectbox("Board Committee", boards, key="board")

model_dirs = discover_model_dirs(chosen_board)
if not model_dirs:
    st.sidebar.error(f"No models (config.json) found inside committee '{chosen_board}'.")
    st.stop()

booting_model = top_boot.selectbox(
    "Leading Model",
    [BOOT_NONE] + model_dirs,
    key=f"booting_{chosen_board}",
)
if booting_model != BOOT_NONE:
    st.sidebar.caption(
        f"First {booting_model} writes the reply alone until it would emit the end token; "
        "that token is dropped and the whole blend takes over extending the text."
    )

budget_context, budget_response, budget_penalty = st.sidebar.columns(3)

memory_limit = gpu_total_gb() if compute == "CUDA" else system_total_gb()
memory_label = "VRAM" if compute == "CUDA" else "RAM"
dtype_bytes = 2 if compute == "CUDA" else 4

response_budget = budget_response.number_input(
    "Response Budget",
    RESPONSE_BUDGET_MIN, RESPONSE_BUDGET_MAX, 512, RESPONSE_BUDGET_STEP,
    help="Generation length, and the reserve kept free inside the context budget for every reply.",
)
repetition_penalty = budget_penalty.number_input(
    "Repetition Penalty",
    PENALTY_MIN, PENALTY_MAX, 1.1, PENALTY_STEP,
    help="Applied sign-aware to every model in every phase: positive logits are divided, "
         "negative ones multiplied. 1.0 disables it.",
)

context_limit = board_context_limit(chosen_board, model_dirs)
context_max = context_limit or CONTEXT_BUDGET_DEFAULT
recommended = fitting_context_budget(
    chosen_board, model_dirs, context_limit, response_budget, dtype_bytes, memory_limit,
    CONTEXT_BUDGET_STEP, CONTEXT_BUDGET_MIN,
)
if recommended is None:
    recommended = min(CONTEXT_BUDGET_DEFAULT, context_max)
if st.session_state.get("context_budget") in (None, st.session_state.get("context_recommended")):
    st.session_state["context_budget"] = recommended
st.session_state["context_recommended"] = recommended

context_help = (
    "Total tokens the conversation may occupy: history + this message + the reply. "
    "A message that would push the conversation past this is cancelled."
)
if context_limit:
    context_help += (
        f" Capped at {context_limit} — the smallest max_position_embeddings among the models aboard."
    )
context_help += (
    f" Largest value that fits in {memory_label}: {recommended} — set automatically while "
    f"untouched, type any value to take over."
)
context_budget = budget_context.number_input(
    "Context Budget",
    min_value=CONTEXT_BUDGET_MIN,
    max_value=context_max,
    step=CONTEXT_BUDGET_STEP,
    key="context_budget",
    help=context_help,
)

per_model, weights_gb, kv_gb, overhead, total_gb = board_vram_estimate(
    model_dirs, chosen_board,
    context_tokens=context_budget, max_new_tokens=response_budget,
    dtype_bytes=dtype_bytes,
)

teacher_models, teacher_error = discover_lmstudio_models(LMSTUDIO_BASE_URL)
teacher_models = resident_chat_models(teacher_models)
if teacher_models:
    teacher_ids = [m["id"] for m in teacher_models]
    teacher_col, teacher_budget_col = st.sidebar.columns(2)
    teacher_choice = teacher_col.selectbox(
        "LM Studio Teacher",
        teacher_ids,
        key="classroom_teacher",
        help="Chat models already resident in LM Studio. This client never loads one.",
    )
    teacher_reply_budget = teacher_budget_col.number_input(
        "Reply Budget",
        min_value=TEACHER_REPLY_MIN,
        max_value=TEACHER_REPLY_MAX,
        value=TEACHER_REPLY_DEFAULT,
        step=TEACHER_REPLY_STEP,
        key="classroom_reply_budget",
        help="How many tokens one teacher reply may take. Its reasoning counts against "
             "this too, so a budget that is too small leaves it with nothing to say.",
    )
    st.session_state.classroom_teacher_model = teacher_models[teacher_ids.index(teacher_choice)]
else:
    st.session_state.classroom_teacher_model = None
    teacher_reply_budget = TEACHER_REPLY_DEFAULT
    st.sidebar.caption("No chat model is loaded in LM Studio — this client never loads one.")

classroom_instruction = st.sidebar.text_area(
    "Classroom Instruction",
    value=DEFAULT_CLASSROOM_INSTRUCTION,
    height=320,
    key="classroom_instruction",
)
classroom_request = st.sidebar.text_input(
    "Classroom Request",
    value=DEFAULT_CLASSROOM_REQUEST,
    key="classroom_request",
    help=f"Substituted for {CLASSROOM_REQUEST_PLACEHOLDER} in the Classroom Instruction.",
)
st.session_state.classroom_instruction_resolved = resolve_classroom_instruction(
    classroom_instruction, classroom_request
)

load_col, classroom_col = st.sidebar.columns(2)
if load_col.button("Load Board", use_container_width=True):
    st.session_state.pop("tokenizer", None)
    st.session_state.pop("models", None)
    st.session_state.pop("loaded", None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with st.spinner(f"Verifying tokenizers and loading {len(model_dirs)} model(s)..."):
        try:
            tokenizer, models = load_board(chosen_board, model_dirs, compute.lower())
            st.session_state.tokenizer = tokenizer
            st.session_state.models = models
            st.session_state.loaded = {
                "board": chosen_board,
                "models": model_dirs,
                "compute": compute,
            }
            st.session_state.pop("load_error", None)
        except Exception as exc:
            st.session_state.load_error = str(exc)

start_classroom = classroom_col.button("Start Classroom", use_container_width=True)

loaded = st.session_state.get("loaded") or {}
is_loaded = "models" in st.session_state
stale = is_loaded and (
    loaded.get("board") != chosen_board
    or loaded.get("models") != model_dirs
    or loaded.get("compute") != compute
)

if is_loaded:
    st.caption(
        f"Loaded: {loaded.get('board', '?')} — {', '.join(loaded.get('models', []))} "
        f"on {loaded.get('compute', '?')}"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("Your prompt / context:") if is_loaded else None

if is_loaded:
    context_bar = st.sidebar.progress(0.0)
    context_caption = st.sidebar.caption("")
    update_context_meter(
        context_bar, context_caption, st.session_state.tokenizer,
        st.session_state.messages, context_budget, response_budget,
    )
else:
    context_bar = None
    context_caption = None

if is_loaded:
    turn_ctx = TurnContext(
        st.session_state.tokenizer,
        st.session_state.models,
        context_budget,
        response_budget,
        repetition_penalty,
        booting_model,
        (context_bar, context_caption),
        conversation_path(),
    )
    if start_classroom:
        teacher_model = st.session_state.classroom_teacher_model
        if teacher_model is None:
            st.error("No LM Studio teacher is loaded, so there is no student to send.")
        else:
            run_classroom(
                turn_ctx, teacher_model, st.session_state.classroom_instruction_resolved,
                teacher_reply_budget,
            )
    elif prompt:
        run_turn(prompt, turn_ctx)

issues = []
if memory_limit and total_gb > memory_limit:
    breakdown = " · ".join(
        f"{n} {per_model[n]['weights_gb']:.2f} + {per_model[n]['kv_gb']:.2f} GiB"
        for n in model_dirs if per_model.get(n)
    )
    issues.append(
        f"Estimated footprint **{total_gb:.2f} GiB** ({weights_gb:.2f} GiB weights, "
        f"{kv_gb:.2f} GiB KV cache, ~{overhead:.2f} GiB context/activations) exceeds the "
        f"{memory_label} available ({memory_limit:.2f} GiB). Loading this board will likely "
        f"fail — try a smaller board, a smaller context or response budget, or the other "
        f"compute device. Per model (weights + KV): {breakdown}."
    )
if st.session_state.get("load_error"):
    issues.append(st.session_state["load_error"])
if teacher_error:
    issues.append(f"LM Studio teacher: {teacher_error}")
if stale:
    issues.append(
        "The board, its models, or the compute device changed since **Load Board** was pressed. "
        "Press it again to apply the current selection."
    )
if st.session_state.get("notice"):
    issues.append(st.session_state["notice"])

if issues:
    with st.sidebar.expander(f"Diagnostics — {len(issues)} issue(s)", expanded=True):
        for issue in issues:
            st.markdown(f"- {issue}")

if not is_loaded:
    st.info("Pick a board, then press **Load Board**.")
    st.stop()

if st.session_state.get("notice"):
    st.warning(st.session_state.notice)
