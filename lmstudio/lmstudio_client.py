import os
import json

import requests

DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234").rstrip("/")
EMBEDDING_TYPES = {"embedding", "embeddings"}


class LMStudioError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Model discovery
# --------------------------------------------------------------------------

def _normalize(model_id, model_type, state, display_name=None, architecture=None,
               quantization=None, params=None, size_bytes=None, max_context=None,
               loaded_context=None, vision=None):
    return {
        "id": model_id,
        "type": (model_type or "unknown").lower(),
        "state": (state or "unknown").lower(),
        "display_name": display_name or model_id,
        "architecture": architecture,
        "quantization": quantization,
        "params": params,
        "size_gb": round(size_bytes / 1024 ** 3, 2) if size_bytes else None,
        "max_context": max_context,
        "loaded_context": loaded_context,
        "vision": vision,
    }


def _from_v1(payload):
    models = []
    for entry in payload.get("models", []):
        instances = entry.get("loaded_instances") or []
        loaded_context = next(
            (i.get("config", {}).get("context_length") for i in instances if i.get("config")), None
        )
        quantization = entry.get("quantization") or {}
        models.append(_normalize(
            entry.get("key"),
            entry.get("type"),
            "loaded" if instances else "not-loaded",
            display_name=entry.get("display_name"),
            architecture=entry.get("architecture"),
            quantization=quantization.get("name") if isinstance(quantization, dict) else quantization,
            params=entry.get("params_string"),
            size_bytes=entry.get("size_bytes"),
            max_context=entry.get("max_context_length"),
            loaded_context=loaded_context,
            vision=(entry.get("capabilities") or {}).get("vision"),
        ))
    return models


def _from_v0(payload):
    models = []
    for entry in payload.get("data", []):
        loaded = (entry.get("state") or "").lower() == "loaded"
        models.append(_normalize(
            entry.get("id"),
            entry.get("type"),
            "loaded" if loaded else "not-loaded",
            display_name=entry.get("publisher"),
            architecture=entry.get("arch"),
            quantization=entry.get("quantization"),
            max_context=entry.get("max_context_length"),
            loaded_context=entry.get("loaded_context_length") if loaded else None,
            vision=entry.get("type") == "vlm",
        ))
    return models


def _from_openai(payload):
    return [_normalize(entry.get("id"), None, None) for entry in payload.get("data", [])]


def list_models(base_url=DEFAULT_BASE_URL, timeout=10):
    problems = []
    for path, extract in (
        ("/api/v1/models", _from_v1),
        ("/api/v0/models", _from_v0),
        ("/v1/models", _from_openai),
    ):
        try:
            response = requests.get(base_url + path, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            problems.append(f"{path}: {exc}")
            continue
        return path, extract(response.json())
    raise LMStudioError(
        f"Cannot reach an LM Studio model list at {base_url}.\n"
        "Start the local server from the Developer tab and check the port.\n- "
        + "\n- ".join(problems)
    )


# --------------------------------------------------------------------------
# Model selection — resident models only
# --------------------------------------------------------------------------

def is_resident_chat_model(model):
    return model["state"] == "loaded" and model["type"] not in EMBEDDING_TYPES


def resident_chat_models(models):
    return [m for m in models if is_resident_chat_model(m)]


def first_resident_model(models):
    for model in models:
        if is_resident_chat_model(model):
            return model
    raise LMStudioError(
        "No chat model is currently loaded on the server. Start the local server "
        "from LM Studio's Developer tab and load a chat model there — this client "
        "never loads a model itself."
    )


def require_resident_model(models, model_id):
    for model in models:
        if model["id"] == model_id:
            if not is_resident_chat_model(model):
                raise LMStudioError(
                    f"'{model_id}' is present but not loaded (state={model['state']!r}, "
                    f"type={model['type']!r}). Load it in LM Studio first — this client "
                    f"never loads a model itself."
                )
            return model
    known = ", ".join(m["id"] for m in models) or "none"
    raise LMStudioError(f"'{model_id}' is not on the server. Known models: {known}")


def format_model(model):
    bits = [model["id"], f"[{model['type']}]", model["state"]]
    if model["quantization"]:
        bits.append(str(model["quantization"]))
    if model["params"]:
        bits.append(str(model["params"]))
    if model["size_gb"]:
        bits.append(f"{model['size_gb']} GiB")
    if model["architecture"]:
        bits.append(str(model["architecture"]))
    if model["loaded_context"]:
        bits.append(f"ctx {model['loaded_context']}")
    elif model["max_context"]:
        bits.append(f"max ctx {model['max_context']}")
    if model["vision"]:
        bits.append("vision")
    return "  ".join(bits)


# --------------------------------------------------------------------------
# Chat completions
# --------------------------------------------------------------------------

def chat(base_url, model, messages, max_tokens=256, temperature=0.7, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        response = requests.post(base_url + "/v1/chat/completions", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LMStudioError(f"Chat request failed: {exc}") from exc
    if response.status_code != 200:
        raise LMStudioError(
            f"HTTP {response.status_code} from /v1/chat/completions: {response.text[:400]}"
        )
    message = response.json()["choices"][0]["message"]
    return message.get("content") or "", message.get("reasoning_content") or ""


def chat_stream(base_url, model, messages, max_tokens=256, temperature=0.7, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    try:
        response = requests.post(
            base_url + "/v1/chat/completions", json=payload, stream=True, timeout=timeout
        )
    except requests.RequestException as exc:
        raise LMStudioError(f"Chat request failed: {exc}") from exc
    with response:
        if response.status_code != 200:
            raise LMStudioError(
                f"HTTP {response.status_code} from /v1/chat/completions: {response.text[:400]}"
            )
        for raw in response.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            yield delta.get("content") or "", delta.get("reasoning_content") or ""


def visible_reply(base_url, model, messages, **kwargs):
    """The reply as the model actually shows it: the reasoning is dropped here,
    once, so no caller ever has to know about it."""
    return "".join(text for text, _reasoning in chat_stream(base_url, model, messages, **kwargs))
