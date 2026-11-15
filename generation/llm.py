import os, json, re, time
from dotenv import load_dotenv

load_dotenv()
_client = None
_PLACEHOLDERS = ("your_key_here", "your-provider", "model-name", "/absolute/path")


def error_text(e):
    """Full error text including the provider's response body (str(e) can be just 'Error code: 429')."""
    t = str(e)
    try:
        t += " " + (e.response.text or "")
    except Exception:
        pass
    return t


def is_quota_error(e):
    """True if the provider says the DAILY quota is used up (waiting a minute will not help)."""
    t = error_text(e).lower()
    return any(k in t for k in ("per day", "perday", "tpd", "rpd", "daily"))


def is_rate_limit(e):
    return getattr(e, "status_code", None) == 429 or type(e).__name__ == "RateLimitError"


def require_config(*names, ping=True, models=None):
    """Fail fast (seconds, not minutes) if LLM settings are missing, still placeholders, or not working."""
    names = names or ("LLM_API_KEY", "LLM_MODEL")
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit("Missing in .env: " + ", ".join(missing)
                         + "  (copy .env.example to .env and fill it in)")
    fake = [n for n in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "JUDGE_MODEL")
            if any(p in (os.getenv(n) or "") for p in _PLACEHOLDERS)]
    if fake:
        raise SystemExit("These .env values are still the example placeholders: " + ", ".join(fake)
                         + "  (put your real provider values in .env)")
    if ping:
        models = list(models) if models else (
            [os.environ["LLM_MODEL"]] + ([os.environ["JUDGE_MODEL"]] if os.getenv("JUDGE_MODEL") else []))
        for m in dict.fromkeys(models):
            try:
                chat("Reply with the word OK.", "ping", max_tokens=16, model=m, retries=1)
            except Exception as e:
                if is_quota_error(e):
                    raise SystemExit(f"Daily quota is used up for model '{m}'. Run again after it resets, "
                                     f"or switch LLM_MODEL / provider in .env.\n{error_text(e)[:300]}")
                if is_rate_limit(e):  # settings are fine, the provider is just busy
                    print(f"[warn] '{m}' is rate-limited right now (429); settings look valid, continuing.",
                          flush=True)
                    continue
                raise SystemExit(f"LLM test call failed for model '{m}' ({type(e).__name__}): "
                                 f"{error_text(e)[:300]}\nCheck LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env.")


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=os.getenv("LLM_BASE_URL") or None,
                         api_key=os.environ["LLM_API_KEY"], max_retries=0)
    return _client


def chat(system, user, temperature=0.0, max_tokens=2000, model=None, retries=5):
    model = model or os.environ["LLM_MODEL"]
    last = ""
    for i in range(retries):
        try:
            r = _get_client().chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            last = (r.choices[0].message.content or "").strip()
            if last:
                return last
            max_tokens = min(max_tokens * 2, 8000)
        except Exception as e:
            if is_quota_error(e) or i == retries - 1:
                raise  # daily quota gone: retrying only wastes time
            time.sleep(20 * (i + 1))  # per-minute limit: wait and retry
    return last


def chat_json(system, user, **kw):
    raw = chat(system, user + "\n\nReturn ONLY valid JSON, no markdown fences.", **kw)
    m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def llm_fn(prompt: str) -> str:
    """Team-wide adapter: prompt string in, reply string out (matches M2/M4 LLMFn)."""
    return chat("You are a helpful assistant.", prompt)