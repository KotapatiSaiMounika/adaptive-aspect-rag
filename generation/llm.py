import os, json, re, time
from dotenv import load_dotenv

load_dotenv()
_client = None


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
            msg = str(e)
            if "per day" in msg or "TPD" in msg or i == retries - 1:
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