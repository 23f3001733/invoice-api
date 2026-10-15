import os, re, json, requests
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)

class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

PROMPT = """Extract fields from the invoice text. Reply with ONLY JSON:
{"vendor": string (company name), "amount": number (total amount due),
 "currency": 3-letter uppercase code (USD/EUR/GBP), "date": payment DUE date as YYYY-MM-DD}"""

def ask_llm(text):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": "llama-3.1-8b-instant",
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text},
            ],
        },
        timeout=25,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])

def regex_backup(text):
    """Plan B if the AI fails."""
    date = re.search(r"\d{4}-\d{2}-\d{2}", text)
    cur = re.search(r"\b(USD|EUR|GBP)\b", text)

    num = r"(\d[\d,]*(?:\.\d+)?)"
    amt = (
        re.search(num + r"\s*(?:USD|EUR|GBP)", text)
        or re.search(r"(?:USD|EUR|GBP)\s*" + num, text)
        or re.search(r"(?:total|amount)[^\d]{0,30}" + num, text, re.I)
    )

    ven = re.search(r"[A-Z][\w-]*(?:\s+[A-Z][\w.&-]*)*\s+(?:Ltd|Inc|LLC|GmbH|Corp)\.?", text)
    return {
        "vendor": ven.group(0) if ven else "Unknown",
        "amount": amt.group(1) if amt else 0,
        "currency": cur.group(1) if cur else "USD",
        "date": date.group(0) if date else "1970-01-01",
    }

def clean(d):
    amount = float(str(d.get("amount", 0)).replace(",", "").replace("$", "") or 0)
    date = re.search(r"\d{4}-\d{2}-\d{2}", str(d.get("date", "")))
    return Invoice(
        vendor=str(d.get("vendor", "Unknown")),
        amount=amount,
        currency=str(d.get("currency", "USD")).upper()[:3],
        date=date.group(0) if date else "1970-01-01",
    )

@app.post("/extract", response_model=Invoice)
def extract(req: ExtractRequest):
    try:
        return clean(ask_llm(req.text))
    except Exception as e:
        print("LLM FAILED:", repr(e))   # yeh Render Logs me dikhega
        try:
            return clean(regex_backup(req.text))
        except Exception:
            return Invoice(vendor="Unknown", amount=0, currency="USD", date="1970-01-01")

@app.get("/")
def home():
    return {"status": "ok"}
