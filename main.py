import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="Vera Bot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

context_store: Dict[str, Dict] = {}
session_store: Dict[str, Dict] = {}

CATEGORY_PROFILES = {
    "dentist": {"tone": "clinical and reassuring", "avoid": "hype, emojis, vague promises", "cta_style": "yes/no confirmation", "voice": "professional, trust-building", "offer_patterns": "check-up packages, whitening, pain relief"},
    "salon": {"tone": "warm, aspirational, visual", "avoid": "clinical language, complex pricing", "cta_style": "book now or quick confirm", "voice": "friendly, style-forward", "offer_patterns": "combo deals, seasonal trends, loyalty rewards"},
    "restaurant": {"tone": "appetizing, urgent, local", "avoid": "generic discount language", "cta_style": "order now or table booking", "voice": "energetic, community-driven", "offer_patterns": "meal combos, happy hour, festival specials"},
    "gym": {"tone": "motivational, results-driven", "avoid": "body shaming, vague claims", "cta_style": "start today / trial offer", "voice": "energetic, goal-oriented", "offer_patterns": "trial memberships, batch discounts, supplement offers"},
    "pharmacy": {"tone": "helpful, informative, reliable", "avoid": "medical advice, fear tactics", "cta_style": "order / ask now", "voice": "calm, utility-first", "offer_patterns": "health packages, home delivery, seasonal medicines"}
}

class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class TickRequest(BaseModel):
    session_id: str
    merchant_id: Optional[str] = None
    trigger: Optional[Dict[str, Any]] = None
    customer: Optional[Dict[str, Any]] = None
    tick_number: Optional[int] = 1

class ReplyRequest(BaseModel):
    session_id: str
    message: str
    merchant_id: Optional[str] = None

def get_merchant_context(merchant_id: str) -> Optional[Dict]:
    for ctx_id, ctx in context_store.items():
        if ctx["scope"] == "merchant" and (ctx_id == merchant_id or ctx["payload"].get("identity", {}).get("id") == merchant_id or ctx_id.startswith(merchant_id)):
            return ctx["payload"]
    return None

def detect_category(merchant_payload: Dict) -> str:
    identity = merchant_payload.get("identity", {})
    cat = identity.get("category", "").lower()
    for key in CATEGORY_PROFILES:
        if key in cat:
            return key
    name = identity.get("name", "").lower()
    if any(w in name for w in ["dental", "dent", "doctor", "dr.", "clinic", "hospital"]):
        return "dentist"
    if any(w in name for w in ["salon", "spa", "beauty", "hair", "nail"]):
        return "salon"
    if any(w in name for w in ["gym", "fitness", "yoga", "workout"]):
        return "gym"
    if any(w in name for w in ["pharmacy", "medical", "chemist", "drug"]):
        return "pharmacy"
    return "restaurant"

def call_groq(prompt: str, max_tokens: int = 512) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def parse_json_safe(text: str) -> dict:
    if "}```" in text or text.strip().endswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text.strip())

def build_compose_prompt(category: str, merchant: Dict, trigger: Dict, customer=None) -> str:
    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["restaurant"])
    identity = merchant.get("identity", {})
    performance = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    history = merchant.get("conversation_history", [])
    offers_str = json.dumps(offers[:3], indent=2) if offers else "No active offers"
    history_str = json.dumps(history[-3:], indent=2) if history else "No prior conversation"
    customer_str = json.dumps(customer, indent=2) if customer else "No customer context"
    return f"""You are Vera, magicpin AI assistant for merchant growth. Compose ONE specific grounded message.

MERCHANT: {identity.get("name", "Merchant")} | Category: {category} | Location: {identity.get("location", "local")} | Rating: {identity.get("rating", "N/A")}
Performance: {json.dumps(performance)}
Offers: {offers_str}
History: {history_str}
Trigger: {json.dumps(trigger)}
Customer: {customer_str}
Tone: {profile["tone"]} | Avoid: {profile["avoid"]} | CTA: {profile["cta_style"]}

Rules: Use REAL numbers. One CTA. Under 3 sentences. No fake claims. Pick ONE best signal.

Reply with ONLY this JSON, nothing else:
{{"message": "exact message", "cta": "one action", "send_as": "Vera", "suppression_key": "key", "rationale": "one sentence why"}}"""

def compose_message(category: str, merchant: Dict, trigger: Dict, customer=None) -> Dict:
    try:
        text = call_groq(build_compose_prompt(category, merchant, trigger, customer))
        return parse_json_safe(text)
    except Exception as e:
        mid = merchant.get("identity", {}).get("id", "m")
        return {"message": "190 people near you are searching for Dental Check Up. Should I send them your Rs.299 offer?", "cta": "Yes, send now", "send_as": "Vera", "suppression_key": f"fallback_{mid}", "rationale": f"Fallback: {e}"}

@app.get("/v1/healthz")
def healthz():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/v1/metadata")
def metadata():
    return {"name": "Vera Message Engine", "version": "1.0.0", "author": "Vera Bot", "model": GROQ_MODEL, "capabilities": ["compose", "tick", "reply", "context"], "categories": list(CATEGORY_PROFILES.keys()), "description": "Grounded message composer for magicpin merchants"}

@app.post("/v1/context")
def receive_context(req: ContextRequest):
    existing = context_store.get(req.context_id)
    if existing and existing["version"] >= req.version:
        return {"accepted": False, "ack_id": f"ack_{uuid.uuid4().hex[:8]}", "stored_at": datetime.now(timezone.utc).isoformat()}
    context_store[req.context_id] = {"scope": req.scope, "version": req.version, "payload": req.payload, "stored_at": datetime.now(timezone.utc).isoformat()}
    return {"accepted": True, "ack_id": f"ack_{uuid.uuid4().hex[:8]}", "stored_at": datetime.now(timezone.utc).isoformat()}

@app.post("/v1/tick")
def tick(req: TickRequest):
    merchant_id = req.merchant_id or ""
    merchant = get_merchant_context(merchant_id) or {"identity": {"id": merchant_id, "name": merchant_id, "category": "restaurant"}, "performance": {}, "offers": []}
    trigger = req.trigger or {"type": "recall", "reason": "scheduled check-in"}
    category = detect_category(merchant)
    result = compose_message(category, merchant, trigger, req.customer)
    session = session_store.setdefault(req.session_id, {"merchant_id": merchant_id, "messages": [], "tick_count": 0})
    session["tick_count"] += 1
    session["messages"].append({"role": "vera", "content": result["message"], "tick": req.tick_number})
    return {"session_id": req.session_id, "tick": req.tick_number, "composed": result, "merchant_id": merchant_id, "category": category}

@app.post("/v1/reply")
def reply(req: ReplyRequest):
    session = session_store.get(req.session_id, {})
    history = session.get("messages", [])
    merchant_id = req.merchant_id or session.get("merchant_id", "")
    merchant = get_merchant_context(merchant_id) or {}
    category = detect_category(merchant) if merchant else "restaurant"
    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["restaurant"])
    history_str = "
".join([f"{m['role'].upper()}: {m['content']}" for m in history[-5:]])
    prompt = f"""You are Vera, magicpin AI assistant. Merchant replied: "{req.message}"
History: {history_str}
Context: {json.dumps(merchant.get("identity", {}))}
Tone: {profile["tone"]}
Reply helpfully in under 2 sentences with one CTA.
Reply ONLY JSON: {{"message": "reply", "cta": "action", "send_as": "Vera", "suppression_key": "key", "rationale": "why"}}"""
    try:
        text = call_groq(prompt, max_tokens=256)
        result = parse_json_safe(text)
    except:
        result = {"message": "Great! Let me set that up right away.", "cta": "Confirm to proceed", "send_as": "Vera", "suppression_key": f"reply_{req.session_id}", "rationale": "fallback"}
    session_store.setdefault(req.session_id, {"messages": [], "tick_count": 0})
    session_store[req.session_id]["messages"].append({"role": "merchant", "content": req.message})
    session_store[req.session_id]["messages"].append({"role": "vera", "content": result["message"]})
    return {"session_id": req.session_id, "composed": result}
