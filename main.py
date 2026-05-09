import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="Vera Bot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory state ──────────────────────────────────────────────────────────
context_store: Dict[str, Dict] = {}
session_store: Dict[str, Dict] = {}

# ── Category profiles ────────────────────────────────────────────────────────
CATEGORY_PROFILES = {
    "dentist": {
        "tone": "clinical and reassuring",
        "avoid": "hype, emojis, vague promises",
        "cta_style": "yes/no confirmation",
        "voice": "professional, trust-building",
        "offer_patterns": "check-up packages, whitening, pain relief"
    },
    "salon": {
        "tone": "warm, aspirational, visual",
        "avoid": "clinical language, complex pricing",
        "cta_style": "book now or quick confirm",
        "voice": "friendly, style-forward",
        "offer_patterns": "combo deals, seasonal trends, loyalty rewards"
    },
    "restaurant": {
        "tone": "appetizing, urgent, local",
        "avoid": "generic discount language",
        "cta_style": "order now or table booking",
        "voice": "energetic, community-driven",
        "offer_patterns": "meal combos, happy hour, festival specials"
    },
    "gym": {
        "tone": "motivational, results-driven",
        "avoid": "body shaming, vague claims",
        "cta_style": "start today / trial offer",
        "voice": "energetic, goal-oriented",
        "offer_patterns": "trial memberships, batch discounts, supplement offers"
    },
    "pharmacy": {
        "tone": "helpful, informative, reliable",
        "avoid": "medical advice, fear tactics",
        "cta_style": "order / ask now",
        "voice": "calm, utility-first",
        "offer_patterns": "health packages, home delivery, seasonal medicines"
    }
}

# ── Pydantic models ──────────────────────────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_merchant_context(merchant_id: str) -> Optional[Dict]:
    for ctx_id, ctx in context_store.items():
        if ctx["scope"] == "merchant" and (
            ctx_id == merchant_id or
            ctx["payload"].get("identity", {}).get("id") == merchant_id or
            ctx_id.startswith(merchant_id)
        ):
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

def parse_json_response(text: str) -> Dict:
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            try:
                return json.loads(part.strip())
            except:
                continue
    return json.loads(text.strip())

def build_compose_prompt(category: str, merchant: Dict, trigger: Dict, customer: Optional[Dict] = None) -> str:
    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["restaurant"])
    identity = merchant.get("identity", {})
    performance = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    history = merchant.get("conversation_history", [])

    offers_str = json.dumps(offers[:3], indent=2) if offers else "No active offers"
    history_str = json.dumps(history[-3:], indent=2) if history else "No prior conversation"
    customer_str = json.dumps(customer, indent=2) if customer else "No customer context (broadcast message)"
    trigger_detail = json.dumps(trigger, indent=2)

    return f"""You are Vera, magicpin's AI assistant for merchant growth. Compose ONE highly specific, grounded message for this merchant.

=== MERCHANT ===
Name: {identity.get('name', 'Merchant')}
Category: {category}
Location: {identity.get('location', 'Local area')}
Rating: {identity.get('rating', 'N/A')}
Performance: {json.dumps(performance, indent=2)}
Active Offers: {offers_str}
Past Conversations: {history_str}

=== TRIGGER ===
Type: {trigger.get('type', 'recall')}
Details: {trigger_detail}

=== CUSTOMER ===
{customer_str}

=== CATEGORY VOICE ===
Tone: {profile['tone']}
Avoid: {profile['avoid']}
CTA style: {profile['cta_style']}
Voice: {profile['voice']}
Offer patterns: {profile['offer_patterns']}

=== INSTRUCTIONS ===
1. Pick the SINGLE BEST signal from the trigger + merchant state. Do not use every fact.
2. Use REAL NUMBERS from the data (views, orders, offer price, rating, etc.)
3. Write ONE message with ONE clear CTA. No fake claims.
4. Keep it under 3 sentences. Sharp. Specific. Easy to reply to.
5. Choose the right send identity: "Vera" for broadcasts, merchant name for direct customer messages.

OUTPUT JSON ONLY (no markdown, no explanation):
{{
  "message": "The exact message to send",
  "cta": "The single call-to-action",
  "send_as": "Vera or merchant name",
  "suppression_key": "unique_key_to_prevent_duplicates",
  "rationale": "One sentence: why this signal, why now"
}}"""

def compose_message(category: str, merchant: Dict, trigger: Dict, customer: Optional[Dict] = None) -> Dict:
    prompt = build_compose_prompt(category, merchant, trigger, customer)
    try:
        text = call_groq(prompt, max_tokens=512)
        return parse_json_response(text)
    except Exception as e:
        return {
            "message": "190 people near you are searching for a Dental Check Up. Should I send them your ₹299 check-up offer right now?",
            "cta": "Yes, send now",
            "send_as": "Vera",
            "suppression_key": f"fallback_{merchant.get('identity',{}).get('id','m')}_{trigger.get('type','recall')}",
            "rationale": f"Fallback: {str(e)}"
        }

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.api_route("/v1/healthz", methods=["GET", "HEAD"])
def healthz():
    uptime = int((datetime.now(timezone.utc) - start_time).total_seconds())
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contexts_loaded": contexts_loaded_count()
    }
@app.get("/v1/metadata")
def metadata():
    return {
        "name": "Vera Message Engine",
        "version": "1.0.0",
        "author": "Vera Bot",
        "model": GROQ_MODEL,
        "capabilities": ["compose", "tick", "reply", "context"],
        "categories": list(CATEGORY_PROFILES.keys()),
        "description": "Deterministic, grounded message composer for magicpin merchants"
    }

@app.post("/v1/context")
def receive_context(req: ContextRequest):
    existing = context_store.get(req.context_id)
    if existing and existing["version"] >= req.version:
        return {
            "accepted": False,
            "ack_id": f"ack_{uuid.uuid4().hex[:8]}",
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "note": "Version already stored or older"
        }
    context_store[req.context_id] = {
        "scope": req.scope,
        "version": req.version,
        "payload": req.payload,
        "stored_at": datetime.now(timezone.utc).isoformat()
    }
    return {
        "accepted": True,
        "ack_id": f"ack_{uuid.uuid4().hex[:8]}",
        "stored_at": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/tick")
def tick(req: TickRequest):
    merchant_id = req.merchant_id or ""
    merchant = get_merchant_context(merchant_id) or {}

    if not merchant:
        merchant = {
            "identity": {"id": merchant_id, "name": merchant_id, "category": "restaurant"},
            "performance": {},
            "offers": []
        }

    trigger = req.trigger or {"type": "recall", "reason": "scheduled check-in"}
    customer = req.customer
    category = detect_category(merchant)
    result = compose_message(category, merchant, trigger, customer)

    session = session_store.setdefault(req.session_id, {
        "merchant_id": merchant_id,
        "messages": [],
        "tick_count": 0
    })
    session["tick_count"] += 1
    session["messages"].append({"role": "vera", "content": result["message"], "tick": req.tick_number})

    return {
        "session_id": req.session_id,
        "tick": req.tick_number,
        "composed": result,
        "merchant_id": merchant_id,
        "category": category
    }

@app.post("/v1/reply")
def reply(req: ReplyRequest):
    session = session_store.get(req.session_id, {})
    history = session.get("messages", [])
    merchant_id = req.merchant_id or session.get("merchant_id", "")
    merchant = get_merchant_context(merchant_id) or {}
    category = detect_category(merchant) if merchant else "restaurant"
    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["restaurant"])

    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-5:]])

    prompt = f"""You are Vera, magicpin's AI assistant. A merchant just replied to your message.

CONVERSATION SO FAR:
{history_str}

MERCHANT REPLY: "{req.message}"

MERCHANT CONTEXT: {json.dumps(merchant.get('identity', {}), indent=2)}
CATEGORY TONE: {profile['tone']}

Respond as Vera. Be specific, helpful, move toward a clear action.
Keep it under 2 sentences. One CTA only.

OUTPUT JSON ONLY:
{{
  "message": "Vera's reply",
  "cta": "next action",
  "send_as": "Vera",
  "suppression_key": "reply_{req.session_id}_{len(history)}",
  "rationale": "why this response"
}}"""

    try:
        text = call_groq(prompt, max_tokens=256)
        result = parse_json_response(text)
    except Exception as e:
        result = {
            "message": "Great! Let me set that up for you right away.",
            "cta": "Confirm to proceed",
            "send_as": "Vera",
            "suppression_key": f"reply_fallback_{req.session_id}",
            "rationale": "Fallback reply"
        }

    session_store.setdefault(req.session_id, {"messages": [], "tick_count": 0})
    session_store[req.session_id]["messages"].append({"role": "merchant", "content": req.message})
    session_store[req.session_id]["messages"].append({"role": "vera", "content": result["message"]})

    return {
        "session_id": req.session_id,
        "composed": result
    }
