import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="Vera Bot", version="1.0.0")
start_time = datetime.now(timezone.utc)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

context_store: Dict[str, Dict] = {}
conversation_store: Dict[str, Dict] = {}

CATEGORY_PROFILES = {
    "dentist": {"tone": "peer_clinical", "avoid": "guaranteed, 100% safe, hype", "cta_style": "binary yes/no", "vocab": "recall, fluoride, caries, CTR"},
    "dentists": {"tone": "peer_clinical", "avoid": "guaranteed, 100% safe, hype", "cta_style": "binary yes/no", "vocab": "recall, fluoride, caries, CTR"},
    "salon": {"tone": "warm aspirational", "avoid": "clinical language", "cta_style": "book now", "vocab": "bridal, keratin, retention, combo"},
    "salons": {"tone": "warm aspirational", "avoid": "clinical language", "cta_style": "book now", "vocab": "bridal, keratin, retention, combo"},
    "restaurant": {"tone": "energetic local", "avoid": "generic discount language", "cta_style": "order now", "vocab": "covers, AOV, BOGO, happy hour"},
    "restaurants": {"tone": "energetic local", "avoid": "generic discount language", "cta_style": "order now", "vocab": "covers, AOV, BOGO, happy hour"},
    "gym": {"tone": "coach to operator", "avoid": "body shaming", "cta_style": "start today", "vocab": "members, conversion, retention, HIIT"},
    "gyms": {"tone": "coach to operator", "avoid": "body shaming", "cta_style": "start today", "vocab": "members, conversion, retention, HIIT"},
    "pharmacy": {"tone": "trustworthy precise", "avoid": "medical advice, fear tactics", "cta_style": "reply CONFIRM", "vocab": "chronic-Rx, compliance, home delivery"},
    "pharmacies": {"tone": "trustworthy precise", "avoid": "medical advice, fear tactics", "cta_style": "reply CONFIRM", "vocab": "chronic-Rx, compliance, home delivery"},
}

AUTO_REPLY_PHRASES = ["thank you for contacting", "our team will respond", "auto reply", "out of office", "automated response"]

class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class TickRequest(BaseModel):
    now: Optional[str] = None
    available_triggers: Optional[List[str]] = None
    session_id: Optional[str] = None
    merchant_id: Optional[str] = None
    trigger: Optional[Dict[str, Any]] = None
    customer: Optional[Dict[str, Any]] = None
    tick_number: Optional[int] = 1

class ReplyRequest(BaseModel):
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: Optional[str] = "merchant"
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None

def get_context(scope: str, context_id: str) -> Optional[Dict]:
    ctx = context_store.get(context_id)
    if ctx and ctx["scope"] == scope:
        return ctx["payload"]
    for cid, ctx in context_store.items():
        if ctx["scope"] == scope and (cid.startswith(context_id) or context_id.startswith(cid)):
            return ctx["payload"]
    return None

def detect_category(merchant: Dict) -> str:
    cat = merchant.get("category_slug", "") or merchant.get("identity", {}).get("category", "")
    cat = cat.lower()
    for key in CATEGORY_PROFILES:
        if key in cat:
            return key
    name = merchant.get("identity", {}).get("name", "").lower()
    if any(w in name for w in ["dental", "clinic", "doctor"]):
        return "dentists"
    if any(w in name for w in ["salon", "spa", "beauty", "hair"]):
        return "salons"
    if any(w in name for w in ["gym", "fitness", "yoga"]):
        return "gyms"
    if any(w in name for w in ["pharmacy", "chemist", "apollo"]):
        return "pharmacies"
    return "restaurants"

def contexts_loaded_count() -> Dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for ctx in context_store.values():
        scope = ctx.get("scope", "")
        if scope in counts:
            counts[scope] += 1
    return counts

def is_auto_reply(message: str) -> bool:
    return any(phrase in message.lower() for phrase in AUTO_REPLY_PHRASES)

def call_groq(prompt: str, max_tokens: int = 600) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def parse_json_safe(text: str) -> dict:
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

def build_compose_prompt(category: str, merchant: Dict, trigger: Dict, customer: Optional[Dict], category_ctx: Optional[Dict]) -> str:
    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["restaurants"])
    identity = merchant.get("identity", {})
    performance = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    history = merchant.get("conversation_history", [])
    signals = merchant.get("signals", [])

    owner_name = identity.get("owner_first_name") or identity.get("name", "Owner")
    name = identity.get("name", "Merchant")
    location = identity.get("locality", "") + " " + identity.get("city", "")
    rating = identity.get("rating", "N/A")
    languages = identity.get("languages", ["en"])

    offers_str = json.dumps(offers[:3]) if offers else "No active offers"
    history_str = json.dumps(history[-3:]) if history else "No prior conversation"
    customer_str = json.dumps(customer) if customer else "None"
    trigger_str = json.dumps(trigger)
    signals_str = ", ".join(signals) if signals else "none"
    cat_digest = json.dumps(category_ctx.get("digest", [])[:2]) if category_ctx else "none"
    peer_stats = json.dumps(category_ctx.get("peer_stats", {})) if category_ctx else "{}"
    lang_note = "Use Hindi-English mix (hi-en)" if "hi" in languages else "Use English"

    lines = [
        "You are Vera, magicpin's AI assistant. Compose ONE grounded, specific, high-compulsion message.",
        "",
        "=== KEY RULES ===",
        "1. Use the owner's FIRST NAME: " + owner_name,
        "2. Use REAL NUMBERS from the data. Never invent numbers.",
        "3. Pick ONE best signal from trigger+merchant. Do not list everything.",
        "4. Under 3 sentences. One CTA. No URLs.",
        "5. " + lang_note,
        "6. Use domain vocab: " + profile["vocab"],
        "7. Avoid: " + profile["avoid"],
        "10. Add judgment — if data suggests NOT to send a promo, say so",
        "",
        "=== MERCHANT ===",
        "Name: " + name + " | Owner: " + owner_name + " | Location: " + location.strip(),
        "Rating: " + str(rating),
        "Performance: " + json.dumps(performance),
        "Signals: " + signals_str,
        "Offers: " + offers_str,
        "Peer stats: " + peer_stats,
        "Category digest: " + cat_digest,
        "Conversation history: " + history_str,
        "",
        "=== TRIGGER ===",
        trigger_str,
        "",
        "=== CUSTOMER (if applicable) ===",
        customer_str,
        "",
        "=== TONE ===",
        profile["tone"] + " | CTA style: " + profile["cta_style"],
        "",
        'OUTPUT ONLY THIS JSON (no markdown):',
        '{"message": "exact message", "cta": "binary_yes_no or open_ended", "send_as": "vera or merchant_on_behalf", "suppression_key": "meaningful_key", "rationale": "one sentence why"}'
    ]
    return "\n".join(lines)

def compose_message(category: str, merchant: Dict, trigger: Dict, customer=None, category_ctx=None) -> Dict:
    try:
        text = call_groq(build_compose_prompt(category, merchant, trigger, customer, category_ctx))
        return parse_json_safe(text)
    except Exception as e:
        owner = merchant.get("identity", {}).get("owner_first_name", "there")
        mid = merchant.get("merchant_id") or merchant.get("identity", {}).get("id", "m")
        ttype = trigger.get("kind") or trigger.get("type", "recall")
        return {
            "message": f"{owner}, 190 people near you are searching — want me to promote your offer right now?",
            "cta": "binary_yes_no",
            "send_as": "vera",
            "suppression_key": f"fallback_{mid}_{ttype}",
            "rationale": f"Fallback: {str(e)}"
        }

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
        "team_name": "Vera Bot",
        "team_members": ["Vera"],
        "model": GROQ_MODEL,
        "approach": "signal-first composer",
        "contact_email": "vera@example.com",
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "capabilities": ["compose", "tick", "reply", "context"],
        "categories": ["dentists", "salons", "restaurants", "gyms", "pharmacies"]
    }

@app.post("/v1/context")
def receive_context(req: ContextRequest):
    existing = context_store.get(req.context_id)
    if existing and existing["version"] >= req.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "ack_id": "ack_" + uuid.uuid4().hex[:8],
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
    context_store[req.context_id] = {
        "scope": req.scope,
        "version": req.version,
        "payload": req.payload,
        "stored_at": datetime.now(timezone.utc).isoformat()
    }
    return {
        "accepted": True,
        "ack_id": "ack_" + req.context_id + "_v" + str(req.version),
        "stored_at": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/tick")
def tick(req: TickRequest):
    actions = []

    if req.available_triggers:
        for trigger_id in req.available_triggers:
            trigger_ctx = get_context("trigger", trigger_id)
            if not trigger_ctx:
                continue
            merchant_id = trigger_ctx.get("merchant_id") or trigger_ctx.get("payload", {}).get("merchant_id")
            customer_id = trigger_ctx.get("customer_id")
            category_slug = trigger_ctx.get("payload", {}).get("category") or trigger_ctx.get("category")
            merchant = get_context("merchant", merchant_id) if merchant_id else {}
            customer = get_context("customer", customer_id) if customer_id else None
            category_ctx = get_context("category", category_slug) if category_slug else None
            if not merchant:
                merchant = {}
            category = detect_category(merchant) if merchant else (category_slug or "restaurants")
            result = compose_message(category, merchant, trigger_ctx, customer, category_ctx)
            send_as = result.get("send_as", "merchant_on_behalf" if customer_id else "vera")
            conv_id = "conv_" + (merchant_id or "m") + "_" + trigger_id.replace("trg_", "")
            actions.append({
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": send_as,
                "trigger_id": trigger_id,
                "template_name": "vera_composed_v2",
                "body": result["message"],
                "cta": result.get("cta", "binary_yes_no"),
                "suppression_key": result.get("suppression_key", trigger_id),
                "rationale": result.get("rationale", "")
            })
            conversation_store[conv_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id": trigger_id,
                "turns": [{"role": "vera", "content": result["message"]}],
                "auto_reply_count": 0
            }
        return {"actions": actions}

    merchant_id = req.merchant_id or ""
    merchant = get_context("merchant", merchant_id) or {
        "identity": {"id": merchant_id, "name": merchant_id, "category": "restaurant"},
        "performance": {}, "offers": []
    }
    trigger = req.trigger or {"type": "recall", "reason": "scheduled check-in"}
    category = detect_category(merchant)
    result = compose_message(category, merchant, trigger, req.customer, None)
    session_id = req.session_id or "session_" + uuid.uuid4().hex[:8]
    conv_id = "conv_" + merchant_id + "_" + session_id
    conversation_store[conv_id] = {
        "merchant_id": merchant_id,
        "turns": [{"role": "vera", "content": result["message"]}],
        "auto_reply_count": 0
    }
    return {
        "actions": [{
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": None,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trigger.get("id", "manual"),
            "body": result["message"],
            "cta": result.get("cta", "binary_yes_no"),
            "suppression_key": result.get("suppression_key", conv_id),
            "rationale": result.get("rationale", "")
        }],
        "session_id": session_id,
        "category": category,
        "composed": result
    }

@app.post("/v1/reply")
def reply(req: ReplyRequest):
    conv_id = req.conversation_id or req.session_id or "unknown"
    conversation = conversation_store.get(conv_id, {"turns": [], "auto_reply_count": 0, "merchant_id": req.merchant_id or ""})
    merchant_id = req.merchant_id or conversation.get("merchant_id", "")
    merchant = get_context("merchant", merchant_id) or {}
    category = detect_category(merchant) if merchant else "restaurants"
    profile = CATEGORY_PROFILES.get(category, CATEGORY_PROFILES["restaurants"])
    turns = conversation.get("turns", [])
    auto_count = conversation.get("auto_reply_count", 0)

    if is_auto_reply(req.message):
        auto_count += 1
        conversation["auto_reply_count"] = auto_count
        conversation_store[conv_id] = conversation
        if auto_count == 1:
            return {"conversation_id": conv_id, "action": "send", "composed": {
                "message": "Looks like an auto-reply. When you're free, just reply 'Yes' to proceed.",
                "cta": "binary_yes_no", "send_as": "vera",
                "suppression_key": conv_id + "_autoreply_1",
                "rationale": "Detected auto-reply."
            }}
        elif auto_count == 2:
            return {"conversation_id": conv_id, "action": "wait", "wait_seconds": 86400, "rationale": "Auto-reply twice. Waiting 24h."}
        else:
            return {"conversation_id": conv_id, "action": "end", "rationale": "Auto-reply 3x. Closing."}

    msg_lower = req.message.lower()
    if any(p in msg_lower for p in ["stop", "don't message", "unsubscribe", "remove me"]):
        return {"conversation_id": conv_id, "action": "end", "composed": {
            "message": "Apologies for the interruption. I won't message again. Say 'Hi Vera' if anything changes.",
            "cta": "none", "send_as": "vera",
            "suppression_key": conv_id + "_optout",
            "rationale": "Merchant opted out."
        }}

    history_str = "\n".join([t["role"].upper() + ": " + t["content"] for t in turns[-4:]])
    owner_name = merchant.get("identity", {}).get("owner_first_name", "")
    prompt_lines = [
        "You are Vera, magicpin AI. Respond to the merchant's reply.",
        "Conversation: " + history_str,
        "Merchant said: " + req.message,
        "Owner: " + owner_name,
        "Tone: " + profile["tone"],
        "Keep under 2 sentences. One CTA.",
        'OUTPUT ONLY JSON: {"message": "reply", "cta": "action", "send_as": "vera", "suppression_key": "key", "rationale": "why"}'
    ]
    try:
        text = call_groq("\n".join(prompt_lines), max_tokens=256)
        result = parse_json_safe(text)
    except Exception:
        result = {"message": "Great! I'll set that up right away.", "cta": "confirm", "send_as": "vera", "suppression_key": conv_id + "_reply", "rationale": "Fallback reply"}

    turns.append({"role": "merchant", "content": req.message})
    turns.append({"role": "vera", "content": result["message"]})
    conversation["turns"] = turns
    conversation_store[conv_id] = conversation

    return {"conversation_id": conv_id, "action": "send", "composed": result}
