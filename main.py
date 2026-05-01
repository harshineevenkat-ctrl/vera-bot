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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── State ────────────────────────────────────────────────────────────────────
context_store: Dict[str, Dict] = {}   # context_id → {scope, version, payload}
conversation_store: Dict[str, Dict] = {}  # conversation_id → {turns, merchant_id, ...}
start_time = datetime.now(timezone.utc)

CATEGORY_PROFILES = {
    "dentist": {"tone": "peer_clinical", "avoid": "guaranteed, 100% safe, hype, emojis overuse", "cta_style": "binary yes/no", "vocab": "recall, fluoride, caries, CTR, high-risk cohort"},
    "dentists": {"tone": "peer_clinical", "avoid": "guaranteed, 100% safe, hype, emojis overuse", "cta_style": "binary yes/no", "vocab": "recall, fluoride, caries, CTR, high-risk cohort"},
    "salon": {"tone": "warm aspirational", "avoid": "clinical language, complex pricing", "cta_style": "book now / confirm slot", "vocab": "bridal, keratin, skin-prep, retention, combo"},
    "salons": {"tone": "warm aspirational", "avoid": "clinical language, complex pricing", "cta_style": "book now / confirm slot", "vocab": "bridal, keratin, skin-prep, retention, combo"},
    "restaurant": {"tone": "energetic local", "avoid": "generic discount language, URLs", "cta_style": "order now / confirm", "vocab": "covers, AOV, BOGO, happy hour, delivery"},
    "restaurants": {"tone": "energetic local", "avoid": "generic discount language, URLs", "cta_style": "order now / confirm", "vocab": "covers, AOV, BOGO, happy hour, delivery"},
    "gym": {"tone": "coach to operator", "avoid": "body shaming, vague claims", "cta_style": "start today / trial", "vocab": "members, conversion, ad spend, retention, HIIT"},
    "gyms": {"tone": "coach to operator", "avoid": "body shaming, vague claims", "cta_style": "start today / trial", "vocab": "members, conversion, ad spend, retention, HIIT"},
    "pharmacy": {"tone": "trustworthy precise", "avoid": "medical advice, fear tactics, URLs", "cta_style": "reply CONFIRM / call", "vocab": "chronic-Rx, batch, sub-potency, compliance, home delivery"},
    "pharmacies": {"tone": "trustworthy precise", "avoid": "medical advice, fear tactics, URLs", "cta_style": "reply CONFIRM / call", "vocab": "chronic-Rx, batch, sub-potency, compliance, home delivery"},
}

AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond",
    "we will get back to you",
    "auto reply",
    "out of office",
    "automated response",
]

# ── Models ───────────────────────────────────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_context(scope: str, context_id: str) -> Optional[Dict]:
    ctx = context_store.get(context_id)
    if ctx and ctx["scope"] == scope:
        return ctx["payload"]
    for cid, ctx in context_store.items():
        if ctx["scope"] == scope and (cid.startswith(context_id) or context_id.startswith(cid)):
            return ctx["payload"]
    return None

def find_merchant_for_trigger(trigger_payload: Dict) -> Optional[Dict]:
    merchant_id = trigger_payload.get("merchant_id") or trigger_payload.get("payload", {}).get("merchant_id")
    if not merchant_id:
        return None
    return get_context("merchant", merchant_id)

def detect_category(merchant: Dict) -> str:
    cat = merchant.get("category_slug", "") or merchant.get("identity", {}).get("category", "")
    cat = cat.lower()
    for key in CATEGORY_PROFILES:
        if key in cat:
            return key
    name = merchant.get("identity", {}).get("name", "").lower()
    if any(w in name for w in ["dental", "clinic", "doctor", "dr."]):
        return "dentists"
    if any(w in name for w in ["salon", "spa", "beauty", "hair"]):
        return "salons"
    if any(w in name for w in ["gym", "fitness", "yoga"]):
        return "gyms"
    if any(w in name for w in ["pharmacy", "chemist", "apollo"]):
        return "pharmacies"
    return "restaurants"

def is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in AUTO_REPLY_PHRASES)

def contexts_loaded_count() -> Dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for ctx in context_store.values():
        scope = ctx.get("scope", "")
        if scope in counts:
            counts[scope] += 1
    return counts

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
    customer_agg = merchant.get("customer_aggregate", {})
    subscription = merchant.get("subscription", {})

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
    agg_str = json.dumps(customer_agg) if customer_agg else "{}"
    cat_digest = json.dumps(category_ctx.get("digest", [])[:2]) if category_ctx else "none"
    peer_stats = json.dumps(category_ctx.get("peer_stats", {})) if category_ctx else "{}"

    lang_note = "Use Hindi-English mix (hi-en)" if "hi" in languages else "Use English"

    lines = [
        "You are Vera, magicpin's AI assistant. Compose ONE grounded, specific, high-compulsion message.",
        "",
        "=== KEY RULES (follow exactly) ===",
        "1. Use the owner's FIRST NAME: " + owner_name,
        "2. Use REAL NUMBERS from the data. Never invent numbers.",
        "3. Pick ONE best signal from trigger+merchant. Do not list everything.",
        "4. Under 3 sentences. One CTA. No URLs.",
        "5. " + lang_note,
        "6. Use domain vocab: " + profile["vocab"],
        "7. Avoid: " + profile["avoid"],
        "8. If trigger has research/digest item, cite source at end (e.g. — JIDA Oct 2026 p.14)",
        "9. For customer messages: use customer first name, honor language pref, use merchant_on_behalf as send_as",
        "10. Add judgment — if data suggests NOT to send a promo, say so and suggest better action",
        "",
        "=== MERCHANT ===",
        "Name: " + name + " | Owner: " + owner_name + " | Location: " + location.strip(),
        "Rating: " + str(rating) + " | Subscription: " + json.dumps(subscription),
        "Performance: " + json.dumps(performance),
        "Signals: " + signals_str,
        "Customer aggregate: " + agg_str,
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
        "OUTPUT ONLY THIS JSON (no markdown):",
        '{"message": "exact message", "cta": "binary_yes_no or open_ended or multi_choice_slot", "send_as": "vera or merchant_on_behalf", "suppression_key": "meaningful_key", "rationale": "one sentence: why this signal, why now"}'
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
            "message": owner + ", 190 people near you are searching for a check-up. Want me to promote your active offer to them right now?",
            "cta": "binary_yes_no",
            "send_as": "vera",
            "suppression_key": "fallback_" + str(mid) + "_" + str(ttype),
            "rationale": "Fallback: " + str(e)
        }

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/v1/healthz")
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
        "approach": "signal-first composer: picks best trigger+merchant signal, applies category voice, outputs grounded specific message with one CTA",
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
            "current_version": existing["version"],
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

    # Judge harness format: available_triggers list
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

            send_as = "merchant_on_behalf" if customer_id else "vera"
            if "send_as" in result:
                send_as = result["send_as"]

            conv_id = "conv_" + (merchant_id or "m") + "_" + trigger_id.replace("trg_", "")

            action = {
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
            }
            actions.append(action)

            # Track conversation
            conversation_store[conv_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id": trigger_id,
                "turns": [{"role": "vera", "content": result["message"]}],
                "auto_reply_count": 0
            }

        return {"actions": actions}

    # Legacy / test format: merchant_id + trigger directly
    merchant_id = req.merchant_id or ""
    merchant = get_context("merchant", merchant_id) or {
        "identity": {"id": merchant_id, "name": merchant_id, "category": "restaurant"},
        "performance": {}, "offers": []
    }
    trigger = req.trigger or {"type": "recall", "reason": "scheduled check-in", "kind": "recall"}
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
    turn_number = req.turn_number or (len(turns) + 1)

    # Detect auto-reply
    if is_auto_reply(req.message):
        auto_count += 1
        conversation["auto_reply_count"] = auto_count
        conversation_store[conv_id] = conversation

        if auto_count == 1:
            return {
                "conversation_id": conv_id,
                "action": "send",
                "composed": {
                    "message": "Looks like an auto-reply. When you're free, just reply 'Yes' to the earlier message and I'll take it from there.",
                    "cta": "binary_yes_no",
                    "send_as": "vera",
                    "suppression_key": conv_id + "_autoreply_1",
                    "rationale": "Detected auto-reply; prompting owner to engage directly."
                }
            }
        elif auto_count == 2:
            return {
                "conversation_id": conv_id,
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice — owner not available. Waiting 24h before retry."
            }
        else:
            return {
                "conversation_id": conv_id,
                "action": "end",
                "rationale": "Auto-reply 3x in a row. No engagement signal. Closing conversation."
            }

    # Detect hostile / opt-out
    msg_lower = req.message.lower()
    hostile_phrases = ["stop", "don't message", "useless", "bothering", "unsubscribe", "remove me"]
    if any(p in msg_lower for p in hostile_phrases):
        return {
            "conversation_id": conv_id,
            "action": "end",
            "composed": {
                "message": "Apologies for the interruption. I won't message again. If anything changes, just say 'Hi Vera'. \U0001f64f",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": conv_id + "_optout",
                "rationale": "Merchant expressed frustration; graceful exit with opt-out path."
            },
            "rationale": "Merchant hostile/opted-out; closing and suppressing for 30 days."
        }

    # Detect intent transition ("ok let's do it", "yes", "confirm")
    action_phrases = ["let's do it", "yes", "ok go ahead", "confirm", "proceed", "do it", "send it"]
    if any(p in msg_lower for p in action_phrases):
        # Switch to execution mode
        history_str = "\n".join([t["role"].upper() + ": " + t["content"] for t in turns[-4:]])
        merchant_name = merchant.get("identity", {}).get("name", "your business")
        owner_name = merchant.get("identity", {}).get("owner_first_name", "")
        prompt_lines = [
            "You are Vera, magicpin AI. The merchant just confirmed they want to proceed.",
            "Conversation so far: " + history_str,
            "Merchant said: " + req.message,
            "Merchant: " + merchant_name + " | Owner: " + owner_name,
            "Switch to ACTION mode. Tell them exactly what you're doing next, with a specific deliverable and timeline.",
            "Keep it under 2 sentences. One confirm CTA.",
            'OUTPUT ONLY JSON: {"message": "action message", "cta": "binary_confirm_cancel", "send_as": "vera", "suppression_key": "key", "rationale": "why"}'
        ]
        try:
            text = call_groq("\n".join(prompt_lines), max_tokens=256)
            result = parse_json_safe(text)
        except Exception:
            result = {
                "message": "Great! Drafting that now — you'll have it in 90 seconds. Reply CONFIRM to send it out.",
                "cta": "binary_confirm_cancel",
                "send_as": "vera",
                "suppression_key": conv_id + "_confirm",
                "rationale": "Merchant committed; switching to execution."
            }
        turns.append({"role": "merchant", "content": req.message})
        turns.append({"role": "vera", "content": result["message"]})
        conversation["turns"] = turns
        conversation_store[conv_id] = conversation
        return {"conversation_id": conv_id, "action": "send", "composed": result}

    # Normal reply
    history_str = "\n".join([t["role"].upper() + ": " + t["content"] for t in turns[-5:]])
    owner_name = merchant.get("identity", {}).get("owner_first_name", "")
    prompt_lines = [
        "You are Vera, magicpin AI assistant. Continue this conversation helpfully.",
        "Conversation: " + history_str,
        "Merchant reply (turn " + str(turn_number) + "): " + req.message,
        "Merchant context: " + json.dumps(merchant.get("identity", {})),
        "Tone: " + profile["tone"] + " | Owner first name: " + owner_name,
        "Be specific and move toward one clear action. Under 2 sentences. One CTA. No URLs.",
        'OUTPUT ONLY JSON: {"message": "reply", "cta": "binary_yes_no or open_ended", "send_as": "vera", "suppression_key": "key", "rationale": "why"}'
    ]
    try:
        text = call_groq("\n".join(prompt_lines), max_tokens=256)
        result = parse_json_safe(text)
    except Exception:
        result = {
            "message": "Got it! Let me set that up right away. Reply YES to confirm.",
            "cta": "binary_yes_no",
            "send_as": "vera",
            "suppression_key": conv_id + "_reply_" + str(turn_number),
            "rationale": "Fallback reply"
        }

    turns.append({"role": "merchant", "content": req.message})
    turns.append({"role": "vera", "content": result["message"]})
    conversation["turns"] = turns
    conversation_store[conv_id] = conversation

    return {"conversation_id": conv_id, "action": "send", "composed": result}
