import os
import time
import json
import re
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional
from groq import Groq

app = FastAPI(title="Vera Bot", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─── In-memory stores ───────────────────────────────────────────────────────
contexts: dict[str, dict] = {}          # context_id → {version, scope, payload}
conversations: dict[str, dict] = {}     # conversation_id → {merchant_id, turn_count, auto_reply_streak, ended, history}
suppressed: set[str] = set()            # suppression_keys already used

# ─── Pydantic models ─────────────────────────────────────────────────────────
class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Any
    delivered_at: Optional[str] = None

class TickRequest(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = []

class ReplyRequest(BaseModel):
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: Optional[str] = None
    turn_number: int = 2

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _is_auto_reply(msg: str) -> bool:
    triggers = [
        "thank you for contacting", "our team will respond",
        "we'll get back to you", "currently unavailable",
        "out of office", "away from the phone", "automatic reply",
        "dhanyawad", "shukriya", "we have received your message"
    ]
    lower = msg.lower()
    return any(t in lower for t in triggers)

def _is_hostile(msg: str) -> bool:
    signals = [
        "stop messaging", "don't message", "not interested", "go away",
        "leave me alone", "waste of time", "useless", "spam",
        "unsubscribe", "opt out", "remove me", "bothering me"
    ]
    lower = msg.lower()
    return any(s in lower for s in signals)

def _is_off_topic(msg: str) -> bool:
    off = ["gst", "income tax", "itr", "loan", "insurance", "marriage", "cricket score"]
    lower = msg.lower()
    return any(o in lower for o in off) and "?" in msg

def _is_intent_commit(msg: str) -> bool:
    positive = ["yes", "let's do it", "go ahead", "proceed", "ok do it", "sure",
                "haan", "bilkul", "sounds good", "do it", "send it", "confirm", "yes please"]
    lower = msg.lower().strip()
    return any(lower.startswith(p) or lower == p for p in positive)

def _get_merchant_context(merchant_id: str) -> Optional[dict]:
    for cid, ctx in contexts.items():
        if ctx["scope"] == "merchant":
            p = ctx["payload"]
            if p.get("merchant_id") == merchant_id or cid == merchant_id:
                return p
    return None

def _get_category_context(slug: str) -> Optional[dict]:
    for cid, ctx in contexts.items():
        if ctx["scope"] == "category" and (cid == slug or ctx["payload"].get("slug") == slug):
            return ctx["payload"]
    return None

def _get_trigger_context(trigger_id: str) -> Optional[dict]:
    ctx = contexts.get(trigger_id)
    if ctx and ctx["scope"] == "trigger":
        return ctx["payload"]
    return None

def _compose_message(merchant_ctx: dict, category_ctx: Optional[dict], trigger_ctx: dict, customer_ctx: Optional[dict] = None) -> dict:
    """Core message composer — uses LLM with tight, case-study-aligned prompt."""
    
    identity = merchant_ctx.get("identity", {})
    performance = merchant_ctx.get("performance", {})
    offers = merchant_ctx.get("offers", [])
    customer_agg = merchant_ctx.get("customer_aggregate", {})
    signals = merchant_ctx.get("signals", [])
    
    owner_name = identity.get("owner_first_name") or identity.get("name", "").split()[0]
    merchant_name = identity.get("name", "")
    category_slug = merchant_ctx.get("category_slug", "")
    locality = identity.get("locality", "")
    city = identity.get("city", "")
    languages = identity.get("languages", ["en"])
    use_hindi = "hi" in languages
    
    # Category voice rules
    voice_rules = ""
    if category_ctx:
        voice = category_ctx.get("voice", {})
        taboo = voice.get("vocab_taboo", [])
        tone = voice.get("tone", "")
        digest = category_ctx.get("digest", [])
        peer_stats = category_ctx.get("peer_stats", {})
        seasonal = category_ctx.get("seasonal_beats", [])
        trend_signals = category_ctx.get("trend_signals", [])
        voice_rules = f"""
CATEGORY VOICE:
- Tone: {tone}
- Taboo words (NEVER use): {', '.join(taboo) if taboo else 'none'}
- Peer avg CTR: {peer_stats.get('avg_ctr', 'N/A')}, Peer avg rating: {peer_stats.get('avg_rating', 'N/A')}
- Category digest items: {json.dumps(digest, indent=2)}
- Seasonal beats: {json.dumps(seasonal)}
- Trend signals: {json.dumps(trend_signals)}
"""
    else:
        digest = []

    # Trigger details
    trigger_kind = trigger_ctx.get("kind", "")
    trigger_payload = trigger_ctx.get("payload", {})
    trigger_id = trigger_ctx.get("id", "")
    urgency = trigger_ctx.get("urgency", 1)

    # Customer details (if customer-facing)
    customer_info = ""
    send_as = "vera"
    cta_style = "open_ended"
    if customer_ctx:
        send_as = "merchant_on_behalf"
        cname = customer_ctx.get("identity", {}).get("first_name", "")
        lapsed = customer_ctx.get("status", {}).get("lapsed_days", "")
        prefs = customer_ctx.get("preferences", {})
        customer_info = f"""
CUSTOMER (message goes to customer, send_as=merchant_on_behalf):
- Name: {cname}
- Lapsed days: {lapsed}
- Preferences: {json.dumps(prefs)}
- Language pref: {'Hindi-English mix (hi-en)' if use_hindi else 'English'}
"""
        cta_style = "binary_yes_no"

    # Build active offer string
    active_offers = [o for o in offers if o.get("status") == "active"]
    offers_str = json.dumps(active_offers[:3]) if active_offers else "no active offers"

    # Key merchant metrics
    views = performance.get("views", performance.get("views_today", 0))
    ctr = performance.get("ctr", 0)
    calls = performance.get("calls", 0)
    high_risk = customer_agg.get("high_risk_adult_count", 0)
    lapsed_180 = customer_agg.get("lapsed_180d_plus", 0)
    retention = customer_agg.get("retention_6mo_pct", 0)

    # Determine the relevant digest item for research triggers
    digest_item = ""
    if trigger_kind == "research_digest" and category_ctx:
        top_item_id = trigger_payload.get("top_item_id", "")
        for d in category_ctx.get("digest", []):
            if d.get("id") == top_item_id:
                digest_item = f"DIGEST ITEM: {d.get('title')} — SOURCE: {d.get('source', '')} — SUMMARY: {d.get('summary', '')}"
                break

    system_prompt = """You are Vera, an AI business assistant for magicpin merchants. 
You compose short, sharp WhatsApp messages to help merchants grow their business.

HARD RULES (violations = 0 score):
1. NO URLs in the body — WhatsApp/Meta will reject them. Never add any http link.
2. NO fabricated numbers — only use numbers from the context given. If a number isn't in context, omit it.
3. NO repetition of previous messages.
4. NO generic openers like "Hope this message finds you well".
5. Max message length: ~120 words. Be concise.
6. Lead with the ONE most compelling signal — the hook must be in the first sentence.
7. Use owner first name if merchant-facing. Use customer first name if customer-facing.
8. Always end with ONE clear, low-friction CTA — not multiple asks.

WHAT MAKES A 50/50 MESSAGE:
- Specificity: real numbers (views, patient counts, % stats, prices, dates) with source citations
- Category fit: domain vocabulary correct for the category (e.g., "covers" for restaurants, "recall" for dental, "retention" for gym)
- Merchant fit: references their real data, their locality, their actual offers
- Trigger relevance: the trigger is the explicit REASON for messaging — state it
- Engagement compulsion: loss aversion, reciprocity, time-bound offer, or social proof

Respond ONLY with valid JSON — no markdown, no explanation. Use this exact format:
{
  "body": "message text here",
  "cta": "binary_yes_no|open_ended|binary_confirm_cancel|multi_choice_slot|none",
  "send_as": "vera|merchant_on_behalf",
  "rationale": "concise explanation of why this message scores high"
}"""

    user_prompt = f"""Compose a message for this merchant/trigger combination.

MERCHANT: {owner_name} — {merchant_name}, {locality}, {city}
CATEGORY: {category_slug}
MERCHANT ID: {merchant_ctx.get('merchant_id', '')}

PERFORMANCE (last 30 days):
- Views: {views}, CTR: {ctr} (vs peer avg CTR mentioned in category context)
- Calls: {calls}
- View delta 7d: {performance.get('delta_7d', {}).get('views_pct', 'N/A')}

CUSTOMER AGGREGATE:
- High-risk adults: {high_risk}
- Lapsed 180d+: {lapsed_180}
- 6-month retention: {retention}

ACTIVE OFFERS: {offers_str}

SIGNALS: {', '.join(signals) if signals else 'none'}

TRIGGER KIND: {trigger_kind} (urgency: {urgency}/5)
TRIGGER PAYLOAD: {json.dumps(trigger_payload)}
{digest_item}

{voice_rules}
{customer_info}

NOW compose the perfect message. Remember:
- If trigger is research_digest: cite the source (journal, page). Add "— SOURCE" at end.
- If trigger is seasonal_perf_dip: reframe the dip as normal, give industry range data, propose retention action.
- If trigger is curious_ask_due: ask one simple question with an upfront reciprocity offer.
- If trigger is recall_due (customer): use customer name, state exact time-since-last-visit, offer specific slots.
- If trigger is supply_alert: state the batch numbers, bounded risk, exact affected customer count.
- If trigger is ipl_match_today or similar event: consider whether it helps or hurts — give the contrarian recommendation if data supports it.
- send_as should be "merchant_on_behalf" if there is a customer, else "vera"
- cta for booking flows: "multi_choice_slot"
- cta for research/info: "open_ended"  
- cta for confirmations: "binary_confirm_cancel"
- cta for simple yes/no: "binary_yes_no"
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return result
    except Exception as e:
        # Fallback — still use real data from context
        fallback_body = f"{owner_name}, we noticed a new opportunity for {merchant_name}."
        if high_risk:
            fallback_body += f" You have {high_risk} high-risk patients worth reaching now."
        if active_offers:
            offer_title = active_offers[0].get("title", "")
            fallback_body += f" Your {offer_title} is ready to promote."
        fallback_body += " Want me to draft an outreach plan?"
        return {
            "body": fallback_body,
            "cta": "binary_yes_no",
            "send_as": send_as,
            "rationale": f"Fallback due to LLM error: {str(e)[:100]}"
        }

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for ctx in contexts.values():
        scope = ctx.get("scope", "")
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }

@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Vera Solo",
        "team_members": ["Harshinee Venkat"],
        "model": "llama-3.3-70b-versatile via Groq",
        "approach": "Context-aware composer with trigger-kind dispatch, category voice enforcement, merchant-data grounding, and conversation-state machine (auto-reply detection, intent transition, hostile exit)",
        "contact_email": "vera@magicpin.com",
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/context")
def receive_context(req: ContextRequest):
    existing = contexts.get(req.context_id)
    if existing and existing["version"] >= req.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": existing["version"],
            "ack_id": f"ack_{uuid.uuid4().hex[:8]}",
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
    contexts[req.context_id] = {
        "scope": req.scope,
        "version": req.version,
        "payload": req.payload if isinstance(req.payload, dict) else {},
        "stored_at": datetime.now(timezone.utc).isoformat()
    }
    return {
        "accepted": True,
        "ack_id": f"ack_{req.context_id}_v{req.version}",
        "stored_at": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/tick")
def tick(req: TickRequest):
    actions = []

    for trigger_id in req.available_triggers:
        trigger_ctx = _get_trigger_context(trigger_id)
        if not trigger_ctx:
            continue

        # Check suppression
        sup_key = trigger_ctx.get("suppression_key", trigger_id)
        if sup_key in suppressed:
            continue

        merchant_id = trigger_ctx.get("merchant_id")
        customer_id = trigger_ctx.get("customer_id")

        merchant_ctx = _get_merchant_context(merchant_id) if merchant_id else None
        if not merchant_ctx:
            continue

        category_slug = merchant_ctx.get("category_slug", "")
        category_ctx = _get_category_context(category_slug)

        customer_ctx = None
        if customer_id:
            cctx = contexts.get(customer_id)
            customer_ctx = cctx["payload"] if cctx else None

        composed = _compose_message(merchant_ctx, category_ctx, trigger_ctx, customer_ctx)

        # Build conversation_id (meaningful, decodable)
        ts_tag = (req.now or datetime.now(timezone.utc).isoformat())[:7].replace("-", "_")
        kind_short = trigger_ctx.get("kind", "msg")[:12]
        conv_id = f"conv_{merchant_id}_{kind_short}_{ts_tag}"

        # Track conversation
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trigger_id,
            "auto_reply_streak": 0,
            "ended": False,
            "last_body": composed.get("body", ""),
            "turn_count": 1
        }

        suppressed.add(sup_key)

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger_ctx.get('kind', 'composed')}_v2",
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "binary_yes_no"),
            "suppression_key": sup_key,
            "rationale": composed.get("rationale", "")
        }
        actions.append(action)

        # One action per tick to avoid spam
        break

    return {"actions": actions}

@app.post("/v1/reply")
def reply(req: ReplyRequest):
    conv_id = req.conversation_id or req.session_id or ""
    conv = conversations.get(conv_id, {})
    merchant_id = req.merchant_id or conv.get("merchant_id", "")
    
    msg = req.message.strip()
    turn = req.turn_number

    # Already ended conversation
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already closed."}

    # ── Hostile / opt-out ────────────────────────────────────────────────────
    if _is_hostile(msg):
        if conv_id:
            conversations[conv_id]["ended"] = True
        return {
            "action": "send",
            "body": "Apologies for the interruption — won't message again. If anything changes, just reply 'Hi Vera' to restart. 🙏",
            "cta": "none",
            "rationale": "Merchant expressed frustration/opt-out. One polite exit message, then closing conversation."
        }

    # ── Auto-reply detection ─────────────────────────────────────────────────
    if _is_auto_reply(msg):
        streak = conv.get("auto_reply_streak", 0) + 1
        if conv_id:
            conversations[conv_id]["auto_reply_streak"] = streak

        if streak == 1:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' or 'No' to the earlier question.",
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected. One gentle prompt to flag it to the owner."
            }
        elif streak == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row — owner not at phone. Waiting 24h before retry."
            }
        else:
            if conv_id:
                conversations[conv_id]["ended"] = True
            return {
                "action": "end",
                "rationale": "Auto-reply 3x in a row. No real engagement signal. Closing conversation."
            }

    # ── Off-topic ─────────────────────────────────────────────────────────────
    if _is_off_topic(msg):
        merchant_ctx = _get_merchant_context(merchant_id)
        trigger_id = conv.get("trigger_id", "")
        trigger_ctx = _get_trigger_context(trigger_id) if trigger_id else {}
        trigger_kind = (trigger_ctx or {}).get("kind", "your campaign") if trigger_ctx else "your campaign"
        return {
            "action": "send",
            "body": f"That's outside what I can help with directly — best to check with your CA or a specialist. Coming back to {trigger_kind} — shall I proceed with the draft?",
            "cta": "binary_yes_no",
            "rationale": "Off-topic ask politely declined; thread redirected back to original trigger."
        }

    # ── Intent commit (merchant says yes / let's do it) ───────────────────────
    if _is_intent_commit(msg):
        merchant_ctx = _get_merchant_context(merchant_id) if merchant_id else {}
        customer_agg = (merchant_ctx or {}).get("customer_aggregate", {})
        offers = (merchant_ctx or {}).get("offers", [])
        active_offers = [o for o in offers if o.get("status") == "active"]
        identity = (merchant_ctx or {}).get("identity", {})
        owner_name = identity.get("owner_first_name", "")
        high_risk = customer_agg.get("high_risk_adult_count", 0)
        lapsed = customer_agg.get("lapsed_180d_plus", 0)
        offer_str = active_offers[0].get("title", "your active offer") if active_offers else "your active offer"

        target_count = high_risk or lapsed or "your customer list"
        body = f"On it{', ' + owner_name if owner_name else ''}. Drafting the campaign for {target_count} customers now — targeting them with {offer_str}. Reply CONFIRM to send, or CANCEL to hold."

        return {
            "action": "send",
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed explicitly — switching from qualification to execution mode with concrete scope and confirm gate."
        }

    # ── General reply — use LLM to continue conversation ─────────────────────
    merchant_ctx = _get_merchant_context(merchant_id) if merchant_id else {}
    trigger_id = conv.get("trigger_id", "")
    trigger_ctx = _get_trigger_context(trigger_id) if trigger_id else {}
    category_slug = (merchant_ctx or {}).get("category_slug", "")
    category_ctx = _get_category_context(category_slug)

    system = """You are Vera, a concise AI business assistant for magicpin. 
The merchant has replied to your earlier message. Continue the conversation naturally.
Rules:
- No URLs in body
- No fabricated numbers
- Stay on the original topic unless merchant explicitly changes it
- Max 80 words
- End with ONE clear next step
Return JSON only: {"action": "send", "body": "...", "cta": "binary_yes_no|open_ended|binary_confirm_cancel|none", "rationale": "..."}"""

    user = f"""Merchant replied: "{msg}" (turn {turn})
Original trigger kind: {(trigger_ctx or {}).get('kind', 'unknown')}
Merchant: {(merchant_ctx or {}).get('identity', {}).get('name', merchant_id)}
Continue the conversation. If merchant is asking for something specific, deliver it now."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.3,
            max_tokens=300
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return result
    except Exception as e:
        return {
            "action": "send",
            "body": "Got it — let me pull that together for you. Give me a moment. Shall I go ahead?",
            "cta": "binary_yes_no",
            "rationale": f"Fallback reply: {str(e)[:80]}"
        }
