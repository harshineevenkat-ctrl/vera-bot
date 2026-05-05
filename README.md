# Vera Message Engine v3.0

A signal-first, grounded message composer built for magicpin's Vera AI challenge.

## What it does

Vera is magicpin's AI assistant for merchant growth. This bot powers the message engine behind Vera — deciding what to say, when to say it, and how to say it for each merchant.

## Approach

**Signal-first composition**: picks ONE best signal from trigger + merchant state + category before writing anything.

1. Receives merchant, customer, category, and trigger context via `/v1/context`
2. Detects business category (dentists, salons, restaurants, gyms, pharmacies)
3. Picks the strongest signal — not every fact, just the right one
4. Generates a grounded, specific message using Llama 3.3 70B via Groq
5. Returns an `actions` array with body, CTA, send_as, suppression_key, and rationale

## Key Design Decisions

### Separate prompts for merchant vs customer scope
- Merchant messages: peer tone, domain vocab, owner first name, real metrics
- Customer messages: warm personal tone, customer first name, language preference, slot data

### Percentage fix
- `delta_pct: -0.50` correctly read as -50% drop, not -0.5%
- All rate/ctr/delta fields auto-converted to human-readable percentages

### Conversation state machine
- Auto-reply detection: 1st warn, 2nd wait 24h, 3rd end
- Hostile/opt-out: graceful exit with suppression
- Intent transition: "Yes/Sure/OK" switches to action mode with specific deliverable
- Customer slot pick: replies addressed to customer, not merchant

### Category voice profiles
Each vertical has its own tone, vocabulary, and avoid list:
- Dentists: peer_clinical — recall, fluoride, caries, high-risk cohort
- Salons: warm aspirational — bridal, keratin, skin-prep, combo
- Restaurants: energetic local — covers, AOV, BOGO, happy hour
- Gyms: coach to operator — members, conversion, retention, HIIT
- Pharmacies: trustworthy precise — chronic-Rx, compliance, home delivery

### Always non-empty body
Every reply path has a meaningful fallback — message body is never empty.

## Model

Llama 3.3 70B via Groq — fast, free tier, near-deterministic at temperature 0.2

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /v1/healthz | Health check + contexts loaded count |
| GET | /v1/metadata | Team info, model, approach |
| POST | /v1/context | Store context (idempotent by version) |
| POST | /v1/tick | Generate actions from available_triggers list |
| POST | /v1/reply | Handle merchant/customer replies with state |

## Running Locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```

Test at: http://localhost:8000/docs

## Deployment

Hosted on Render. Environment variable GROQ_API_KEY set in dashboard.

Live URL: https://vera-bot-adqs.onrender.com
