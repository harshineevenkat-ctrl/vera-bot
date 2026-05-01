# Vera Message Engine

A deterministic, grounded message composer for magicpin's Vera AI challenge.

## Approach

The bot implements a `compose(category, merchant, trigger, customer?)` function that:

1. **Reads merchant context** from `/v1/context` and stores it in-memory by version
2. **Detects category** (dentist, salon, restaurant, gym, pharmacy) from merchant identity
3. **Picks the single best signal** — trigger type + merchant performance + live offers
4. **Generates a grounded message** using Gemini 1.5 Flash with strict JSON output
5. **Returns** message, CTA, send_as identity, suppression key, and rationale

## Model Choice

**Gemini 1.5 Flash** — fast, reliable, free-tier friendly. Temperature 0.3 for near-deterministic outputs on same inputs.

## Key Design Decisions

- **Signal selection before writing**: The prompt forces the model to pick ONE signal first, not summarize everything
- **Category voice profiles**: Each vertical has its own tone, avoid list, and CTA style baked into the prompt
- **Suppression keys**: Generated per merchant + trigger type to prevent duplicate sends
- **Stateful sessions**: Conversation history tracked per session for coherent reply chains
- **Fallback safety**: Every endpoint has a safe fallback if LLM fails

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /v1/healthz | Health check |
| GET | /v1/metadata | Bot info |
| POST | /v1/context | Store merchant/customer/trigger context |
| POST | /v1/tick | Generate next message |
| POST | /v1/reply | Handle merchant reply and continue conversation |

## Running Locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```

## Deployment

Hosted on Render. Environment variable `GROQ_API_KEY` set in dashboard.
