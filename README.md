# 🤖 Vera Message Engine v3.0

> Signal-first AI message composer built for **magicpin's Vera AI Challenge**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat)](https://groq.com)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat)](https://vera-bot-adqs.onrender.com)

**Live Demo:** https://vera-bot-adqs.onrender.com

---

## 🧠 What it does

Vera is magicpin's AI assistant for merchant growth. This bot powers the **message engine** behind Vera — deciding what to say, when to say it, and how to say it for each merchant.

---

## 🏗️ System Architecture
Merchant/Customer Context

↓

/v1/context  ←── Store trigger + merchant state

↓

/v1/tick     ←── Pick ONE best signal

↓

Signal Selection Engine

(category + trigger + merchant state)

↓

Llama 3.3 70B via Groq

(temperature 0.2 — near deterministic)

↓

/v1/reply    ←── Conversation state machine

↓

Actions Array

(body + CTA + send_as + suppression_key + rationale)

---

## ✨ Key Features

- **Signal-first composition** — picks ONE best signal, not every fact
- **5 category voice profiles** — Dentists, Salons, Restaurants, Gyms, Pharmacies
- **Conversation state machine** — auto-reply detection, hostile/opt-out handling
- **Percentage auto-fix** — delta_pct -0.50 correctly reads as -50% drop
- **Always non-empty body** — every reply path has a meaningful fallback

---

## 🎯 Category Voice Profiles

| Category | Tone | Key Terms |
|---|---|---|
| Dentists | peer_clinical | recall, fluoride, caries |
| Salons | warm aspirational | bridal, keratin, skin-prep |
| Restaurants | energetic local | covers, AOV, BOGO |
| Gyms | coach to operator | retention, HIIT, conversion |
| Pharmacies | trustworthy precise | chronic-Rx, compliance |

---

## 🔌 API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | /v1/healthz | Health check |
| GET | /v1/metadata | Team info, model, approach |
| POST | /v1/context | Store context |
| POST | /v1/tick | Generate actions |
| POST | /v1/reply | Handle replies |

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```

Test at: http://localhost:8000/docs

---

## 🛠️ Tech Stack

- **Backend:** FastAPI + Python
- **LLM:** Llama 3.3 70B via Groq
- **Deployment:** Render
Live URL: https://vera-bot-adqs.onrender.com
