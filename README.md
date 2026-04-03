# Startup Analyst AI

Production-grade multi-agent AI system that validates startup ideas and generates investor-ready pitch decks. Built with LangGraph, Groq LLM, and Tavily API, it features parallel agent execution, human-in-the-loop validation, reflection-based critique loops, and structured outputs with full observability and persistence.

## Architecture

- **8 specialist agents** built with LangGraph
- **Parallel execution** — market research, competitor analysis, 
  and financial modelling run simultaneously (fan-out/fan-in)
- **Devil's Advocate loop** — pitch deck is critiqued and revised 
  up to 2 times using a reflection pattern
- **Human-in-the-loop** — validation scorecard shown before 
  full analysis runs
- **Real web search** via Tavily API — grounded market data, 
  not hallucinated numbers
- **SQLite checkpointing** — crash recovery and session persistence
- **LangSmith observability** — full token and latency tracing

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph |
| LLM | Groq (Llama 3.3 70B) |
| Web search | Tavily Search API |
| Structured outputs | Pydantic |
| Observability | LangSmith |
| UI | Gradio |
| Containerisation | Docker |

## System Flow
```
User idea
    ↓
Idea Validator (scored 1-30)
    ↓
Human approval gate ← you decide here
    ↓
Market Researcher (Tavily) ──┐
Competitor Analyst           ├── parallel
Financials Agent ────────────┘
    ↓
Synthesizer (merges all outputs)
    ↓
Pitch Deck Writer
    ↓
Devil's Advocate ←──────────┐
    ↓                        │ loop (max 2x)
Revised Pitch Writer ────────┘
    ↓
Output Writer (8 files saved)
```

## Output Files Per Run

- `1_validation.md` — scored validation scorecard
- `2_market_research.md` — real market data with sources
- `3_competitor_analysis.md` — SWOT-style competitor breakdown
- `4_financial_model.md` — revenue model and break-even
- `5_startup_brief.md` — synthesized executive brief
- `6_pitch_deck.md` — 10-slide narrative with speaker notes
- `7_critique.md` — Devil's Advocate critique with severity ratings
- `FULL_REPORT.md` — complete analysis in one document

## Setup
```bash
git clone https://github.com/lavanya0505/AI-STARTUP-ANALYST
cd startup-analyst
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=startup-analyst
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

Run terminal version:
```bash
python3 main.py
```

Run web UI:
```bash
python3 app.py
```

Run with Docker:
```bash
docker build -t startup-analyst .
docker run -p 7860:7860 --env-file .env startup-analyst
```

## Resume Bullet

> Built a production multi-agent LangGraph system with parallel 
> execution, Pydantic structured outputs, Tavily web search, 
> Devil's Advocate reflection loops, human-in-the-loop approval, 
> SQLite checkpointing, and LangSmith observability — deployed 
> via Gradio + Docker
