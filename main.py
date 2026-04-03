from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict
from agents.market_researcher import market_researcher
from agents.output_writer import save_outputs
from agents.schemas import ValidationOutput, CompetitorOutput, FinancialsOutput, CritiqueOutput
import os
import time
import re
import uuid
from langchain_core.tracers.context import tracing_v2_enabled

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Retry helper ────────────────────────────────────────────────
def invoke_with_retry(chain, prompt, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return chain.invoke(prompt)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search(r"try again in (\d+)m(\d+)", err)
                if match:
                    wait = int(match.group(1)) * 60 + int(match.group(2)) + 5
                else:
                    wait = (attempt + 1) * 60
                print(f"  [Retry] Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retry attempts reached")

# ── State ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    startup_idea: str
    validation_result: str
    market_research: str
    competitor_analysis: str
    financial_model: str
    final_brief: str
    pitch_deck: str
    critique: str
    revision_count: int
    critique_data: dict
    output_folder: str
    validation_data: dict
    market_data: dict
    competitor_data: dict
    financials_data: dict
    recommendation: str
    validation_score: int
    human_decision: str

# ── Agents ──────────────────────────────────────────────────────

def idea_validator(state: AgentState) -> dict:
    print("[Idea Validator] Starting...")
    structured_llm = llm.with_structured_output(ValidationOutput)

    result: ValidationOutput = invoke_with_retry(
        structured_llm,
        f"""You are a brutally honest startup idea validator used by top VCs.
Be strict. Most ideas score between 12-22. Reserve 8-10 only for exceptional cases.

Startup idea: {state['startup_idea']}

Scoring guide:
PROBLEM CLARITY (1-10)
- 1-3: Vague or non-existent problem
- 4-6: Problem exists but is minor or not painful enough
- 7-8: Clear, painful problem with identifiable sufferers
- 9-10: Urgent, expensive, widespread problem with no good solution

NOVELTY (1-10)
- 1-3: Exact copies already exist
- 4-6: Some differentiation but crowded space
- 7-8: Meaningful differentiation, clear unique angle
- 9-10: Genuinely new approach, strong defensibility

TAM (1-10)
- 1-3: Niche market under $100M
- 4-6: Moderate market $100M-$1B
- 7-8: Large market $1B-$10B
- 9-10: Massive market over $10B

recommendation rules:
- total >= 22 → "proceed"
- total 15-21 → "pivot"
- total < 15  → "abandon"

Calculate total_score as sum of all three scores."""
    )

    print(f"[Idea Validator] Score: {result.total_score}/30 → {result.recommendation}")

    formatted = f"""## Validation Scorecard

**{result.one_line_summary}**

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Problem Clarity | {result.problem_clarity_score}/10 | {result.problem_clarity_reason} |
| Novelty | {result.novelty_score}/10 | {result.novelty_reason} |
| TAM | {result.tam_score}/10 | {result.tam_reason} |
| **Total** | **{result.total_score}/30** | |

**Recommendation: {result.recommendation.upper()}**"""

    return {
        "validation_result": formatted,
        "validation_data": result.model_dump(),
        "validation_score": result.total_score,
        "recommendation": result.recommendation
    }


def competitor_analyst(state: AgentState) -> dict:
    print("[Competitor Analyst] Starting...")
    structured_llm = llm.with_structured_output(CompetitorOutput)

    result: CompetitorOutput = invoke_with_retry(
        structured_llm,
        f"""You are a competitive intelligence analyst.

Startup idea: {state['startup_idea']}

Find 4-5 real competitors. For each, list specific strengths
and weaknesses. Rate threat level honestly.
Identify the genuine market gap this startup could fill."""
    )

    competitors_md = "\n".join([
        f"""### {c.name}
- **Threat level:** {c.threat_level}
- **Strengths:** {', '.join(c.strengths)}
- **Weaknesses:** {', '.join(c.weaknesses)}"""
        for c in result.competitors
    ])

    formatted = f"""## Competitor Analysis

{competitors_md}

## Market Gap
{result.market_gap}

## How to Differentiate
{result.differentiation}"""

    print(f"[Competitor Analyst] Found {len(result.competitors)} competitors.")
    return {
        "competitor_analysis": formatted,
        "competitor_data": result.model_dump()
    }


def financials_agent(state: AgentState) -> dict:
    print("[Financials Agent] Starting...")
    structured_llm = llm.with_structured_output(FinancialsOutput)

    result: FinancialsOutput = invoke_with_retry(
        structured_llm,
        f"""You are a startup financial modeller.

Startup idea: {state['startup_idea']}

Choose the most appropriate revenue model for this specific idea.
Give realistic break-even estimates for an Indian startup.
List the top 4 cost categories in order of magnitude.

IMPORTANT: break_even_users and break_even_months must be
plain integers with no quotes, e.g. 1000 not "1000"."""
    )

    formatted = f"""## Financial Model

**Revenue Model:** {result.revenue_model}
*Why:* {result.revenue_model_reason}

### Pricing
| Tier | Price |
|------|-------|
| Free | {result.free_tier} |
| Paid | {result.paid_tier_price} |
| Enterprise | {result.enterprise_price} |

### Break-Even
- **Users needed:** {result.break_even_users:,} paying users
- **Timeline:** ~{result.break_even_months} months

### Top Costs
{chr(10).join(f'- {cost}' for cost in result.top_costs)}"""

    print(f"[Financials Agent] Model: {result.revenue_model}, "
          f"Break-even: {result.break_even_months} months")
    return {
        "financial_model": formatted,
        "financials_data": result.model_dump()
    }


def synthesizer(state):
    decision = state.get("human_decision", "proceed")
    
    pivot_instruction = ""
    if decision == "pivot":
        score_data = state.get("validation_data", {})
        pivot_instruction = f"""
IMPORTANT: The founder has chosen to PIVOT this idea. 
Validation flagged these concerns:
- Problem Clarity: {score_data.get('problem_clarity_score', '?')}/10 — {score_data.get('problem_clarity_reason', '')}
- Novelty: {score_data.get('novelty_score', '?')}/10 — {score_data.get('novelty_reason', '')}
- TAM: {score_data.get('tam_score', '?')}/10 — {score_data.get('tam_reason', '')}

Your brief MUST include a dedicated "Pivot Recommendations" section 
that suggests 2-3 specific ways to adjust the idea to address these 
weaknesses. Be concrete — suggest different target markets, feature 
changes, or business model adjustments."""

    response = invoke_with_retry(llm,
        f"""Senior startup analyst. Synthesize into a startup brief. Keep under 600 words.
IDEA: {state['startup_idea']}
VALIDATION: {state['validation_result']}
MARKET: {state['market_research']}
COMPETITORS: {state['competitor_analysis']}
FINANCIALS: {state['financial_model']}
{pivot_instruction}

Sections: Executive Summary · Market Opportunity · Competitive Position · 
Business Model · Key Risks{'· Pivot Recommendations' if decision == 'pivot' else ''}""")
    
    return {"final_brief": response.content}


def pitch_deck_writer(state):
    decision = state.get("human_decision", "proceed")
    
    pivot_note = ""
    if decision == "pivot":
        pivot_note = """
NOTE: This is a PIVOT analysis. Slide 2 (Solution) must explicitly 
address what changes are needed from the original idea. 
Slide 10 (The Ask) should mention the pivot strategy."""

    response = invoke_with_retry(llm,
        f"""Pitch deck writer. 10 slides, concise. Each slide: title + 3 bullets + one speaker sentence.
BRIEF: {state['final_brief']}
{pivot_note}
Slides: Problem→Solution→Market→How It Works→Competition→Business Model→Traction→Team→Financials→Ask""")
    return {"pitch_deck": response.content}


def devils_advocate(state: AgentState) -> dict:
    print(f"\n[Devil's Advocate] Reviewing pitch deck "
          f"(revision #{state.get('revision_count', 0)})...")
    structured_llm = llm.with_structured_output(CritiqueOutput)

    result: CritiqueOutput = invoke_with_retry(
        structured_llm,
        f"""You are a brutal but fair VC associate reviewing a startup pitch deck.

STARTUP IDEA: {state['startup_idea']}
PITCH DECK: {state['pitch_deck']}
MARKET DATA: {state['market_research'][:1500]}

Find 4-5 specific weaknesses. For each:
- Name the exact slide
- Describe the specific issue
- Rate severity: high/medium/low
- Suggest a concrete fix

verdict rules:
- "strong": no high severity, fewer than 2 medium
- "needs_work": 1-2 high OR 3+ medium
- "weak": 3+ high severity

revision_needed: true if needs_work or weak"""
    )

    weaknesses_md = "\n\n".join([
        f"""**{w.slide}** — Severity: `{w.severity.upper()}`
- Issue: {w.issue}
- Fix: {w.suggested_fix}"""
        for w in result.weaknesses
    ])

    formatted = f"""## Devil's Advocate Critique
**Verdict: {result.overall_verdict.upper()}**
**Strongest slide:** {result.strongest_slide}
**Weakest slide:** {result.weakest_slide}

### Issues Found:
{weaknesses_md}"""

    high_count = sum(1 for w in result.weaknesses if w.severity == "high")
    print(f"[Devil's Advocate] Verdict: {result.overall_verdict} | "
          f"High severity issues: {high_count}")

    return {
        "critique": formatted,
        "critique_data": result.model_dump()
    }


def revised_pitch_writer(state: AgentState) -> dict:
    revision_num = state.get("revision_count", 0) + 1
    print(f"\n[Pitch Deck Writer] Writing revision #{revision_num}...")

    response = invoke_with_retry(
        llm,
        f"""You are an expert pitch deck writer making targeted revisions.

ORIGINAL PITCH DECK: {state['pitch_deck']}
CRITIQUE TO ADDRESS: {state['critique']}
MARKET DATA: {state['market_research']}

1. Fix every issue raised in the critique
2. Add specific data points where claims were unsupported
3. Mark every changed section with [REVISED]
4. Keep unchanged slides exactly as they were
5. Do not add new slides"""
    )

    print(f"[Pitch Deck Writer] Revision #{revision_num} complete.")
    return {
        "pitch_deck": response.content,
        "revision_count": revision_num
    }


def should_revise(state: AgentState) -> str:
    revision_count = state.get("revision_count", 0)
    critique_data = state.get("critique_data", {})

    if revision_count >= 2:
        print(f"\n[Router] Max revisions reached. Moving to output.")
        return "output_writer"

    verdict = critique_data.get("overall_verdict", "strong")
    revision_needed = critique_data.get("revision_needed", False)

    if revision_needed or verdict in ["needs_work", "weak"]:
        print(f"\n[Router] Verdict '{verdict}' — requesting revision #{revision_count + 1}.")
        return "revised_pitch_writer"

    print(f"\n[Router] Verdict '{verdict}' — pitch is strong. Moving to output.")
    return "output_writer"


# ── Graph 1: validation only ────────────────────────────────────
# Small, fast, runs before human sees anything.
# No checkpointing needed — it's just one node.

def build_validation_graph():
    graph = StateGraph(AgentState)
    graph.add_node("idea_validator", idea_validator)
    graph.set_entry_point("idea_validator")
    graph.add_edge("idea_validator", END)
    return graph.compile()


# ── Graph 2: full analysis ──────────────────────────────────────
# Only runs after human approves. Clean, no interrupt needed.

def build_analysis_graph(memory):
    graph = StateGraph(AgentState)

    graph.add_node("market_researcher", market_researcher)
    graph.add_node("competitor_analyst", competitor_analyst)
    graph.add_node("financials_agent", financials_agent)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("pitch_deck_writer", pitch_deck_writer)
    graph.add_node("devils_advocate", devils_advocate)
    graph.add_node("revised_pitch_writer", revised_pitch_writer)
    graph.add_node("output_writer", save_outputs)

    # market_researcher runs first (sequential)
    # then fans out to competitor + financials in parallel
    # synthesizer waits for both
    graph.set_entry_point("market_researcher")
    graph.add_edge("market_researcher", "competitor_analyst")
    graph.add_edge("market_researcher", "financials_agent")
    graph.add_edge("competitor_analyst", "synthesizer")
    graph.add_edge("financials_agent", "synthesizer")
    graph.add_edge("synthesizer", "pitch_deck_writer")
    graph.add_edge("pitch_deck_writer", "devils_advocate")

    graph.add_conditional_edges(
        "devils_advocate",
        should_revise,
        {
            "revised_pitch_writer": "revised_pitch_writer",
            "output_writer": "output_writer"
        }
    )
    graph.add_edge("revised_pitch_writer", "devils_advocate")
    graph.add_edge("output_writer", END)

    return graph.compile(checkpointer=memory)


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    idea = input("Enter your startup idea: ")
    thread_id = str(uuid.uuid4())

    print(f"\nAnalysing: {idea}")
    print(f"Session ID: {thread_id}")
    print("=" * 60)

    # Shared base state
    base_state = {
        "startup_idea": idea,
        "validation_result": "",
        "market_research": "",
        "competitor_analysis": "",
        "financial_model": "",
        "final_brief": "",
        "pitch_deck": "",
        "critique": "",
        "revision_count": 0,
        "critique_data": {},
        "output_folder": "",
        "validation_data": {},
        "market_data": {},
        "competitor_data": {},
        "financials_data": {},
        "recommendation": "",
        "validation_score": 0,
        "human_decision": ""
    }

    # ── Phase 1: validate ──────────────────────────────────────
    print("\nRunning validation...\n")
    validation_graph = build_validation_graph()
    validation_result = validation_graph.invoke(base_state)

    # Show scorecard
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE — AWAITING YOUR DECISION")
    print("=" * 60)
    print(validation_result["validation_result"])
    print("=" * 60)

    # ── Human decision ─────────────────────────────────────────
    print("\nOptions: proceed / pivot / abort")
    human_input = input("Your decision: ").strip().lower()

    while human_input not in ["proceed", "pivot", "abort"]:
        print("Please enter: proceed, pivot, or abort")
        human_input = input("Your decision: ").strip().lower()

    if human_input == "abort":
        print("\n✗ Analysis aborted. No files saved.")

    else:
        # ── Phase 2: full analysis ─────────────────────────────
        # Merge validation output into state for analysis graph
        analysis_state = {**base_state, **validation_result, "human_decision": human_input}

        print(f"\nResuming with decision: {human_input}")
        print("=" * 60)

        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": f"analysis: {idea[:50]}"
        }

        with SqliteSaver.from_conn_string("checkpoints.db") as memory:
            analysis_graph = build_analysis_graph(memory)
            result = analysis_graph.invoke(analysis_state, config=config)

        print("\n" + "=" * 60)
        print(f"✓ Analysis complete.")
        print(f"✓ Session ID: {thread_id}")
        print(f"✓ Files saved to: {result['output_folder']}/")
        print(f"✓ Open FULL_REPORT.md for the complete analysis.")