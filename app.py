import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict
from agents.market_researcher import market_researcher
from agents.output_writer import save_outputs
from agents.schemas import (ValidationOutput, CompetitorOutput,
                             FinancialsOutput, CritiqueOutput)
import os, time, re, uuid

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
                wait = int(match.group(1))*60 + int(match.group(2)) + 5 if match else (attempt+1)*60
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
def idea_validator(state):
    structured_llm = llm.with_structured_output(ValidationOutput)
    result = invoke_with_retry(structured_llm,
        f"""You are a brutally honest startup idea validator used by top VCs.
Be strict. Most ideas score between 12-22.

Startup idea: {state['startup_idea']}

PROBLEM CLARITY (1-10): 1-3 vague, 4-6 minor, 7-8 clear painful, 9-10 urgent widespread
NOVELTY (1-10): 1-3 copies exist, 4-6 some diff, 7-8 meaningful, 9-10 genuinely new
TAM (1-10): 1-3 under $100M, 4-6 $100M-$1B, 7-8 $1B-$10B, 9-10 over $10B

recommendation: proceed if >=22, pivot if 15-21, abandon if <15
total_score = sum of all three.""")

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

def competitor_analyst(state):
    structured_llm = llm.with_structured_output(CompetitorOutput)
    result = invoke_with_retry(structured_llm,
        f"""Competitive intelligence analyst. Startup idea: {state['startup_idea']}
Find 4-5 real competitors. List strengths, weaknesses, threat level.
Identify the genuine market gap.""")

    competitors_md = "\n".join([
        f"### {c.name}\n- **Threat:** {c.threat_level}\n"
        f"- **Strengths:** {', '.join(c.strengths)}\n"
        f"- **Weaknesses:** {', '.join(c.weaknesses)}"
        for c in result.competitors])

    return {
        "competitor_analysis": f"## Competitor Analysis\n{competitors_md}\n\n"
                               f"## Market Gap\n{result.market_gap}\n\n"
                               f"## Differentiation\n{result.differentiation}",
        "competitor_data": result.model_dump()
    }

def financials_agent(state):
    structured_llm = llm.with_structured_output(FinancialsOutput)
    result = invoke_with_retry(structured_llm,
        f"""Startup financial modeller. Idea: {state['startup_idea']}
Best revenue model, realistic Indian startup break-even estimates.
Top 4 costs. break_even_users and break_even_months as plain integers.""")

    return {
        "financial_model": f"""## Financial Model
**Revenue Model:** {result.revenue_model} — {result.revenue_model_reason}

| Tier | Price |
|------|-------|
| Free | {result.free_tier} |
| Paid | {result.paid_tier_price} |
| Enterprise | {result.enterprise_price} |

**Break-even:** {result.break_even_users:,} users in ~{result.break_even_months} months

**Top Costs:** {', '.join(result.top_costs)}""",
        "financials_data": result.model_dump()
    }

def synthesizer(state):
    response = invoke_with_retry(llm,
        f"""Senior startup analyst. Synthesize into a startup brief. Keep under 600 words.
IDEA: {state['startup_idea']}
VALIDATION: {state['validation_result']}
MARKET: {state['market_research']}
COMPETITORS: {state['competitor_analysis']}
FINANCIALS: {state['financial_model']}

Sections: Executive Summary · Market Opportunity · Competitive Position · Business Model · Key Risks""")
    return {"final_brief": response.content}

def pitch_deck_writer(state):
    response = invoke_with_retry(llm,
        f"""Pitch deck writer. 10 slides, concise. Each slide: title + 3 bullets + one speaker sentence.
BRIEF: {state['final_brief']}
Slides: Problem→Solution→Market→How It Works→Competition→Business Model→Traction→Team→Financials→Ask""")
    return {"pitch_deck": response.content}

def devils_advocate(state):
    structured_llm = llm.with_structured_output(CritiqueOutput)
    result = invoke_with_retry(structured_llm,
        f"""Brutal VC associate. Find 4-5 weaknesses in this pitch deck.
IDEA: {state['startup_idea']}
PITCH: {state['pitch_deck']}
MARKET: {state['market_research'][:1500]}

Per weakness: slide name, issue, severity (high/medium/low), fix.
verdict: strong=no high+<2medium, needs_work=1-2high or 3+medium, weak=3+high
revision_needed: true if needs_work or weak""")

    weaknesses_md = "\n\n".join([
        f"**{w.slide}** — `{w.severity.upper()}`\n- {w.issue}\n- Fix: {w.suggested_fix}"
        for w in result.weaknesses])

    return {
        "critique": f"## Critique — {result.overall_verdict.upper()}\n"
                   f"**Strongest:** {result.strongest_slide} | "
                   f"**Weakest:** {result.weakest_slide}\n\n{weaknesses_md}",
        "critique_data": result.model_dump()
    }

def revised_pitch_writer(state):
    revision_num = state.get("revision_count", 0) + 1
    response = invoke_with_retry(llm,
        f"""Revise this pitch deck addressing every critique. Mark changes with [REVISED].
PITCH: {state['pitch_deck']}
CRITIQUE: {state['critique']}
MARKET DATA: {state['market_research'][:1500]}""")
    return {"pitch_deck": response.content, "revision_count": revision_num}

def should_revise(state):
    if state.get("revision_count", 0) >= 2:
        return "output_writer"
    critique_data = state.get("critique_data", {})
    if critique_data.get("revision_needed") or \
       critique_data.get("overall_verdict") in ["needs_work", "weak"]:
        return "revised_pitch_writer"
    return "output_writer"

# ── Graphs ──────────────────────────────────────────────────────
def build_validation_graph():
    graph = StateGraph(AgentState)
    graph.add_node("idea_validator", idea_validator)
    graph.set_entry_point("idea_validator")
    graph.add_edge("idea_validator", END)
    return graph.compile()

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

    graph.set_entry_point("market_researcher")
    graph.add_edge("market_researcher", "competitor_analyst")
    graph.add_edge("market_researcher", "financials_agent")
    graph.add_edge("competitor_analyst", "synthesizer")
    graph.add_edge("financials_agent", "synthesizer")
    graph.add_edge("synthesizer", "pitch_deck_writer")
    graph.add_edge("pitch_deck_writer", "devils_advocate")
    graph.add_conditional_edges("devils_advocate", should_revise, {
        "revised_pitch_writer": "revised_pitch_writer",
        "output_writer": "output_writer"
    })
    graph.add_edge("revised_pitch_writer", "devils_advocate")
    graph.add_edge("output_writer", END)
    return graph.compile(checkpointer=memory)

# ── Base state helper ────────────────────────────────────────────
def make_base_state(idea):
    return {
        "startup_idea": idea,
        "validation_result": "", "market_research": "",
        "competitor_analysis": "", "financial_model": "",
        "final_brief": "", "pitch_deck": "", "critique": "",
        "revision_count": 0, "critique_data": {}, "output_folder": "",
        "validation_data": {}, "market_data": {}, "competitor_data": {},
        "financials_data": {}, "recommendation": "",
        "validation_score": 0, "human_decision": ""
    }

# ── Gradio functions ─────────────────────────────────────────────

def run_validation(idea):
    """Phase 1: validate the idea and return scorecard."""
    if not idea.strip():
        return "Please enter a startup idea.", "", gr.update(visible=False)

    validation_graph = build_validation_graph()
    result = validation_graph.invoke(make_base_state(idea))

    scorecard = result["validation_result"]
    score = result["validation_score"]
    rec = result["recommendation"].upper()

    status = f"✅ Validation complete — Score: {score}/30 — Recommendation: {rec}"

    # Show the approval buttons
    return status, scorecard, gr.update(visible=True)


def run_analysis(idea, decision):
    """Phase 2: run full analysis after human decision."""
    if decision == "abort":
        return "❌ Aborted. No analysis run.", "", "", "", "", ""

    # Re-run validation to get the state
    validation_graph = build_validation_graph()
    val_result = validation_graph.invoke(make_base_state(idea))

    analysis_state = {**make_base_state(idea), **val_result, "human_decision": decision}
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"analysis: {idea[:50]}"
    }

    with SqliteSaver.from_conn_string("checkpoints.db") as memory:
        analysis_graph = build_analysis_graph(memory)
        result = analysis_graph.invoke(analysis_state, config=config)

    status = f"✅ Complete — Files saved to: {result['output_folder']}"

    return (
        status,
        result.get("market_research", ""),
        result.get("competitor_analysis", ""),
        result.get("financial_model", ""),
        result.get("pitch_deck", ""),
        result.get("critique", "")
    )

# ── UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="Startup Analyst AI") as demo:

    gr.Markdown("# 🚀 Startup Analyst AI")
    gr.Markdown(
        "Enter your startup idea. The system validates it first — "
        "you decide whether to proceed before the full analysis runs."
    )

    # ── Input row ──
    with gr.Row():
        idea_input = gr.Textbox(
            label="Your Startup Idea",
            placeholder="e.g. An AI legal assistant for freelancers in India...",
            lines=2,
            scale=4
        )
        validate_btn = gr.Button("Validate →", variant="primary", scale=1)

    # ── Validation output ──
    validation_status = gr.Textbox(label="Status", interactive=False)
    validation_output = gr.Markdown(label="Validation Scorecard")

    # ── Human decision (hidden until validation completes) ──
    with gr.Row(visible=False) as decision_row:
        gr.Markdown("### Your decision:")
        proceed_btn = gr.Button("✅ Proceed — run full analysis", variant="primary")
        pivot_btn = gr.Button("🔄 Proceed with pivot recommendations", variant="secondary")
        abort_btn   = gr.Button("❌ Abort", variant="stop")

    # ── Analysis outputs ──
    analysis_status = gr.Textbox(label="Analysis Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("📊 Market Research"):
            market_out = gr.Markdown()
        with gr.Tab("⚔️ Competitors"):
            competitor_out = gr.Markdown()
        with gr.Tab("💰 Financials"):
            financials_out = gr.Markdown()
        with gr.Tab("🎤 Pitch Deck"):
            pitch_out = gr.Markdown()
        with gr.Tab("🔥 Critique"):
            critique_out = gr.Markdown()

    # ── Wiring ──
    validate_btn.click(
        fn=run_validation,
        inputs=[idea_input],
        outputs=[validation_status, validation_output, decision_row]
    )

    proceed_btn.click(
        fn=run_analysis,
        inputs=[idea_input, gr.State("proceed")],
        outputs=[analysis_status, market_out, competitor_out,
                 financials_out, pitch_out, critique_out]
    )

    pivot_btn.click(
        fn=run_analysis,
        inputs=[idea_input, gr.State("pivot")],
        outputs=[analysis_status, market_out, competitor_out,
                 financials_out, pitch_out, critique_out]
    )

    abort_btn.click(
        fn=run_analysis,
        inputs=[idea_input, gr.State("abort")],
        outputs=[analysis_status, market_out, competitor_out,
                 financials_out, pitch_out, critique_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # listen on all interfaces, not just localhost
        server_port=7860,
        share=False
    )