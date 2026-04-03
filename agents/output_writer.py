import os
from datetime import datetime

def save_outputs(state: dict) -> dict:
    print("\n[Output Writer] Saving files...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_slug = state["startup_idea"][:40].replace(" ", "_").replace("/", "-")
    folder = f"outputs/{idea_slug}_{timestamp}"
    os.makedirs(folder, exist_ok=True)

    # Individual files
    files = {
        "1_validation.md": state.get("validation_result", ""),
        "2_market_research.md": state.get("market_research", ""),
        "3_competitor_analysis.md": state.get("competitor_analysis", ""),
        "4_financial_model.md": state.get("financial_model", ""),
        "5_startup_brief.md": state.get("final_brief", ""),
        "6_pitch_deck.md": state.get("pitch_deck", ""),
        "7_critique.md": state.get("critique", ""),
    }

    for filename, content in files.items():
        filepath = os.path.join(folder, filename)
        with open(filepath, "w") as f:
            f.write(f"# {filename.replace('.md','').replace('_',' ').title()}\n\n")
            f.write(f"**Idea:** {state['startup_idea']}\n\n")
            f.write("---\n\n")
            f.write(content)
        print(f"[Output Writer] Saved: {filepath}")

    # Combined report — defined before use
    combined_path = os.path.join(folder, "FULL_REPORT.md")

    revision_count = state.get("revision_count", 0)
    critique_data = state.get("critique_data", {})
    verdict = critique_data.get("overall_verdict", "N/A")
    validation_score = state.get("validation_score", 0)
    recommendation = state.get("recommendation", "N/A")

    with open(combined_path, "w") as f:
        f.write("# Startup Analysis Report\n\n")
        f.write(f"**Idea:** {state['startup_idea']}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}\n\n")
        f.write(f"**Validation Score:** {validation_score}/30 ")
        f.write(f"| **Recommendation:** {recommendation.upper()} ")
        f.write(f"| **Pitch Revisions:** {revision_count} ")
        f.write(f"| **Final Verdict:** {verdict.upper()}\n\n")
        f.write("---\n\n")

        for filename, content in files.items():
            section = filename.replace(".md", "").replace("_", " ").title()
            f.write(f"## {section}\n\n{content}\n\n---\n\n")

    print(f"[Output Writer] Full report saved: {combined_path}")
    print(f"[Output Writer] All files saved to: {folder}/")

    return {"output_folder": folder}