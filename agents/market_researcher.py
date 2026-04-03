from tavily import TavilyClient
from langchain_groq import ChatGroq
import os
import time

def market_researcher(state: dict) -> dict:
    print("[Market Researcher] Starting web search...")

    from langchain_groq import ChatGroq
    import os
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    idea = state["startup_idea"]

    keyword_prompt = f"""Extract 3 search queries to research this startup idea's market.
Return ONLY a Python list of 3 strings, nothing else.
Example: ["query one", "query two", "query three"]

Startup idea: {idea}"""

    queries = None
    for attempt in range(3):
        try:
            keyword_response = llm.invoke(keyword_prompt)
            import ast
            queries = ast.literal_eval(keyword_response.content.strip())
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"[Market Researcher] Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                queries = None
                break

    if not queries:
        queries = [
            f"{idea} market size 2024",
            f"{idea} competitors India",
            f"{idea} industry trends"
        ]

    print(f"[Market Researcher] Running {len(queries)} searches...")

    all_results = []
    for query in queries:
        try:
            results = tavily.search(query=query, max_results=2)
            for r in results["results"]:
                all_results.append(f"SOURCE: {r['url']}\n{r['content']}")
        except Exception as e:
            print(f"[Market Researcher] Search failed for '{query}': {e}")

    raw_data = "\n\n---\n\n".join(all_results)

    synthesis_prompt = f"""You are a market research analyst. Using ONLY the real data below, write a structured market research report.

STARTUP IDEA: {idea}

REAL SEARCH DATA:
{raw_data}

Write a report with these sections:
1. Market Size (use real numbers from the data where available)
2. Key Trends (cite specific findings)
3. Target Audience
4. Growth Rate

Important: only use facts from the search data. If something isn't in the data, say "data not found"."""

    # Retry the synthesis call too
    for attempt in range(3):
        try:
            response = llm.invoke(synthesis_prompt)
            print("[Market Researcher] Done.")
            return {"market_research": response.content}
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 30
                print(f"[Market Researcher] Rate limited on synthesis. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[Market Researcher] Error: {e}")
                return {"market_research": "Market research unavailable due to API error."}

    return {"market_research": "Market research unavailable — rate limit exceeded after retries."}