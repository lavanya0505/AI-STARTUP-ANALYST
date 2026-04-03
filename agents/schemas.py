from pydantic import BaseModel,Field , field_validator
from typing import List, Literal

# ── Idea Validator Schema ──────────────────────────────────────
class ValidationOutput(BaseModel):
    problem_clarity_score: int = Field(ge=1, le=10)
    problem_clarity_reason: str
    novelty_score: int = Field(ge=1, le=10)
    novelty_reason: str
    tam_score: int = Field(ge=1, le=10)
    tam_reason: str
    total_score: int  # sum of all three, out of 30
    recommendation: Literal["proceed", "pivot", "abandon"]
    one_line_summary: str  # e.g. "Strong idea in a growing market"

# ── Market Researcher Schema ───────────────────────────────────
class MarketOutput(BaseModel):
    market_size_2024: str   # e.g. "$1.2B"
    market_size_2028: str   # projected
    cagr: str               # e.g. "19.6%"
    top_trends: List[str]   # list of 3-5 trends
    target_audience: str
    data_confidence: Literal["high", "medium", "low"]
    sources_found: int      # how many real sources Tavily returned

# ── Competitor Analyst Schema ──────────────────────────────────
class Competitor(BaseModel):
    name: str
    strengths: List[str]
    weaknesses: List[str]
    threat_level: Literal["high", "medium", "low"]

class CompetitorOutput(BaseModel):
    competitors: List[Competitor]
    market_gap: str          # the gap your startup fills
    differentiation: str     # how to stand out

# ── Financials Schema ──────────────────────────────────────────
class FinancialsOutput(BaseModel):
    revenue_model: Literal["SaaS", "marketplace", "freemium",
                            "transactional", "advertising", "hybrid"]
    revenue_model_reason: str
    free_tier: str
    paid_tier_price: str
    enterprise_price: str
    break_even_users: int
    break_even_months: int
    top_costs: List[str]

    # These validators run before Pydantic type-checks the field.
    # If the LLM sends "1000" (string), we convert it to 1000 (int).
    @field_validator("break_even_users", "break_even_months", mode="before")
    @classmethod
    def coerce_to_int(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

# ── Devil's Advocate Schema ────────────────────────────────────
class Weakness(BaseModel):
    slide: str
    issue: str
    severity: Literal["high", "medium", "low"]
    suggested_fix: str

class CritiqueOutput(BaseModel):
    weaknesses: List[Weakness]
    strongest_slide: str
    weakest_slide: str
    overall_verdict: Literal["strong", "needs_work", "weak"]
    revision_needed: bool