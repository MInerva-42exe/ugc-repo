# ===============================
# main.py — FastAPI backend + static frontend + robust retrieval/filters
# ===============================

import os
import json
import time
import logging
from itertools import cycle
from functools import lru_cache
from typing import List, Optional, Dict, Any, Set
from contextlib import asynccontextmanager
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# GCP
from google.cloud import firestore, secretmanager
from google.api_core import exceptions
from google.auth import default as google_auth_default  # noqa: F401

# Vertex / Gemini
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import google.generativeai as genai

# ---------- Config ----------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "315200759683")  # project number
REGION = "us-central1"

FIRESTORE_COLLECTION = "success_stories"
API_KEY_SECRET_NAMES = ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]
GENERATION_MODEL_NAME = "gemini-1.5-flash"
VECTOR_SEARCH_ENDPOINT_ID = "7018404821443018752"
VECTOR_SEARCH_DEPLOYED_INDEX_ID = "customer_stories_v2_deploy_1754336152946"

# ---------- Logging & globals ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

firestore_client = None
gemini_api_keys: List[str] = []
api_key_cycler = None
embed_model = None
vector_index_endpoint = None
PROJECT_NUMBER = None

# ---------- Models ----------
class StoryResponse(BaseModel):
    notion_id: str
    customer_name: Optional[str] = ""
    customer_email: Optional[str] = ""
    customer_linkedin: Optional[str] = ""
    customer_designation: Optional[str] = ""
    organization_name: Optional[str] = ""
    org_photo: Optional[str] = ""
    org_logo: Optional[str] = ""
    product: Optional[str] = ""
    industry: Optional[str] = ""
    summary: Optional[str] = ""
    problem: Optional[str] = ""
    solution: Optional[str] = ""
    outcome: Optional[str] = ""
    raw_testimonial_quote: Optional[str] = ""
    case_study_link: Optional[str] = ""
    is_enterprise_case_study: Optional[bool] = False
    is_reference_customer: Optional[bool] = False
    full_story_text_content: Optional[str] = ""
    similarity_score: Optional[float] = None

    # NEW: metadata we’ll use to filter & display
    country: Optional[List[str]] = Field(default_factory=list)
    industry_tags: Optional[List[str]] = Field(default_factory=list)
    product_tags: Optional[List[str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    found_stories: List[StoryResponse] = Field(default_factory=list)
    message: str
    search_type: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

# ---------- Prompt for LLM enhancement ----------
def _build_enhanced_prompt(query: str, stories: List[Dict[str, Any]]) -> str:
    ctx = []
    for s in stories:
        content = s.get("full_story_text_content", "No content available").strip()
        ctx.append(f"STORY (ID: {s.get('notion_id','')}):\n{content}\n---")
    context = "\n".join(ctx)
    return f"""Based on the user's query and the provided story contexts, generate a JSON array of objects.
User Query: "{query}"

For each story, extract the following fields. If a field is not present, use an empty string "" or false.
- "notion_id": The story ID
- "summary": A 2-3 sentence overview.
- "problem": The customer's main challenge.
- "solution": The ManageEngine product or method used.
- "outcome": The measurable results or benefits.

Return ONLY the valid JSON array, with no other text or markdown.

CONTEXTS:
{context}

JSON Response:
"""

# ---------- Vector search (cached) ----------
@lru_cache(maxsize=100)
def _cached_vector_search(query: str, num_neighbors: int = 8) -> List[tuple]:
    """Return list of (external_id, distance)."""
    try:
        if not embed_model or not vector_index_endpoint:
            return []
        embedding = embed_model.get_embeddings([query])[0].values
        resp = vector_index_endpoint.find_neighbors(
            deployed_index_id=VECTOR_SEARCH_DEPLOYED_INDEX_ID,
            queries=[embedding],
            num_neighbors=num_neighbors,
        )
        return [(n.id, n.distance) for n in resp[0]] if resp else []
    except Exception as e:
        logger.error(f"Vector search failed: {e}", exc_info=True)
        return []

# ---------- Firestore helpers ----------
def _fetch_stories_by_notion_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch Firestore 'in' queries (max 10) and key by notion_id field."""
    story_map: Dict[str, Dict[str, Any]] = {}
    if not ids:
        return story_map
    for i in range(0, len(ids), 10):
        chunk = ids[i:i+10]
        docs = (
            firestore_client.collection(FIRESTORE_COLLECTION)
            .where("notion_id", "in", chunk)
            .stream()
        )
        for snap in docs:
            data = snap.to_dict() or {}
            key = data.get("notion_id") or snap.id
            story_map[key] = data
    return story_map

def _fetch_reference_customers(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch up to `limit` docs flagged as reference_consent == True."""
    out: List[Dict[str, Any]] = []
    try:
        q = (
            firestore_client.collection(FIRESTORE_COLLECTION)
            .where("reference_consent", "==", True)
            .limit(limit)
        )
        for snap in q.stream():
            d = snap.to_dict() or {}
            d["notion_id"] = d.get("notion_id") or snap.id
            out.append(d)
    except Exception as e:
        logger.error(f"Fallback fetch of reference customers failed: {e}", exc_info=True)
    return out

# ---------- Normalization & intent helpers ----------
def _normalize_story_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Map Firestore fields -> API/UI schema without mutating input."""
    if not d:
        return {}
    out = dict(d)

    # alias fields
    if "linkedin_profile" in d and not d.get("customer_linkedin"):
        out["customer_linkedin"] = d["linkedin_profile"]
    if "testimonial_quote" in d and not d.get("raw_testimonial_quote"):
        out["raw_testimonial_quote"] = d["testimonial_quote"]

    # arrays normalized
    for k in ("country", "industry_tags", "product_tags"):
        v = out.get(k)
        if v is None:
            out[k] = []
        elif isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = []

    # booleans for badges
    out["is_reference_customer"] = bool(out.get("reference_consent", False))
    out["is_enterprise_case_study"] = bool(out.get("is_enterprise_case_study", False))
    return out

def detect_intent(user_query: str) -> str:
    q = (user_query or "").strip().lower()
    if any(m in q for m in [
        "what can you do", "what do you do", "who are you", "help",
        "how to use", "how do i use", "capabilities", "commands",
        "about you", "about this", "what is this", "how can you help"
    ]):
        return "meta_help"
    return "search"

def _wants_linkedin(q: str) -> bool:
    q = (q or "").lower()
    needles = [
        "linkedin", "linked-in", "profile with linkedin",
        "named person", "named spokesperson", "spokesperson with a linkedin"
    ]
    return any(n in q for n in needles)

REGION_TO_COUNTRIES = {
    "asia": {
        "india","indonesia","pakistan","bangladesh","japan","philippines","vietnam",
        "turkey","iran","thailand","myanmar","south korea","korea","iraq","afghanistan",
        "saudi arabia","uzbekistan","malaysia","yemen","nepal","north korea","taiwan",
        "sri lanka","kazakhstan","syria","cambodia","jordan","azerbaijan","uae",
        "united arab emirates","tajikistan","israel","laos","kyrgyzstan","singapore",
        "oman","state of palestine","palestine","kuwait","georgia","mongolia","armenia",
        "qatar","bahrain","timor-leste","bhutan","maldives","brunei","lebanon","china","hong kong"
    },
    "europe": {"united kingdom","uk","england","scotland","wales","northern ireland","germany","france","italy","spain","netherlands","belgium","sweden","denmark","norway","finland","ireland","switzerland","austria","poland","czechia","czech republic","portugal","greece","hungary","romania","slovakia"},
    "north america": {"united states","usa","us","canada","mexico"},
    "oceania": {"australia","new zealand"},
    "latin america": {"brazil","argentina","chile","peru","colombia","uruguay","paraguay","bolivia","ecuador","panama","costa rica","guatemala","el salvador","honduras","nicaragua","dominican republic"},
    "africa": {"south africa","nigeria","egypt","kenya","morocco","ethiopia","ghana","algeria","tunisia","tanzania","uganda"},
}

def _desired_countries_from_query(q: str) -> Set[str]:
    ql = (q or "").lower()
    wants: Set[str] = set()
    for region, countries in REGION_TO_COUNTRIES.items():
        if region in ql:
            wants.update(countries)
    # common single-country nicknames
    if any(k in ql for k in ["usa","u.s.a","united states","us only","from us","in us"]):
        wants.update({"united states","usa","us"})
    if "uk" in ql or "united kingdom" in ql or "britain" in ql:
        wants.update({"united kingdom","uk","england","scotland","wales","northern ireland"})
    singles = ["india","australia","new zealand","canada","singapore","japan","germany","france"]
    for c in singles:
        if c in ql:
            wants.add(c)
    return wants

# ---------- LLM classifier (greeting/irrelevant/relevant) ----------
def _classify_query(query: str) -> str:
    prompt = f"""Analyze the user's query and classify it into ONLY one of the following categories: "relevant", "irrelevant", "greeting".

Instructions:
- "relevant": The query is about IT, ManageEngine, software, technology, or customer success stories.
- "irrelevant": The query is about non-IT topics like cooking, sports, art, etc.
- "greeting": The query is a simple social greeting like "hi" or "hello".

Respond with a JSON object containing a single key "category".

User Query: "{query}"

JSON Response:
"""
    final_cls = "relevant"
    llm_text = "Classification failed."
    try:
        key = next(api_key_cycler)
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, response_mime_type="application/json"
            ),
        )
        llm_text = resp.text
        final_cls = json.loads(llm_text).get("category", "relevant").lower()
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        llm_text = f"Error: {e}"
    try:
        if firestore_client:
            firestore_client.collection("query_classification_logs").add({
                "query": query, "llm_response_text": llm_text,
                "classified_as": final_cls, "timestamp": firestore.SERVER_TIMESTAMP
            })
    except Exception as e:
        logger.error(f"Failed to log classification result: {e}")
    return final_cls

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global firestore_client, gemini_api_keys, api_key_cycler, embed_model, vector_index_endpoint, PROJECT_NUMBER
    PROJECT_NUMBER = os.environ.get("GCP_PROJECT_ID")
    if not PROJECT_NUMBER or not PROJECT_NUMBER.isdigit():
        raise RuntimeError("GCP_PROJECT_ID env var not set or not a number.")
    try:
        vertexai.init(project=PROJECT_NUMBER, location=REGION)
        embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        endpoint_name = f"projects/{PROJECT_NUMBER}/locations/{REGION}/indexEndpoints/{VECTOR_SEARCH_ENDPOINT_ID}"
        vector_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_name)

        secret_client = secretmanager.SecretManagerServiceClient()
        keys = [
            secret_client.access_secret_version(
                request={"name": f"projects/{PROJECT_NUMBER}/secrets/{n}/versions/latest"}
            ).payload.data.decode("UTF-8")
            for n in API_KEY_SECRET_NAMES
        ]
        gemini_api_keys[:] = [k.strip() for k in keys if k]
        if not gemini_api_keys:
            raise RuntimeError("No Gemini API keys found.")
        api_key_cycler = cycle(gemini_api_keys)

        firestore_client = firestore.Client()
        logger.info("✅ All clients initialized successfully!")
    except Exception as e:
        logger.error(f"❌ FATAL STARTUP ERROR: {e}", exc_info=True)
        raise
    yield
    logger.info("Shutting down application...")

# ---------- App & CORS ----------
app = FastAPI(title="Customer Success Stories API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # legal with wildcard
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# ---------- API ----------
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/query", response_model=ChatResponse)
async def handle_query(request: QueryRequest):
    if not all([firestore_client, embed_model, vector_index_endpoint]):
        raise HTTPException(status_code=500, detail="Core clients not initialized.")

    user_query = (request.query or "").strip()
    logger.info(f"Processing query: '{user_query}'")

    # 1) Meta/help
    if detect_intent(user_query) == "meta_help":
        msg = (
            "I can help you find relevant customer success stories, case studies, and success examples.\n\n"
            "Try things like:\n"
            "• “Show me case studies for the finance industry.”\n"
            "• “Stories about ADManager Plus in healthcare.”\n"
            "• “Enterprise reference customers for Log360.”\n\n"
            "You can add filters like product names, industries, countries/regions, or ask for enterprise/reference customers."
        )
        return ChatResponse(found_stories=[], message=msg, search_type="meta_help")

    # 2) Topic classifier
    classification = _classify_query(user_query)
    logger.info(f"Query classified as: {classification}")
    if classification == "greeting":
        return ChatResponse(found_stories=[], message="Hello! I'm your Customer Success Stories assistant. How can I help?", search_type="casual")
    if classification == "irrelevant":
        return ChatResponse(found_stories=[], message="I specialize in ManageEngine customer success stories. I can't help with topics outside of IT management.", search_type="out_of_scope")

    # 3) Intent modifiers
    ql = user_query.lower()
    enterprise_terms = ["enterprise case study", "enterprise customer", "enterprise case studies", "enterprise customers", "enterprise reference", "enterprise"]
    reference_terms = ["reference customer", "reference case", "customer reference", "reference customers"]
    is_enterprise_request = any(t in ql for t in enterprise_terms)
    is_reference_request = any(t in ql for t in reference_terms)

    search_query = user_query
    if is_reference_request:
        search_query += " customer reference consent contact information"

    # 4) Vector retrieval (wider K + looser threshold for reference)
    neighbors = 100 if (is_reference_request or is_enterprise_request) else 20
    vector_results = _cached_vector_search(search_query, num_neighbors=neighbors)

    SIMILARITY_THRESHOLD = 0.25  # ~75% similarity default
    if is_reference_request:
        SIMILARITY_THRESHOLD = 0.15  # looser for sparse ref queries

    confident = [(doc_id, dist) for (doc_id, dist) in vector_results if (1.0 - dist) >= SIMILARITY_THRESHOLD]

    logger.info(f"[diag] vector_neighbors={len(vector_results)}  confident_kept={len(confident)}  threshold={SIMILARITY_THRESHOLD}")
    logger.info(f"[diag] top_ids={ [i for i,_ in vector_results[:10]] }")
    logger.info(f"[diag] kept_ids={ [i for i,_ in confident] }")

    if not confident:
        return ChatResponse(found_stories=[], message="I couldn't find any high-confidence stories for that. Please try rephrasing your query.", search_type="vector_failed_confidence")

    retrieved_ids = [doc_id for (doc_id, _) in confident]
    story_map = _fetch_stories_by_notion_ids(retrieved_ids)

    ordered: List[Dict[str, Any]] = []
    missing_from_firestore = []
    for doc_id, dist in confident:
        data = story_map.get(doc_id)
        if data:
            data["similarity_score"] = round(1.0 - dist, 4)
            ordered.append(data)
        else:
            missing_from_firestore.append(doc_id)

    logger.info(f"[diag] joined={len(ordered)}  missing_in_firestore={missing_from_firestore}")

    # --- NEW: hard filters from query (country/region + linkedin) ---
    desired_countries = _desired_countries_from_query(user_query)
    must_have_linkedin = _wants_linkedin(user_query)

    if desired_countries:
        def keep_country(s: Dict[str, Any]) -> bool:
            sn = _normalize_story_keys(s)
            story_countries = [str(c).lower().strip() for c in sn.get("country", [])]
            return any(c in desired_countries for c in story_countries)
        ordered = list(filter(keep_country, ordered))
    # normalize keys even if no country filter
    ordered = [_normalize_story_keys(s) for s in ordered]

    if must_have_linkedin:
        ordered = [s for s in ordered if s.get("customer_linkedin")]

    # 5) Filter for enterprise/reference badges if requested
    if is_enterprise_request or is_reference_request:
        filtered = []
        for s in ordered:
            if is_enterprise_request and s.get("is_enterprise_case_study") is True:
                filtered.append(s)
            elif is_reference_request and s.get("is_reference_customer") is True:
                filtered.append(s)
        used_fallback = False
        if is_reference_request and len(filtered) < 5:
            existing_ids = {s.get("notion_id") for s in filtered}
            extra_refs = _fetch_reference_customers(limit=50)
            for s in extra_refs:
                s = _normalize_story_keys(s)
                if s.get("notion_id") not in existing_ids:
                    s.setdefault("is_reference_customer", True)
                    s["similarity_score"] = s.get("similarity_score") or 0.0
                    filtered.append(s)
                if len(filtered) >= 5:
                    break
            used_fallback = True
            logger.info(f"[diag] fallback_applied={used_fallback}  final_ref_count={len(filtered)}")
        ordered = filtered

    if not ordered:
        return ChatResponse(found_stories=[], message="No stories matched all your filters. Try relaxing a constraint.", search_type="filtered_empty")

    # 6) LLM enhancement (cap list to avoid token blowups)
    TOP_FOR_LLM = 10
    stories_for_llm = ordered[:TOP_FOR_LLM]
    enhanced_prompt = _build_enhanced_prompt(user_query, stories_for_llm)
    generated_data = None
    try:
        key = next(api_key_cycler)
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        resp = model.generate_content(
            enhanced_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        txt = (resp.text or "").strip()
        if "```" in txt:
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.S | re.I)
            txt = m.group(1).strip() if m else txt
        generated_data = json.loads(txt) if txt else None
        if isinstance(generated_data, list):
            logger.info(f"✅ LLM enhancement successful! Parsed {len(generated_data)} story summaries.")
        else:
            generated_data = None
            logger.info("ℹ️ LLM returned non-list JSON; skipping enhancement.")
    except Exception as e:
        logger.error(f"LLM enhancement failed: {e}", exc_info=True)
        generated_data = None

    # 7) Merge + validate
    final_stories: List[StoryResponse] = []
    llm_map = {it.get("notion_id"): it for it in generated_data} if isinstance(generated_data, list) else {}

    for s in ordered:
        enhanced = s.copy()
        enhanced.update(llm_map.get(s.get("notion_id"), {}))
        try:
            final_stories.append(StoryResponse(**enhanced))
        except Exception as e:
            logger.error(f"Pydantic validation failed for story {s.get('notion_id')}: {e}")

    if not final_stories:
        return ChatResponse(found_stories=[], message="I couldn't find any stories that met all your criteria.", search_type="no_matches_found")

    msg = f"Found {len(final_stories)} relevant stories for you!"
    return ChatResponse(found_stories=final_stories, message=msg, search_type="hybrid_enhanced" if generated_data else "vector_only")

# ---------- Static mount (AFTER routes so /api/* wins) ----------
BASE_DIR = Path(__file__).parent.resolve()
app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="frontend")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
