from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from clustering import IdeaClusterer

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    app.state.clusterer = IdeaClusterer(["placeholder idea"])
    yield
    # shutdown (if you ever need cleanup, add here)

app = FastAPI(lifespan=lifespan)

# Request/Response models
class IdeasRequest(BaseModel):
    ideas: List[str]

class SuggestRequest(BaseModel):
    idea: str

@app.post("/categorize")
def categorize(req: IdeasRequest) -> Dict[str, Any]:
    clusterer = app.state.clusterer
    clusterer.re_embed(req.ideas)
    k, score, labels = clusterer.cluster()
    groups = clusterer.group()
    keywords = clusterer.label_groups_optimized()
    return {
        "best_k": k,
        "silhouette": score,
        "groups": groups,
        "keywords": keywords,
    }

@app.post("/suggest")
def suggest(req: SuggestRequest) -> Dict[str, str]:
    clusterer = app.state.clusterer
    tip = clusterer.improve_idea(req.idea)
    return {"idea": req.idea, "suggestion": tip}

@app.get("/health")
def health(prewarm: bool = False) -> Dict[str, str]:
    if prewarm:
        # force model load so it's cached in memory
        _ = app.state.clusterer.improve_idea("Warmup idea")
    return {"status": "ok"}
