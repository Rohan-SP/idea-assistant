from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()
clusterer = None

class IdeasRequest(BaseModel):
    ideas: List[str]

class SuggestRequest(BaseModel):
    idea: str

def get_clusterer():
    global clusterer
    if clusterer is None:
        from clustering import IdeaClusterer
        clusterer = IdeaClusterer(["placeholder idea"])
    return clusterer

@app.post("/categorize")
def categorize(req: IdeasRequest) -> Dict[str, Any]:
    c = get_clusterer()
    c.re_embed(req.ideas)
    k, score, labels = c.cluster()
    groups = c.group()
    keywords = c.label_groups_optimized()
    return {
        "best_k": int(k),
        "silhouette": float(score),
        "groups": groups,
        "keywords": keywords,
    }


@app.post("/suggest")
def suggest(req: SuggestRequest) -> Dict[str, str]:
    c = get_clusterer()
    tip = c.improve_idea(req.idea)
    return {"idea": req.idea, "suggestion": tip}

@app.get("/health")
def health():
    return {"status": "ok"}
