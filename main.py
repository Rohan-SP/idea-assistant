# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from clustering import IdeaClusterer  # put your class in idea_clusterer.py

app = FastAPI()

# Request/Response models
class IdeasRequest(BaseModel):
    ideas: List[str]

class SuggestRequest(BaseModel):
    idea: str

@app.on_event("startup")
def load_model():
    # initialize once
    global clusterer
    clusterer = IdeaClusterer([])

@app.post("/categorize")
def categorize(req: IdeasRequest) -> Dict[str, Any]:
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
    tip = clusterer.improve_idea(req.idea)
    return {"idea": req.idea, "suggestion": tip}
