from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import time



class IdeaClusterer:
    def __init__(self,ideas, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.ideas = ideas
        self.vectors = self.model.encode(
            ideas,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def cluster(self):
        n = len(self.ideas)
        k_max = min(10, n-1)

        best_k = 0
        best_score = -1.0
        best_labels = []
        for i in range(2,k_max + 1):
            kmeansObject = KMeans(n_clusters=i, random_state=42, n_init="auto")
            labels = kmeansObject.fit_predict(self.vectors)
            score = silhouette_score(self.vectors, labels)
            if score > best_score:
                best_k = i
                best_score = score
                best_labels = labels
            print(f"k={i}, silhouette={score:.4f}")


        return best_k, best_score, best_labels





"""
start = time.time()  # start timer

end = time.time()    # end timer
print(f"Execution time: {end - start:.4f} seconds")

"""
with open("data/ideas.txt") as f:
    ideaList = [line.strip() for line in f if line.strip()]


test = IdeaClusterer(ideaList)



# ignore
print(test.cluster())
