import math
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt



class IdeaClusterer:
    def __init__(self,ideas, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.ideas = ideas
        self.vectors = self.model.encode(
            ideas,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    def re_embed(self, new_ideas):
        self.ideas = new_ideas
        self.vectors = self.model.encode(
            new_ideas,
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
        self.labels = best_labels
        self.cluster_count = best_k
        return best_k, best_score, best_labels
    
    def group(self):
        output = {}

        for nthCluster in range(self.cluster_count):
            output[nthCluster] = []

        for i in range(len(self.labels)):
            output[self.labels[i]].append(self.ideas[i])

        return output
    

    
    # this is essentially a 
    def label_groups(self, top_n=3):
        all_clusters = self.group()

        self.label_names = {}
        
        # Get all unique words from all the clusters
        unique_words_all_clusters = {}

        # Populate the unique word dict
        for nthCluster in range(self.cluster_count):
            unique_words_all_clusters[nthCluster] = []

        # Loop through each cluster
        for i in range(self.cluster_count):

            # Loop through each doc in the cluster
            for doc in all_clusters[i]:

                # Sperate the words in each doc
                doc_words = doc.split(" ")
                
                # Then for each word in the doc if it hasn't been seen add it to it's correspondning cluster unique word list
                for word in doc_words:
                    if word not in unique_words_all_clusters[i]:
                        unique_words_all_clusters[i].append(word)

        tf_score = {}
        idf_score = {}
        tfidf_scores = {} 
        # create the map of matrices 
        # and create the output dict of int:list[scores]
        for nthCluster in range(self.cluster_count):
            tf_score[nthCluster] = []
            idf_score[nthCluster] = []
            tfidf_scores[nthCluster] = {}

            for nth_word in range(len(unique_words_all_clusters[nthCluster])):
                tf_score[nthCluster].append([])
                idf_score[nthCluster].append(0)

                for doc in range(len(all_clusters[nthCluster])):
                    tf_score[nthCluster][nth_word].append(self.tf_calc(unique_words_all_clusters[nthCluster][nth_word], all_clusters[nthCluster][doc]))

                    # IDF logic
                    doc_words = all_clusters[nthCluster][doc].split()   # split once
                    if unique_words_all_clusters[nthCluster][nth_word] in doc_words:
                        idf_score[nthCluster][nth_word] += 1
                idf_score[nthCluster][nth_word] = math.log((len(all_clusters[nthCluster]) + 1) / (1 + idf_score[nthCluster][nth_word])) + 1

                # Combine TF and IDF → TF–IDF per doc for this word
                idf_val = idf_score[nthCluster][nth_word]
                tfidf_vals = [tf_val * idf_val for tf_val in tf_score[nthCluster][nth_word]]

                # (optional) aggregate; average is a good default for snippet-length docs
                N = len(all_clusters[nthCluster]) or 1
                avg_tfidf = sum(tfidf_vals) / N

                # store per-word score for this cluster
                word = unique_words_all_clusters[nthCluster][nth_word]
                tfidf_scores[nthCluster][word] = avg_tfidf
            
            # logic to put get the words that have the highest score
            sorted_words = sorted(tfidf_scores[nthCluster].items(),key=lambda x: x[1], reverse=True)
            self.label_names[nthCluster] = [w for w, _ in sorted_words[:top_n]]
            #print(unique_words_all_clusters[nthCluster])
            #print(len(unique_words_all_clusters[nthCluster]))

        # print(self.label_names)
        return self.label_names
        # Will need the list of all unique words in the 


    def label_groups_optimized(self, top_n=3):
        all_clusters = self.group()
        self.label_names = {}

        for nthCluster, docs in all_clusters.items():
            if not docs:
                self.label_names[nthCluster] = []
                continue

            # Use sklearn's TF–IDF with English stopwords removed
            vectorizer = TfidfVectorizer(stop_words="english")
            X = vectorizer.fit_transform(docs)

            # Average TF–IDF score across all docs in this cluster
            avg_scores = X.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()

            # Sort terms by average score
            top_indices = avg_scores.argsort()[::-1][:top_n]
            top_terms = [terms[i] for i in top_indices]

            self.label_names[nthCluster] = top_terms

        return self.label_names
    def tf_calc(self, word, doc):
        words = doc.split()
        return words.count(word) / len(words) if len(words) > 0 else 0 
    
              


"""
start = time.time()  # start timer

end = time.time()    # end timer
print(f"Execution time: {end - start:.4f} seconds")

"""

with open("data/my_ideas.txt", "r", encoding="utf-8", errors="ignore") as f:
    ideaList = [line.strip().lower() for line in f if line.strip()]


# This is assuming that the data from ideaList has been cleaned and is ready to be inputted, for testing purposes it provides a sterile data input

test = IdeaClusterer(ideaList)

test.cluster()

test.label_groups_optimized()

reducer = umap.UMAP(n_neighbors=25, min_dist=0.5, random_state=42)

embedding_2d = reducer.fit_transform(test.vectors)

# Make sure clusters are labeled
labels = np.array(test.labels)
cluster_names = test.label_names  # already generated by test.label_groups()

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=labels,
    cmap="tab10",
    s=25,
    alpha=0.7
)

# Add cluster names at centroids
for cluster_id in range(test.cluster_count):
    mask = labels == cluster_id
    cluster_points = embedding_2d[mask]
    centroid = cluster_points.mean(axis=0)

    keywords = ", ".join(cluster_names.get(cluster_id, []))

    plt.text(
        centroid[0],
        centroid[1],
        keywords,
        fontsize=10,
        weight="bold",
        ha="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2)
    )

plt.title("UMAP Visualization of Idea Clusters", fontsize=14)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()