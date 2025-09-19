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
    def label_groups(self):
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


        for nthCluster in range(self.cluster_count):
            print(unique_words_all_clusters[nthCluster])
            print(len(unique_words_all_clusters[nthCluster]))

    # Will need the list of all unique words in the 
    def tf_idf():
        pass
                
    
              


"""
start = time.time()  # start timer

end = time.time()    # end timer
print(f"Execution time: {end - start:.4f} seconds")

"""

with open("data/ideas.txt") as f:
    ideaList = [line.strip().lower() for line in f if line.strip()]

# This is assuming that the data from ideaList has been cleaned and is ready to be inputted, for testing purposes it provides a sterile data input

test = IdeaClusterer(ideaList)

test.cluster()

test.label_groups()

print("done")


# ignore
# print(test.cluster())
