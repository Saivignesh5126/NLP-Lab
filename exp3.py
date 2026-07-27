import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Download WordNet data
nltk.download('wordnet')


# Input headlines
headlines = []

n = int(input("Enter number of headlines: "))

for i in range(n):
    headline = input(f"Enter headline {i+1}: ")
    headlines.append(headline)

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(X)

print("\nCosine Similarity Matrix:")
print(similarity_matrix)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

print("\nHeadline Clusters:")
for i in range(len(headlines)):
    print(f"{headlines[i]} -> Cluster {kmeans.labels_[i]}")

# WordNet Similarity
w1 = input("\nEnter first word: ")
w2 = input("Enter second word: ")

s1 = wn.synsets(w1)
s2 = wn.synsets(w2)

if s1 and s2:
    similarity = s1[0].path_similarity(s2[0])
    print("WordNet Similarity:", similarity)
else:
    print("Similarity not found.")