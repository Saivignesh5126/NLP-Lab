import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

docs = []

n = int(input(&quot;Enter number of documents: &quot;))
for i in range(n):
docs.append(input(&quot;Enter document: &quot;))

query = input(&quot;\nEnter search query: &quot;)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

query_vec = vectorizer.transform([query])

scores = cosine_similarity(query_vec, X)

print(&quot;\nTF-IDF Similarity Scores:&quot;)
for i, s in enumerate(scores[0]):
print(&quot;Document&quot;, i+1, &quot;:&quot;, round(s, 3))

svd = TruncatedSVD(n_components=2)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

lsa_scores = cosine_similarity(query_lsa, X_lsa)

print(&quot;\nLSA Similarity Scores:&quot;)
for i, s in enumerate(lsa_scores[0]):
print(&quot;Document&quot;, i+1, &quot;:&quot;, round(s, 3))

best = np.argmax(lsa_scores)
print(&quot;\nMost Relevant Document:&quot;)
print(docs[best])
