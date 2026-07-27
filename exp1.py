""" words stemming 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Get user input
text = input("Enter a sentence: ")

# Tokenization
tokens = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Display results
print("\nOriginal Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)

print("\nLemmatized Words:")
print(lemmatized_words)

# Comparison
print("\nComparison:")
print("Stemming reduces words to root forms, which may not always be meaningful.")
print("Lemmatization converts words to meaningful base forms using vocabulary knowledge.")"""

import nltk
from nltk.tokenize import sent_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')

# User input
text = input("Enter a paragraph: ")

# Sentence Tokenization
sentences = sent_tokenize(text)

# Display results
print("\nOriginal Text:")
print(text)

print("\nTokenized Sentences:")
for i, sentence in enumerate(sentences, start=1):
    print(f"{i}. {sentence}")