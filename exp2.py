import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import HiddenMarkovModelTrainer
nltk.download('punkt')
train_data = [
    [('The', 'DT'), ('cat', 'NN'), ('sits', 'VBZ'), ('on', 'IN'), ('the', 'DT'), ('mat', 'NN')],
    [('A', 'DT'), ('dog', 'NN'), ('runs', 'VBZ'), ('fast', 'RB')],
    [('She', 'PRP'), ('is', 'VBZ'), ('reading', 'VBG'), ('a', 'DT'), ('book', 'NN')],
    [('He', 'PRP'), ('plays', 'VBZ'), ('cricket', 'NN')],
    [('Birds', 'NNS'), ('fly', 'VBP'), ('high', 'RB')],
    [('I', 'PRP'), ('love', 'VBP'), ('Python', 'NNP')],
    [('Students', 'NNS'), ('study', 'VBP'), ('hard', 'RB')],
    [('The', 'DT'), ('sun', 'NN'), ('shines', 'VBZ'), ('brightly', 'RB')]
]
trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)
text = input("Enter a sentence: ")
tokens = word_tokenize(text)
tagged_words = hmm_tagger.tag(tokens)
print("\nTokens:")
print(tokens)

# Display POS tags
print("\nPOS Tags (HMM):")
for word, tag in tagged_words:
    print(f"{word:15} -> {tag}")