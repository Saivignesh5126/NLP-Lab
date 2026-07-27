import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score,recall_score,f1_score

nltk.download('punkt')
nltk.download('punkt_tab')
keywords=['threats',"reduces","controls","helps"]
sentence=input("enter biomedical sentence:")
actual=int(input("actual relation (1/0):"))
tokens=word_tokenize(sentence.lower())
print("\ntokens:")
print(tokens)
predicted=0

for word in tokens:
    if word in keywords:
        predicted=1
print("\nPredicted Relation:",predicted)
y_true=[actual]
y_pred=[predicted]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)