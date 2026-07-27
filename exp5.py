import nltk
from nltk import word_tokenize,pos_tag
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
text=input("enter legal text")
tokens=word_tokenize(text)
tags=pos_tag(tokens)
print("\n Detected named entites:")
count=0
for word,tag in tags:
    if tag=="NNP":
        print(word,"->ENTITY")
        count+=1
actual=int(input("\nenter actual number of entities:"))
accuracy=(min(count,actual)/max(count,actual))*100
print("\nPredicted Entities:",count)
print("NER accuracy:",round(accuracy,2),"%")