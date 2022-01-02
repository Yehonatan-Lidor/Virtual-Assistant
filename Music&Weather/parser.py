import pandas as pd

import nltk
from nltk.corpus import stopwords

import spacy
from spacy import displacy

class parser:
    def __init__(self, df):
        self.df = df
    def create_tabel(self):
        for i in self.df.itertuples():
            print(i[3])
            print(self.getHowManyWords(i[3]))
    def getMostCommonList(self):
        weather = {}
        music = {}
        for i in self.df.itertuples():
            if i[2] == "GetWeather":
                word_list = i[3].split(' ')
                for word in word_list:
                    if word in weather.keys():
                        weather[word] += 1
                    else:
                        weather[word] = 1
            elif i[2] == "PlayMusic":
                word_list = i[3].split(' ')
                for word in word_list:
                    if word in music.keys():
                        music[word] += 1
                    else:
                        music[word] = 1
        return (music, weather)
    def getHowManyWords(self, query):
        return len(query.split(' '))

    def check_question(sentence):
        QUESTIONS = ("how", "when", "where", "what", "whose", "which", "why", "which")
        question_list = [] 
        sentence = sentence.lower()
        for i in range(len(QUESTIONS)):
            question_list.append(QUESTIONS[i] in sentence)
        return question_list

    def count_am_is_are(sentence):
        WORDS = ("am", "is", "are", "s", "m")
        sentence_lst = sentence.split(' ')
        count = 0
        for word in sentence_lst:
            if word in WORDS:
                count += 1
        return count

    def count_stop_words(sentence):
        return sum([word in stopwords.words('english') for word in sentence.split(' ')])
        
    def count_entities(sentence, ner=spacy.load("en_core_web_sm")):
        txt = ner(sentence)
        entities = [0, 0, 0]
        ENTITIES_LABELS = {"PERSON": 0, "ORG": 1, "GPE": 2}
        for word in txt:
            entities[word.label_] += 1
        return entities



def main():
    df = pd.read_csv('dataset.csv')
    x = parser(df)
    print(x.create_tabel())

if __name__ == "__main__":
    main()
    

#print(dict(sorted(music.items(), key=lambda item: item[1])))
#print(dict(sorted(weather.items(), key=lambda item: item[1])))

