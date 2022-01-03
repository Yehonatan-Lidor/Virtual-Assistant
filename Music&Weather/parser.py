import pandas as pd
import nltk
import spacy
from spacy import displacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class parser:
    def __init__(self, df):
        self.df = df
        self.weatherList = []
        self.musicList = []
    def create_table(self):
        data = []
        for i in self.df.itertuples():
            item = []
            if self.isThereExclamationMark(i[3]) == True:
                item.append(1)
            else:
                item.append(0)
            if self.isThereQuestionMark(i[3]) == True:
                item.append(1)
            else:
                item.append(0)
            
            

    def count_entities(self, sentence, ner=spacy.load("en_core_web_sm")):
        txt = ner(sentence)
        entities = [0, 0, 0, 0]
        ENTITIES_LABELS = {"PERSON": 0, "ORG": 1, "GPE": 2, "DATE": 3}
        for word in txt.ents:
            if word.label_ == "PERSON":
                entities[0] +=1
            elif word.label_ == "ORG":
                entities[1] +=1
            elif word.label_ == "GPE":
                entities[2] +=1
            elif word.label_ == "DATE":
                entities[3] +=1
        return entities
    def getMostCommonList(self):
        weather = {}
        music = {}
        for i in self.df.itertuples():
            if i[2] == "GetWeather":
                word_list = self.getFilteredText(i[3])
                for word in word_list:
                    if word in weather.keys():
                        weather[word] += 1
                    else:
                        weather[word] = 1
            elif i[2] == "PlayMusic":
                word_list = self.getFilteredText(i[3])
                for word in word_list:
                    if word in music.keys():
                        music[word] += 1
                    else:
                        music[word] = 1
        temp_music = dict(sorted(music.items(), key=lambda item: item[1]))
        temp_music = list(temp_music)[-24:]
        temp_weather = dict(sorted(weather.items(), key=lambda item: item[1]))
        temp_weather = list(temp_weather)[-24:]

        final_weather = []
        for i in temp_music:
            if i not in temp_weather:
                final_weather.append(i)
        final_music = []
        for i in temp_weather:
            if i not in temp_music:
                final_music.append(i)
        return (final_music, final_weather)
    def getHowManyWords(self, query):
        return len(query.split(' '))
    def getHowManyChars(self, query):
        return len(query)
    def getPartTag(self, query):
        tokens = nltk.word_tokenize(query)
        pos_tagged_tokens = nltk.pos_tag(tokens)
        ret = {}
        for i in pos_tagged_tokens:
            if i[1] not in ret:
                ret[i[1]] = 1
            elif i[1] in ret:
                ret[i[1]] +=1
        return ret
    def getFilteredText(self,query):
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(query)
        
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        
        filtered_sentence = []
        
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
    def isThereQuestionMark(self, query):
        if '?' in query:
            return True
        return False
    def isThereExclamationMark(self, query):
        if '!' in query:
            return True
        return False
    def commonWords(self, query):
        #weather
        countWeather = 0
        for i in query.split(' '):
            if i in self.weatherList:
                countWeather += 1
        #music
        countMusic = 0
        for i in query.split(' '):
            if i in self.musicList:
                countMusic += 1
        return [countWeather, countMusic]
def main():
    l = [1,2,3]
    b = [4,5]
    l += b 
    print(l)
    #df = pd.read_csv('dataset.csv')
    #x = parser(df)
    #x.create_table()

if __name__ == "__main__":
    main()
    



