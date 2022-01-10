import pandas as pd
import nltk
import spacy
from spacy import displacy
import truecase



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

class parser:
    def __init__(self, df):
        self.df = df
        self.weatherList = []
        self.musicList = []
    def create_table(self):
        x = self.getMostCommonList()
        self.weatherList = x[1]
        self.musicList = x[0]
        print(x[0])
        data = []
        ner = spacy.load("en_core_web_sm")
        for i in self.df.itertuples():

            item = []
            item.append(i[3])
            #ExclamationMark
            if self.isThereExclamationMark(i[3]) == True:
                item.append(1)
            else:
                item.append(0)
            #QuestionMark
            if self.isThereQuestionMark(i[3]) == True:
                item.append(1)
            else:
                item.append(0)
            #countWords
            item.append(self.getHowManyWords(i[3]))
            #countChars
            item.append(self.getHowManyChars(i[3]))
            #questions 
            item += self.check_question(i[3])
            #count am-is-are
            item.append(self.count_am_is_are(i[3]))
            #count stop words
            item.append(self.count_stop_words(i[3]))
            #count enteties
            item += self.count_entities(i[3], ner)
            #common words
            item += self.commonWords(i[3])

            #part tag:
            tags = self.getPartTag(i[3])
            #add verb
            item.append(tags["VRB"])
            #add adj
            item.append(tags["ADJ"])
            #add noun
            item.append(tags["NON"])

            item.append(i[2])

            data.append(item)
        return data
    def check_question(self,sentence):
        QUESTIONS = ("how", "when", "where", "what", "whose", "which", "why", "who")
        question_list = [] 
        sentence = sentence.lower()
        for i in range(len(QUESTIONS)):
            question_list.append(int(QUESTIONS[i] in sentence))
        return question_list
    def count_am_is_are(self, sentence):
        WORDS = ("am", "is", "are", "s", "m")
        sentence_lst = sentence.split(' ')
        count = 0
        for word in sentence_lst:
            if word in WORDS:
                count += 1
        return count
    def count_stop_words(self, sentence):
        return sum([word in stopwords.words('english') for word in sentence.split(' ')])
    def count_entities(self, sentence, ner=spacy.load("en_core_web_sm")):
        txt = ner(truecase.get_true_case(sentence))
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
        VRB = "VBZ,VBP,VBN,VBD,VBG,VB"
        ADJ = "JJ,JJR,JJS,RB,RBR,RBS"
        NON = "NN,NNS,NNP,NNPS"
        tokens = nltk.word_tokenize(query)
        pos_tagged_tokens = nltk.pos_tag(tokens)
        ret = {'VRB': 0, 'ADJ': 0, "NON": 0}
        for i in pos_tagged_tokens:
            if i[1] in VRB:
                ret['VRB'] += 1
            elif i[1] in ADJ:
                ret['ADJ'] += 1
            elif i[1] in NON:
                ret['NON'] += 1
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
    df = pd.read_csv('dataset.csv')
    x = parser(df)
    list_frame = x.create_table()

    df = pd.DataFrame(list(list_frame),
            columns =["query", 'ExclamationMark', 'QuestionMark', "countWords", "countChars", 
               "how", "when", "where", "what", "whose", "which", "why", "who",
               "count am-is-are", "count stop words", "PERSON", "ORG", "GPE", "DATE", "count music common", "count weather common" , "VRB", "ADJ","NON", "output"])
    df.to_csv('features.csv')

if __name__ == "__main__":
    main()
    



