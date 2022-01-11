import pandas as pd
import nltk
import spacy
from spacy import displacy
import truecase



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class parser:
    def __init__(self, df):
        self.df = df
        self.weatherList = []
        self.musicList = []
        self.getList = []
        self.sendList = []
    def tests(self): 
        x = self.getMostCommonList()
        print(x)
        print('""""""""""""""""""""')
        print(x[0])
        print('""""""""""""""""""""')
        print(x[1])
        print('""""""""""""""""""""')
        print(x[2])
        print('""""""""""""""""""""')
        print(x[3])
        print('""""""""""""""""""""')

    def create_table(self):
        x = self.getMostCommonList()
        self.getList = x[2]
        self.weatherList = x[1]
        self.musicList = x[0]
        self.sendList = x[3]

        data = []
        ner = spacy.load("en_core_web_lg")
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

            #count tell and going
            val = i[3].lower().count('tell') + i[3].lower().count('going')
            item.append(val)

            #count text and message and please and see
            val = i[3].lower().count('tell') + i[3].lower().count('going') + i[3].lower().count('message') + i[3].lower().count('see')
            item.append(val)


            #output
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
        get = {}
        send = {}
        search = {}

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
            elif i[2] == "GET_MESSAGE":
                word_list = self.getFilteredText(i[3])
                for word in word_list:
                    if len(word) > 1 and not word == "'s":
                        if word.lower() in get.keys():
                            get[word.lower()] += 1
                        else:
                            get[word.lower()] = 1
            elif i[2] == "SEND_MESSAGE":
                word_list = self.getFilteredText(i[3])
                for word in word_list:
                    if len(word) > 1 and not word == "'s":
                        if word.lower() in send.keys():
                            send[word.lower()] += 1
                        else:
                            send[word.lower()] = 1
            elif i[2] == "SEARCH":
                word_list = self.getFilteredText(i[3])
                for word in word_list:
                    if len(word) > 1 and not word == "'s":
                        if word.lower() in search.keys():
                            search[word.lower()] += 1
                        else:
                            search[word.lower()] = 1                        
        temp_music = dict(sorted(music.items(), key=lambda item: item[1]))
        temp_music = list(temp_music)[-24:]
        temp_weather = dict(sorted(weather.items(), key=lambda item: item[1]))
        temp_weather = list(temp_weather)[-24:]
        temp_get = dict(sorted(get.items(), key=lambda item: item[1]))
        temp_get = list(temp_get)[-24:]
        temp_send = dict(sorted(send.items(), key=lambda item: item[1]))
        temp_send = list(temp_send)[-24:]
        temp_search = dict(sorted(search.items(), key=lambda item: item[1]))
        temp_search = list(temp_search)[-24:]

        final_music = []
        for i in temp_music:
            if i not in temp_get and  i not in temp_weather and i not in temp_send:
                final_music.append(i)

        final_weather = []
        for i in temp_weather:
            if i not in temp_get and  i not in temp_music and i not in temp_send:
                final_weather.append(i)


        final_get = []
        for i in temp_get:
            if i not in temp_music and  i not in temp_weather and i not in temp_send:
                final_get.append(i)
        
        final_send = []
        for i in temp_send:
            if i not in temp_get and  i not in temp_weather and i not in temp_music:
                final_send.append(i)


        return [final_music, final_weather, final_get , final_send]
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
        #remove punck
        words = nltk.word_tokenize(query.lower())
        word_tokens= [word for word in words if word.isalnum()]
        #remove stop words
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        wnl = WordNetLemmatizer()
        lemmatized_words = []
        for word in filtered_sentence:
            lemmatized_words.append(wnl.lemmatize(word))
            
        return lemmatized_words
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
        #send
        countSend = 0
        for i in query.split(' '):
            if i in self.sendList:
                countSend += 1
        #get
        countGet = 0
        for i in query.split(' '):
            if i in self.getList:
                countGet += 1
        return [countMusic, countWeather, countGet, countSend]
def main():
    df = pd.read_csv('final_dataset.csv')
    x = parser(df)
    list_frame = x.create_table()

    df = pd.DataFrame(list(list_frame),
            columns =["query", 'ExclamationMark', 'QuestionMark', "countWords", "countChars", 
               "how", "when", "where", "what", "whose", "which", "why", "who",
               "count am-is-are", "count stop words", "PERSON", "ORG", "GPE", "DATE", "count music common", "count weather common" ,"count get common", "count send common" , "VRB", "ADJ","NON","common_t_g","common_s_p_m_t", "output"])
    df.to_csv('features.csv')

if __name__ == "__main__":
    main()
    



