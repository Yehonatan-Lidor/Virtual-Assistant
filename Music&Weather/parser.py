import pandas as pd

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
def main():
    df = pd.read_csv('dataset.csv')
    x = parser(df)
    print(x.create_tabel())

if __name__ == "__main__":
    main()
    

#print(dict(sorted(music.items(), key=lambda item: item[1])))
#print(dict(sorted(weather.items(), key=lambda item: item[1])))

