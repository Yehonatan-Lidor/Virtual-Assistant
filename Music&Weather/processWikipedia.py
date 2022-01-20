import pandas as pd
import spacy
from spacy import displacy
import truecase
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def count_entities(sentence, ner=spacy.load("en_core_web_lg")):
    txt = ner(truecase.get_true_case(sentence))
    ret = []
    for word in txt.ents:
        if word.label_ == "PERSON":
            ret.append("PERSON(" + str(word) + ")")
        elif word.label_ == "ORG":
            ret.append("ORG(" + str(word) + ")")
        elif word.label_ == "GPE":
            ret.append("GPE(" + str(word) + ")")
        elif word.label_ == "DATE":
            ret.append("DATE(" + str(word) + ")")

    if len(ret) == 0:
        return None
    return ret
from nltk.tokenize import sent_tokenize
import re
def truecasing_by_sentence_segmentation(input_text):
    # split the text into sentences
    sentences = sent_tokenize(input_text, language='english')
    # capitalize the sentences
    sentences_capitalized = [s.capitalize() for s in sentences]
    # join the capitalized sentences
    text_truecase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))
    return text_truecase

def truecasing_by_pos(input_text):
    # tokenize the text into words
    words = nltk.word_tokenize(input_text)
    # apply POS-tagging on words
    tagged_words = nltk.pos_tag([word.lower() for word in words])
    # apply capitalization based on POS tags
    capitalized_words = [w.capitalize() if t in ["NN","NNS"] else w for (w,t) in tagged_words]
    # capitalize first word in sentence
    capitalized_words[0] = capitalized_words[0].capitalize()
    # join capitalized words
    text_truecase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(capitalized_words))
    return text_truecase


def main():
    ner = spacy.load("en_core_web_lg")
    df = pd.read_csv('qa.csv')
    list_ner = []
    rest = []
    for sample in df.itertuples():
        end = count_entities(sample[1], ner)
        if end == None:
            rest.append(sample[1])
        else:
            list_ner.append([sample[1], end])
    df_books = pd.read_csv('books.csv', on_bad_lines='skip')
    books_w_stop = []
    new_rest = []
    for i in rest:
        for j in df_books.itertuples():
                if len(j[2].split(" ")) > 3 and i.lower().find(j[2].lower()) != -1 and i not in books_w_stop:
                    books_w_stop.append([i, j[2]])
                elif i not in rest:
                    new_rest.append(i)

    print(books_w_stop)            
    print(len(books_w_stop))

    print(new_rest)
    print(len(new_rest))







    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")
    print("----------------------------------")

    
        



if __name__ == "__main__":
    main()