import pandas as pd
import spacy
from spacy import displacy
import truecase
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def count_entities(sentence, ner=spacy.load("en_core_web_lg")):
    txt = ner(truecase.get_true_case(sentence))
    entities = [0, 0, 0, 0]
    for word in txt.ents:
        print(word)
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

    text = "did Isle Wight become island"

    count_entities(text)
    print("===============")

    count_entities(truecase.get_true_case(text))
    print("===============")


    x = truecasing_by_sentence_segmentation(text)
    print(x)
    x = truecasing_by_pos(x)
    print(x)
    count_entities(x)
    print("===============")

    
    #ner = spacy.load("en_core_web_sm")
    #df = pd.read_csv('qa.csv')
    #count = 0
    #for i in df.itertuples():
    #    print(i[1])
    #    count_entities(i[1], ner)
    #    print('"""""""""""""""')
    #    count += 1
    #    if count == 12:
    #        break



if __name__ == "__main__":
    main()