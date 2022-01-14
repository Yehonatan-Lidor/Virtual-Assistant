import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import pandas as pd
from transformers.utils.dummy_pt_objects import PegasusForCausalLM
def add(text):
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(text)
    return augmented_text

def main():
    count = 0
    df = pd.read_csv('final_dataset.csv')
    final = []
    for i in df.itertuples():
        if i[2] == "GET_MESSAGE":
            count += 1
            final.append([i[2], i[3], i[4]])
            if count < 1300:
                aug_text = add(i[3])
                final.append([i[2], aug_text, i[4]])
        else:
            final.append([i[2], i[3], i[4]])

    df = pd.DataFrame(list(final),
               columns =['Intent', 'Query', "Param"])
    print(df)
    df.to_csv('aug_final_dataset.csv')


if __name__ == "__main__":
    main()
