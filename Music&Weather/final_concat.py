import pandas as pd

def main():
    df = pd.read_csv('qa.csv')
    df1 = pd.read_csv('dataset1.csv')
    final_list = []
    for i in df.itertuples():
        final_list.append(["SEARCH", i[1], "NONE"])

    for i in df1.itertuples():
        final_list.append([i[2], i[3], i[4]])
    df = pd.DataFrame(list(final_list),
               columns =['Intent', 'Query', "Param"])

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('final_dataset.csv')


if __name__ == "__main__":
    main()