import pandas as pd

def main():
    df_m1 = pd.read_csv('messaging.csv')
    df_m2 = pd.read_csv('messaging1.csv')
    df_m3 = pd.read_csv('messaging2.csv')


    final_list= []
    for i in df_m1.itertuples():
        
        if i[2] == "GET_MESSAGE":
            final_list.append([i[2], i[3], i[4]])

    for i in df_m2.itertuples():
        if i[2] == "GET_MESSAGE":
            final_list.append([i[2], i[3], i[4]])

    for i in df_m3.itertuples():
        if i[2] == "GET_MESSAGE":
            final_list.append([i[2], i[3], i[4]])

    count = 0
    for i in df_m2.itertuples():
        if count == 4000:
            break
        if i[2] == "SEND_MESSAGE":
            count += 1
            final_list.append([i[2], i[3], i[4]])
    #connect with the original database
    df_main = pd.read_csv('dataset.csv')
    temp = []
    for i in df_main.itertuples():
        final_list.append([i[2], i[3], i[4]])
    df = pd.DataFrame(list(final_list),
               columns =['Intent', 'Query', "Param"])

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('final_dataset.csv')
    

 




if __name__ == "__main__":
    main()