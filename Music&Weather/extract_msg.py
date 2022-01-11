import pandas as pd

def main():
    df = pd.read_csv('messaging_eval.csv')
    df = df.drop(columns=['domain'])
    parsed = []
    a  = ''
    for parsed_sentence in df.itertuples():
        temp = [parsed_sentence[1], parsed_sentence[2], '']
        if 'IN:' in temp[1]:
            temp[1]= temp[1].split('IN:')[1].split(' ')[0]
            txt = parsed_sentence[2].split('IN:')[1]
            index = txt.find('[')
            temp[2] = txt[index : -1]
            temp[0], temp[1] = temp[1], temp[0]

            parsed.append(temp)
    df = pd.DataFrame(list(parsed),
    columns =['Intent', 'Query', "Param"])
    df.to_csv('messaging.csv')


if __name__ == '__main__':
    main()