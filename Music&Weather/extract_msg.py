import pandas as pd
import re

def main():
    df = pd.read_csv('messaging_eval.csv')
    df = df.drop(columns=['domain'])
    parsed = []
    exp = re.compile('\[(.*?)\]')
    for parsed_sentence in df.itertuples():
        temp = [parsed_sentence[1], parsed_sentence[2], '']
        if 'IN:' in temp[1]:
            temp[1]= temp[1].split('IN:')[1].split(' ')[0]
            txt = parsed_sentence[2].split('IN:')[1]
            index = txt.find('[')
            temp[2] = txt[index : -1]
            temp[0], temp[1] = temp[1], temp[0]
            if temp[0] == 'GET_MESSAGE' or temp[0] == 'SET_MESSAGE':
                param = exp.findall(temp[2])
                param = [ [i.split(' ')[0], ' '.join(i.split(' ')[1: -1])  ] for i in param]
                param2 = ' '.join(  ['{}({})'.format(i[0], i[1]) for i in param] )

                query = temp[1]
                for i in param:
                    query = query.replace(i[1], i[0])
                
                query = query.split(' ')
                for i in range(len(query)):
                    if 'SL:' not in query[i]:
                        query[i] = 'O'
                query = ' '.join(query)
                item = [temp[0], temp[1], query, param2]
                print(item)
                parsed.append(item)
    print(param_types)
    df = pd.DataFrame(list(parsed),
    columns =['Intent', 'Query', "Param1", "Param2"])
    df.to_csv('messaging.csv')


if __name__ == '__main__':
    main()