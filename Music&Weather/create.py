import pandas as pd





def main():
    #load data
    filter_one = []
    file_labels = open("label.txt", "r")
    file_in = open("in.txt", "rb")
    file_out = open("out.txt", "rb")

    labels_content = file_labels.read().replace("\r", "").split('\n')
    file_labels.close()

    in_content = file_in.read().decode().replace("\r", "").split('\n')
    file_in.close()
    out_content = file_out.read().decode().replace("\r", "").split('\n')
    file_out.close()

    #clean data with 2 or more #
    for i, x, z in zip(labels_content, in_content, out_content):
        if i.count("#") < 2:
            filter_one.append([i,x,z])
        
    filter_two = []
    #split data with # - to one ask
    for item in filter_one:
        #if item does not have #
        if "#" not in item[0]:
            items_in = item[1].strip().split(' ')
            items_out = item[2].strip().split(' ')
            new_out = []
            for i in range(len(items_out)):
                if items_out[i] != "O":
                    new_out.append(items_out[i]+ "(" + items_in[i] +"),")
            filter_two.append([item[0], item[1],item[2]," ".join(new_out)])
        else:
            if "and then" in item[1].lower():
                new_label = item[0].split('#')
                new_in = item[1].lower().split('and then')
                new_in[0] = new_in[0].strip()
                new_in[1] = new_in[1].strip()
                size = len(new_in[0].split(' '))
                list_words = new_in[0].split(' ') + ['O', '0'] + new_in[1].split(' ')
                out_options = item[2].strip().split(' ')
                first = []
                second = []
                f = []
                s = []
                for i in range(len(out_options)):
                    if i < size:
                        f.append(out_options[i])
                        if out_options[i] != 'O':
                            first.append(out_options[i] + "(" + list_words[i]+ "),")
                    if i > size and i < len(list_words):
                        s.append(out_options[i])
                        if out_options[i] != 'O':
                            second.append(out_options[i] + "(" + list_words[i]+ "),")
                end_item_first = [new_label[0],new_in[0]," ".join(f), " ".join(first)]
                end_item_second = [new_label[1],new_in[1]," ".join(s), " ".join(second)]
                filter_two.append(end_item_first)
                filter_two.append(end_item_second)
    #filter 3 - delete items with the wrong intent
    filter_three = []
    for item in filter_two:
        if item[0] == "PlayMusic" or item[0] == "GetWeather":
            filter_three.append(item)
    count = 0
    for i in filter_three:
        if i[0] == "GetWeather":
            count += 1
    print(count)
    print(len(filter_three)- count)
    """
    [
        [0,1,2,3],
        [6,4,2,4]
    

    ]
    """
    df = pd.DataFrame(list(filter_three),
               columns =['Intent', 'Query', "Param1", "Param2"])
    print(df)
    df.to_csv('param1.csv')
                

if __name__ == "__main__":
    main()

