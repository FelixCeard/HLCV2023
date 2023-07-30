import enchant
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import json

def parse_list(line:str):
    line = line.replace('[', '').replace(']', '')
    words = line.split(',')
    words = [w.replace("'", '').strip() for w in words if w != ""]
    return words

if __name__ == '__main__':
    # load the original attributes to get the "class"
    with open('./../data/attributes_cleaned.txt', 'r') as file:
        content_og = file.readlines()

    unique_attributes = set()
    duplicate_count = dict()

    id_to_class = dict()
    fails = 0
    for line in content_og:
        try:
            line = line.split(';')
            image_id = line[0].strip()
            class_id = line[1].strip()
            id_to_class[image_id] = class_id
        except Exception as e:
            fails += 1
    print('Failed to get the class to', fails, 'images')

    # load the extracted lens attributes
    with open('./../data/lens_attributes.txt', 'r') as file:
        content = file.readlines()

    # load english dictionary
    d = enchant.Dict("en_US")

    dataset = []

    output_file = open('./lens_attributes_dictionary.txt', 'w')
    fails = 0
    for line in tqdm(content):
        exploded = line.split(';')

        image_id = exploded[0]
        captions = exploded[2]
        attributes = exploded[3]

        # parse string list into python list
        parsed = parse_list(attributes)


        # words_in_dictionary = [w for w in parsed if d.check(w)]
        # num_attributes_dictionary = 0
        # num_attributes_split = 0
        number_of_attributes = 0

        words_in_dictionary = []
        for w in parsed:
            if d.check(w):
                words_in_dictionary.append(w)
                # num_attributes_dictionary += 1
            else:
                last_word = w.split(' ')[-1]
                if d.check(last_word):
                    words_in_dictionary.append(f'{last_word}')
            #         num_attributes_split += 1
        # remove duplicates
        words_in_dictionary = list(set(words_in_dictionary))
        for w in words_in_dictionary:
            lbefore = len(unique_attributes)
            unique_attributes.add(w)
            lafter = len(unique_attributes)
            if lbefore == lafter:
                # duplicate
                if w not in duplicate_count.keys():
                    duplicate_count[w] = 0
                duplicate_count[w] += 1

        number_of_attributes = len(words_in_dictionary)
        try:
            # output_file.write(
            #     f'{image_id}; {id_to_class.get(image_id)}; {words_in_dictionary} \n'
            # )
            dataset.append({
                'num_attr': number_of_attributes
            })

        except Exception as e:
            fails += 1

    print('Failed to trim the attributes of', fails, 'images')

    df = pd.DataFrame(dataset)

    print('num_attr')
    print('mean:', df['num_attr'].mean(), ', std:', df['num_attr'].std(), ', min:', df['num_attr'].min(), ', max:', df['num_attr'].max())
    # print('num_total')
    # print('mean:', df['num_total'].mean(), ', std:', df['num_total'].std(), ', min:', df['num_total'].min(), ', max:', df['num_total'].max())
    # print('num_split')
    # print('mean:', df['num_split'].mean(), ', std:', df['num_split'].std(), ', min:', df['num_split'].min(), ', max:', df['num_split'].max())
    # print('num_dictionary')
    # print('mean:', df['num_dictionary'].mean(), ', std:', df['num_dictionary'].std(), ', min:', df['num_dictionary'].min(), ', max:', df['num_dictionary'].max())

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('num_attr')

    df['num_attr'].hist(ax=ax)
    # df['num_split'].hist(ax=ax[1])
    # df['num_dictionary'].hist(ax=ax[2])
    fig.savefig('./histograms_n.png', dpi=200)
    plt.close(fig)

    with open('unique_attributes.txt', 'w') as file:
        file.write('\n'.join(list(unique_attributes)))

    print('Number of DUPLICATE attributes', len(duplicate_count))
    plt.hist(list(duplicate_count.values()))
    plt.plot()
    plt.savefig('duplicates.png', dpi=200)

    with open('duplicates.json', 'w') as file:
        json.dump(duplicate_count, file, indent=4)

    # only keep attribute that are duplicates
    fails = 0
    removed = 0
    num_attributes = []
    for line in tqdm(content, desc='Only keeping duplicate attributes'):
        exploded = line.split(';')

        image_id = exploded[0]
        captions = exploded[2]
        attributes = exploded[3]

        # parse string list into python list
        parsed = parse_list(attributes)


        # words_in_dictionary = [w for w in parsed if d.check(w)]
        # num_attributes_dictionary = 0
        # num_attributes_split = 0
        number_of_attributes = 0

        words_in_dictionary = []
        for w in parsed:
            if d.check(w):
                words_in_dictionary.append(w)
                # num_attributes_dictionary += 1
            else:
                last_word = w.split(' ')[-1]
                if d.check(last_word):
                    words_in_dictionary.append(f'{last_word}')
            #         num_attributes_split += 1
        # remove duplicates
        words_in_dictionary = list(set(words_in_dictionary))
        words_in_dictionary = [w for w in words_in_dictionary if duplicate_count.get(w) is not None]

        if len(words_in_dictionary) == 0:
            removed += 1
            continue

        num_attributes.append(len(words_in_dictionary))

        try:
            output_file.write(
                f'{image_id}; {id_to_class.get(image_id)}; {words_in_dictionary} \n'
            )

        except Exception as e:
            fails += 1


    print('Removed', removed, 'Images due to them not having duplicate attributes')
    print('Could not parse', fails, 'Images')

    plt.hist(num_attributes)
    plt.gca().set_title('Only duplicates')
    plt.plot()
    plt.savefig('only_duplicates.png', dpi=200)

