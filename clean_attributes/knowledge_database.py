from collections import Counter

import pandas as pd
from tqdm import tqdm


def parse_list(line: str):
    line = line.replace('[', '').replace(']', '')
    words = line.split(',')
    words = [w.replace("'", '').strip() for w in words if w != ""]
    return words


if __name__ == '__main__':
    # process input
    with open('lens_attributes_dictionary.txt', 'r') as file:
        content = file.readlines()

    class_attributes = dict()
    for line in tqdm(content):
        line = line.split(';')
        id = line[0]
        classes = parse_list(line[1])
        attributes = parse_list(line[2])

        for image_class in classes:
            if image_class not in class_attributes.keys():
                class_attributes[image_class] = []
            class_attributes[image_class].append(set(attributes))

    # get the union of all attributes per class
    for image_class in class_attributes.keys():
        total_set: set = class_attributes[image_class][0]
        total_list = []
        for i in range(len(class_attributes[image_class])):
            # total_set = total_set.intersection(class_attributes[image_class][i])
            total_list += list(class_attributes[image_class][i])
        count = Counter(total_list)

        dataset = []
        for key, value in count.items():
            dataset.append({'name': str(key), 'num': value})

        df = pd.DataFrame(dataset)
        df.to_csv(f'{image_class}.csv')
        # sns.histplot(data=df, x='name', y='num')
        # plt.show()