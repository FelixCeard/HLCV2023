import json
import os

import torch
from explainn import ExplaiNN
from tqdm import tqdm


def parse_list(line: str):
    line = line.replace('[', '').replace(']', '')
    words = line.split(',')
    words = [w.replace("'", '').strip() for w in words if w != ""]
    return words


if __name__ == '__main__':
    """
    Because we need a knowledge database, I will use ExplaiNN to automatically
    find the best rules to describe the said task
    """

    # 1. read the attributes-image association file
    with open('/home/hlcv_team017/HLCV2023/clean_attributes/lens_attributes_dictionary.txt', 'r') as file:
        content = file.readlines()

    class_attributes = dict()
    for line in tqdm(content, desc='Attributes per class'):
        line = line.split(';')
        id = line[0]
        classes = parse_list(line[1])
        attributes = parse_list(line[2])

        for image_class in classes:
            if image_class not in class_attributes.keys():
                class_attributes[image_class] = []
            class_attributes[image_class].append(set(attributes))

    # 2. Get for each image class the number of unique attributes
    num_attributes = {}
    attribute_order = {}
    max_set = None
    for image_class in tqdm(class_attributes.keys(), desc='Number of attributes per class + order them'):
        total_set: set = class_attributes[image_class][0]
        for i in range(len(class_attributes[image_class])):
            total_set = total_set.union(class_attributes[image_class][i])
        num_attributes[image_class] = len(total_set)
        attribute_order[image_class] = list(total_set)
        if max_set == None:
            max_set = total_set
        else:
            max_set = total_set.union(max_set)

    max_set = list(max_set)

    # 2.5 save the max_set to a file to recover the class names from the ids
    json_max_set = dict()
    for i, img_id in enumerate(max_set):
        json_max_set[i] = img_id
    os.makedirs('/home/hlcv_team017/HLCV2023/clean_attributes/Explainn_extra', exist_ok=True)
    with open('/home/hlcv_team017/HLCV2023/clean_attributes/Explainn_extra/attribute_ids.json', 'w') as file:
        json.dump(json_max_set, file)

    # 3. construct the head and tail matrices
    matrices = dict()
    for line in tqdm(content, desc='Sample to vector_entry'):
        line = line.split(';')
        id = line[0]
        classes = parse_list(line[1])
        attributes = parse_list(line[2])

        for image_class in classes:
            if image_class not in matrices.keys():
                matrices[image_class] = []

            torch_vector = torch.zeros(len(max_set), dtype=torch.int8)

            for attrib in attributes:
                # get index of attribute
                indx = max_set.index(attrib)
                torch_vector[indx] = 1

            matrices[image_class].append(torch.unsqueeze(torch_vector, dim=0))

    # 3.5 concatenate all the matrices
    heads = []
    tails = []
    for i, key in tqdm(enumerate(matrices.keys()), desc='Concatenating (1/2)', total=len(matrices.keys())):
        heads.append(
            torch.cat(matrices[key], dim=0)
        )
        tails.append(
            torch.nn.functional.one_hot(
                torch.tensor([i for _ in range(len(matrices[key]))]),
                num_classes=len(matrices.keys())
            )
        )

    # into one
    print('Concatenating (2/2)')
    head = torch.cat(heads, dim=0)
    tail = torch.cat(tails, dim=0)

    # 4. save to file
    torch.save(head, '/home/hlcv_team017/HLCV2023/clean_attributes/Explainn_extra/head.pt')
    torch.save(tail, '/home/hlcv_team017/HLCV2023/clean_attributes/Explainn_extra/tail.pt')

    # 5. Convert them to ExplaiNN things
    print('Running ExplaiNN')
    explainn = ExplaiNN(num_labels=tail.shape[-1], vebose=True, dir_name='/home/hlcv_team017/HLCV2023/clean_attributes/ExplaiNN')
    explainn.create_dat_file(head=head, tail=tail)
    explainn.create_config_file(pattern_type='label', args={'NOISYHEAD':"true"})
    explainn.run(real_time_shell=True, display_shell=True)
