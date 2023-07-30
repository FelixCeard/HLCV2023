import enchant
from tqdm import tqdm

if __name__ == '__main__':
    # load the extracted lens attributes
    with open('unique_lens_attributes.txt', 'r') as file:
        content = file.readlines()

    # load english dictionary
    d = enchant.Dict("en_US")

    # only keep words in the english dictionary
    word_in_dictionary = []
    word_not_in_dictionary = []
    for line in tqdm(content):
        line = line.strip().replace('\n', '')  # remove white leading/trailing white spaces

        # only keep the last word if the full word is not in the dictionary
        # e.g. plastic bottle -> bottle

        if d.check(line):
            word_in_dictionary.append(line)
        else:
            last_word = line.split(' ')[-1]

            if d.check(last_word):
                word_in_dictionary.append(f'{last_word} == {line}')
            else:
                word_not_in_dictionary.append(line)


    # remove duplicates
    word_in_dictionary = list(set(word_in_dictionary))
    with open('cleaned_dictionary.txt', 'w') as file:
        file.write('\n'.join(word_in_dictionary))


    with open('trash_dictionary.txt', 'w') as file:
        file.write('\n'.join(word_not_in_dictionary))

    print('Done')
    print('Number of unique attributes', len(content), '/', len(word_in_dictionary))
    print('Kept', 100.0 * len(word_in_dictionary)/len(content))