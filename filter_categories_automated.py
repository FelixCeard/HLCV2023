import json

# import streamlit as st
import os
from tqdm import tqdm


def get_object_names():
    """return all the folder names in ./dataset"""
    path_to_objects = "./dataset/VisualGenom/objects"
    return os.listdir(path_to_objects)

# def get_object_names():
#     """returns all of the words saved in the dictionary.txt files (each word is on a new line)"""
#     with open('./dataset/dictionary.txt', 'r') as file:
#         return file.read().splitlines()


class App:
    def __init__(self):
        # if 'index' not in st.session_state:
        #     st.session_state.index = 0
        self.current_index = 0
        self.object_names = get_object_names()

        with open('./dataset/dictionary.json', 'r') as file:
            self.dictionary = json.load(file)

        self.max_display_images = 3

        self.accept_file = open('./dataset/accept.txt', 'w')
        self.reject_file = open('./dataset/reject.txt', 'w')

    def __len__(self):
        return len(self.object_names)

    @property
    def disable_previous(self):
        return True if self.current_index == 0 else False

    @property
    def disable_next(self):
        return True if self.current_index == len(self.object_names)-1 else False

    def get_obj_name(self):
        return self.object_names[self.current_index]

    def update_index_callback(self):
        # print(self.current_index)
        pass
        # st.session_state.index = self.current_index

    def next(self):
        self.current_index += 1
        # self.update_index_callback()

    def previous(self):
        self.current_index -= 1
        # self.update_index_callback()

    def accept(self):
        self.next()
        self.accept_file.write(self.get_obj_name() + '\n')

    def reject(self):
        self.next()
        self.reject_file.write(self.get_obj_name() + '\n')

    def get_images(self):
        # get all of the images give a path
        image_paths = []
        for root, dirs, files in os.walk(f"./dataset/VisualGenom/objects/{self.get_obj_name()}"):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
                if len(image_paths) == self.max_display_images:
                    break
        return image_paths

    def in_dictonary(self):
        return self.get_obj_name().upper() in self.dictionary.keys()


if __name__ == '__main__':

    app = App()

    pbar = tqdm(total=len(app))

    while not app.disable_next:
        if app.in_dictonary():
            app.accept()
        else:
            app.reject()
        pbar.update(1)

