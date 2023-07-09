import streamlit as st
import os

def get_object_names():
    """returns all of the words saved in the dictionary.txt files (each word is on a new line)"""
    with open('./dataset/accept.txt', 'r') as file:
        return file.read().splitlines()

class App:
    def __init__(self):
        if 'index' not in st.session_state:
            st.session_state.index = 0
        self.current_index = st.session_state.index
        self.object_names = get_object_names()

        self.max_display_images = 3

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
        print(self.current_index)
        st.session_state.index = self.current_index

    def next(self):
        self.current_index += 1
        self.update_index_callback()

    def previous(self):
        self.current_index -= 1
        self.update_index_callback()

    def accept(self):
        self.current_index += 1
        self.update_index_callback()
        # todo

    def reject(self):
        self.current_index += 1
        self.update_index_callback()
        # todo

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




if __name__ == '__main__':

    cstm_app = App()

    c1 = st.container()
    c1L, c1R = c1.columns(2)
    previous = c1L.button("Previous", disabled=cstm_app.disable_previous, on_click=cstm_app.previous)
    next = c1R.button("Next", disabled=cstm_app.disable_next, on_click=cstm_app.next)

    c2 = st.container()
    c2L, c2R, = c2.columns(2)
    c2RL = c2.columns(8)
    c2L.markdown(f"## Object: `{cstm_app.get_obj_name()}`")
    c2R.progress(float(cstm_app.current_index/len(cstm_app)))
    c2R.markdown(f"Progress: {cstm_app.current_index}/{len(cstm_app)} - {cstm_app.current_index/len(cstm_app) :.2g} %")
    accept = c2RL[0].button("✅", on_click=cstm_app.accept)
    reject = c2RL[1].button("❌", on_click=cstm_app.reject)

    # display sample images
    c3 = st.container()
    c3C = c3.columns(3)

    for i, image_path in enumerate(cstm_app.get_images()):
        c3C[i].image(image_path, width=200)
        c3C[i].markdown(f"Image {i+1}")
