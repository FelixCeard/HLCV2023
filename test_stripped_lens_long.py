import glob
import os.path
import os

from skimage import io
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from striped_lens.model import Lens, LensProcessor

from tqdm import tqdm

import torch
import time


def get_uids(path: str):
    # b3cc8f7f3cd0495ea3a8ddfae3902921_c0115832864e4938b898d5be34089cf5.jpeg
    return path.split('/')[-1].split('\\')[-1].split('_')[0]


def get_name(path: str):
    return path.split('/')[-1].split('\\')[-1].split('_')[1].split('.')[0]


class CustomDataset(Dataset):
    def __init__(self, root: str):
        self.root = root  # path to thumbnail folder

        self.paths = glob.glob(os.path.join(root, '*.jpeg'))
        self.uids = [get_uids(path) for path in self.paths]
        self.name = [get_uids(path) for path in self.paths]

        self.length = len(self.paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path = self.paths[index]
        uid = self.uids[index]
        name = self.name[index]

        # load image
        image = io.imread(path)

        return image, uid, name


if __name__ == '__main__':
    """
    Extract for each image in the dataset the attributes
    """

    BATCH_SIZE = 1
    NUM_WORKERS = 0

    os.makedirs('lens_attributes', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load lense
    lens_start = time.time()
    lens = Lens()
    lens_end = time.time()
    print("Time to load lens: ", lens_end - lens_start)

    # load lens processor
    processor_start = time.time()
    processor = LensProcessor()
    processor_end = time.time()
    print("Time to load processor: ", processor_end - processor_start)


    print('Loading dataset')
    time_start = time.time()
    dataset = CustomDataset('data/thumbnails')
    time_end = time.time()
    print('Loading dataset took', time_end - time_start, 'seconds')

    time_start = time.time()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    time_end = time.time()
    print('Loading dataloader took', time_end - time_start, 'seconds')

    # save attributes and tags to two files
    lens_attributes = open('lens_attributes/lens_attributes.txt', 'w')
    # lens_tags = open('lens_attributes/lens_tags.txt', 'w')

    with torch.no_grad():
        for imgs, uids, names in tqdm(dataloader, total=len(dataset)):

            imgs = imgs.to(device)

            # convert the images to logits
            samples = processor(imgs, None)

            attributes = lens(
                samples,
                return_tags=True,
                return_attributes=True,
                return_global_caption=False,
                return_intensive_captions=False,
                return_complete_prompt=False
            )

            imgs.cpu()

            for uid, name, attribute, tag in zip(uids, names, attributes['attributes'], attributes['tags']):
                lens_attributes.write(f"{uids}; {name}; {attribute}; {tag};\n")

