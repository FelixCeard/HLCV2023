#!/usr/bin/env python
import os
import os.path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

import objaverse
import requests
from tqdm.auto import tqdm

def download_image(arg):
    # url, folder = arg
    try:
        url, uid, name = arg
        r = requests.get(url)
        if r.status_code != 200:
            # print('Error while downloading', url)
            return False
        file_name = url.split('/')[-1]
        # file_path = os.path.join('thumbnails', f"{str(name)}_{uid}_{file_name}")
        file_path = os.path.join('thumbnails', f"{uid}_{file_name}")
        with open(file_path, 'wb') as f:
            f.write(r.content)
        # bar_download.update(1)
        return True
    except Exception as e:
        # bar_skip.update(1)
        return False

#
# class Downloader:
#     def __init__(self):
#         self.uids = objaverse.load_uids()
#         # for informative debugging
#         print('Found', len(self.uids), 'items')
#
#         # setup the folder
#         self.folder = os.makedirs('thumbnails', exist_ok=True)
#
#         self.BATCH_SIZE = 10
#
#         self.attributes = open('attributes.txt', 'w')
#
#         # bar_download.total = len(self.uids)
#         # bar_skip.total = len(self.uids)
#
#     def load_batch_anotation(self, batch_id):
#         uids = self.uids[batch_id * self.BATCH_SIZE: (batch_id + 1) * self.BATCH_SIZE]
#         annotations = objaverse.load_annotations(uids[:10])
#
#         urls = []
#
#         for uid in uids:
#             annotation = annotations[uid]
#             name = annotation['name']
#             tags = annotation['tags']
#             categories = annotation['categories']
#
#             # only keep images with categories (sucht that we are able to train a classifier)
#
#             thumbnails = annotation['thumbnails'].get('images')
#             if thumbnails is None or len(categories) == 0:
#                 # bar_skip.update(1)
#                 continue
#
#             # get all of the available sizes and download the image closely larger than 256x256 pixels
#             availables_sizes = [(t['width'], t['size'], t['url']) for t in thumbnails]
#
#             min_size = 100000
#             min_url = None
#
#             for width, size, url in availables_sizes:
#                 # s = min(width, size)
#
#                 if width > 256:
#                     min_size = min(width, min_size)
#                     if min_size == width:
#                         min_url = url
#
#             # download the min_url
#             urls.append((min_url, uid, name))
#
#             # write the attributes to the file
#             # if len(tags) == 0:
#             #     self.attributes.write(f"{uid}; {[str(c['name']) for c in categories]}; \n")
#             # else:
#             self.attributes.write(f"{uid}; {[str(c['name']) for c in categories]}; {[str(c['name']) for c in tags]};\n")
#
#         return urls
#
#     def download_batch(self, batch_id):
#
#         urls = self.load_batch_anotation(batch_id)
#
#         pool = Pool(cpu_count())
#         results = pool.map(download_image, urls)
#         pool.close()
#         pool.join()
#
#     def download_images(self):
#         indices = list(range(len(self.uids) // self.BATCH_SIZE))
#
#         # for i in tqdm(indices, total=len(self.uids) // self.BATCH_SIZE, leave=True):
#         # for i in range(len(self.uids) // self.BATCH_SIZE):
#         #     self.download_batch(i)
#         #     if i == 5:
#         #         exit()


if __name__ == '__main__':
    """
    Downloads all of the 800k thumbnails (in max resolutions) from the Objaverse dataset.
    Also saving the meta-data of the item:
        attributes.txt: 
            Format: 
                uid1; name1; ['category1', 'category2', ...]; ['tag1', 'tag2', ...]
                uid2; name2; ['category1', 'category2', ...]; ['tag1', 'tag2', ...]
    """

    uids_global = objaverse.load_uids()
    # global uids
    # for informative debugging
    print('Found', len(uids_global), 'items')

    # setup the folder
    folder = os.makedirs('thumbnails', exist_ok=True)

    BATCH_SIZE = 10

    attributes = open('attributes.txt', 'w')


    # bar_download.total = len(self.uids)
    # bar_skip.total = len(self.uids)

    def load_batch_anotation(batch_id):
        uids = uids_global[batch_id * BATCH_SIZE: (batch_id + 1) * BATCH_SIZE]
        annotations = objaverse.load_annotations(uids[:10])

        urls = []

        for uid in uids:
            annotation = annotations[uid]
            name = annotation['name']
            categories = annotation['categories']
            tags = annotation['tags']

            # only keep images with categories (sucht that we are able to train a classifier)

            thumbnails = annotation['thumbnails'].get('images')
            if thumbnails is None or len(categories) == 0:
                # bar_skip.update(1)
                continue

            # get all of the available sizes and download the image closely larger than 256x256 pixels
            availables_sizes = [(t['width'], t['size'], t['url']) for t in thumbnails]

            min_size = 100000
            min_url = None

            for width, size, url in availables_sizes:
                # s = min(width, size)

                if width > 256:
                    min_size = min(width, min_size)
                    if min_size == width:
                        min_url = url

            # download the min_url
            urls.append((min_url, uid, name))

            # write the attributes to the file
            # attributes.write(f"{uid}; {[str(c['name']) for c in categories]}\n")
            attributes.write(f"{uid}; {[str(c['name']) for c in categories]}; {[str(c['name']) for c in tags]};\n")

        return urls


    def download_batch(batch_id):

        urls = load_batch_anotation(batch_id)

        for url in urls:
            download_image(url)

        # pool = Pool(cpu_count())
        # results = pool.map(download_image, urls)
        # pool.close()
        # pool.join()


    def download_images():
        indices = list(range(len(uids_global) // BATCH_SIZE))[:5]

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            list(tqdm(executor.map(download_batch, indices), total=len(indices), leave=True))

        # for i in tqdm(indices, total=len(indices), leave=True):
        # # for i in range(len(self.uids) // self.BATCH_SIZE):
        #     download_batch(i)
        #     if i == 5:
        #         exit()

    download_images()