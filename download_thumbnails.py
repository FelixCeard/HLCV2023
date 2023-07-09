#!/home/hlcv_team017/miniconda3/envs/hlcv-ss23/bin/python
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
        file_path = os.path.join('data/thumbnails', f"{uid}_{file_name}")
        with open(file_path, 'wb') as f:
            f.write(r.content)
        # bar_download.update(1)
        return True
    except Exception as e:
        # bar_skip.update(1)
        return False

if __name__ == '__main__':
    """
    Downloads all of the 800k thumbnails (in max resolutions) from the Objaverse dataset.
    Also saving the meta-data of the item:
        attributes.txt: 
            Format: 
                uid1; name1; ['category1', 'category2', ...]; ['tag1', 'tag2', ...]
                uid2; name2; ['category1', 'category2', ...]; ['tag1', 'tag2', ...]
    """

    os.makedirs('data', exist_ok=True)

    succinct_failures = 0

    uids_global = objaverse.load_uids()
    # global uids
    # for informative debugging
    print('Found', len(uids_global), 'items')

    # setup the folder
    folder = os.makedirs('data/thumbnails', exist_ok=True)

    BATCH_SIZE = 10

    attributes = open('data/attributes.txt', 'w')


    # bar_download.total = len(self.uids)
    # bar_skip.total = len(self.uids)

    def load_batch_anotation(batch_id):
        uids = uids_global[batch_id * BATCH_SIZE: (batch_id + 1) * BATCH_SIZE]
        annotations = objaverse.load_annotations(uids)

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
        # global succinct_failures
        print(batch_id, '/', len(uids_global) // BATCH_SIZE)
        try:
            urls = load_batch_anotation(batch_id)
        except Exception as e:
            print('Error in batch:', batch_id)
            print(e)
            return

        for url in urls:
            download_image(url)



    def download_images():
        indices = list(range(len(uids_global) // BATCH_SIZE))

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            list(tqdm(executor.map(download_batch, indices), total=len(indices)))


    download_images()