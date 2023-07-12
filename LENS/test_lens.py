import requests
from lens import Lens, LensProcessor
from PIL import Image
import torch
import time
import json

if __name__ == '__main__':
    """
    Test how long the GPU cluster needs to load and parse one sample
    """

    # download image
    print('Downloading image')
    image_start = time.time()
    img_url = 'https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?w=720'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    question = "What is the image about?"
    image_end = time.time()
    print("Time to download image: ", image_end - image_start)

    # load lens
    # measure time to load lens
    lens_start = time.time()
    lens = Lens()
    lens_end = time.time()
    print("Time to load lens: ", lens_end - lens_start)

    # load lens processor
    processor_start = time.time()
    processor = LensProcessor()
    processor_end = time.time()
    print("Time to load processor: ", processor_end - processor_start)

    # inference
    print('Normal inference')
    with torch.no_grad():
        processor_inference_start = time.time()
        samples = processor([raw_image], [question])
        processor_inference_end = time.time()
        print("Time to process sample: ", processor_inference_end - processor_inference_start)

        lens_inference_start = time.time()
        output = lens(samples)
        lens_inference_end = time.time()
        print("Time to infer: ", lens_inference_end - lens_inference_start)

        print("Output: ", output)


        # test attributes
        attribute_start = time.time()
        attributes = lens(samples, return_tags=True, return_attributes=True, return_global_caption=False, return_intensive_captions=False, return_complete_prompt=False)
        attribute_end = time.time()
        print("Time to infer attributes: ", attribute_end - attribute_start)
        print('Attributes: ', attributes.keys())

        with open('attributes.json', 'w') as f:
            json.dump(attributes['attributes'], f)

        with open('tags.json', 'w') as f:
            json.dump(attributes['tags'], f)


