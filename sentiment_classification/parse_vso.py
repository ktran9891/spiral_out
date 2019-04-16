import os
import tempfile
import urllib
import cv2
import pandas as pd
from tqdm import tqdm


# Administrative constants
vso_dir = './vso/'
h5_file_name = 'vso.h5'


def process_anps(anp_file_name):
    # Get the ANP from the file name
    adj = anp_file_name.split('_')[0]
    noun = anp_file_name.split('_')[1].split('.')[0]

    # Download each image listed in the file
    imgs = []
    urls = []
    with open(vso_dir + anp_file_name, 'rb') as file_handle:
        file_lines = file_handle.readlines()
    #for line in tqdm(file_lines, total=len(file_lines), desc='Images for this ANP'):
    for line in file_lines:
        url = str(line).split(' ')[1]

        # Write the image to a temporary file and then get the RGB from it
        _, img_path = tempfile.mkstemp()
        try:
            urllib.request.urlretrieve(url, img_path)
            img = cv2.imread(img_path)
            imgs.append(img)
            urls.append(url)

        # Ignore broken links
        except urllib.error.HTTPError:
            pass

        # Delete the temporary file
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass

    # Report some 404 errors
    print('Got %i out of %i images for %s_%s (%i%%)'
          % (len(imgs), len(file_lines), adj, noun,
             len(imgs)/len(file_lines)*100))

    # Save the results
    data_dict = {'Adjective': [adj for _ in imgs],
                 'Noun': [noun for _ in imgs],
                 'RGB': [img for img in imgs],
                 'URL': [url for url in urls]}
    df = pd.DataFrame(data_dict)
    df.to_hdf(h5_file_name, 'VSO', append=True)


# get all of the attribute-noun-pairs in Columbia's visual sentiment ontology
# (VSO) dataset
file_names = os.listdir(vso_dir)
iterator = map(process_anps, file_names)
list(tqdm(iterator, total=len(file_names), desc='ANPs'))
