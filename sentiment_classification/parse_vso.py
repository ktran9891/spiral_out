import os
import gc
import tempfile
import urllib
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# Administrative constants
vso_dir = './vso/'
pickle_dir = './parsed_vso/'


# Get the ANP from the file names
file_names = os.listdir(vso_dir)
for anp_file_name in tqdm(file_names, desc='ANPs'):
    adj = anp_file_name.split('_')[0]
    noun = anp_file_name.split('_')[1].split('.')[0]

    # Download each image listed in the file
    imgs = []
    urls = []
    with open(vso_dir + anp_file_name, 'rb') as file_handle:
        file_lines = file_handle.readlines()
    for line in tqdm(file_lines, total=len(file_lines), desc='Images for this ANP'):
        url = str(line).split(' ')[1]

        # Write the image to a temporary file and then get the RGB from it
        _, img_path = tempfile.mkstemp()
        try:
            urllib.urlretrieve(url, img_path)
            img = cv2.imread(img_path)

            # Ignore broken links
            if img is not None:
                imgs.append(img)
                urls.append(url)

        # Delete the temporary file
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass

    # Report some 404 errors
    print('Got %i out of %i images for %s_%s (%i%%)'
          % (len(imgs), len(file_lines), adj, noun,
             len(imgs)/len(file_lines)*100.))

    # Save the results
    data_dict = {'URL': urls,
                 'RGB': imgs}
    df = pd.DataFrame(data_dict)
    df.to_pickle(pickle_dir + '%s_%s.pkl' % (adj, noun))

    # Collect the garbage just in case we run into memory errors
    gc.collect()
