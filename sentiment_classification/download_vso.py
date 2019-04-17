import os
import tempfile
import urllib
import cv2
import pickle
import multiprocess
import pandas as pd
from tqdm import tqdm


# Administrative constants
vso_dir = './vso/'
pickle_dir = './parsed_vso/'


def parse_anp_images(anp_file_name):
    adj = anp_file_name.split('_')[0]
    noun = anp_file_name.split('_')[1].split('.')[0]
    pickle_file = pickle_dir + '%s_%s.pkl' % (adj, noun)

    # Only parse it if the file does not already exist
    if not os.path.isfile(pickle_file):

        # Download each image listed in the file
        imgs = []
        urls = []
        with open(vso_dir + anp_file_name, 'rb') as file_handle:
            file_lines = file_handle.readlines()
        for line in tqdm(file_lines, total=len(file_lines), desc='%s_%s' % (adj, noun)):
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

        # Save the results
        data = {'URL': urls,
                'RGB': imgs}
        df = pd.DataFrame(data)
        with open(pickle_file, 'wb') as file_handle:
            pickle.dump(df, file_handle)


# Get the ANP from the file names
file_names = os.listdir(vso_dir)
with multiprocess.Pool(272) as pool:
    iterator = pool.imap(parse_anp_images, file_names)
    list(tqdm(iterator, total=len(file_names), desc='All ANPs'))
