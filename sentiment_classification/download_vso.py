import os
import tempfile
import urllib
import cv2
import multiprocess
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Administrative constants
vso_dir = './vso/'
training_dir = './train/'
test_dir = './test/'
train_size = 0.8


def parse_anp_images(anp_file_name):
    adj = anp_file_name.split('_')[0]
    noun = anp_file_name.split('_')[1].split('.')[0]

    # Download each image listed in the file for this ANP
    imgs = []
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

        # Ignore broken links
        except urllib.error.HTTPError:
            pass

        # Delete the temporary file
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass

    # Split the images
    train_imgs, test_imgs = train_test_split(imgs, train_size=train_size, test_size=1-train_size)

    # Save the images
    for i, img in enumerate(train_imgs):
        cv2.imwrite(training_dir + '%s_%s_%i.jpg' % (adj, noun, i), img)
    for i, img in enumerate(test_imgs):
        cv2.imwrite(test_dir + '%s_%s_%i.jpg' % (adj, noun, i), img)


# Download and split all images for all ANPs
file_names = os.listdir(vso_dir)
with multiprocess.Pool(272, maxtasksperchild=1) as pool:
    iterator = pool.imap(parse_anp_images, file_names)
    list(tqdm(iterator, total=len(file_names), desc='All ANPs'))
