import urllib
import pandas as pd
import multiprocess
from tqdm import tqdm


# Read the catalog
df = pd.read_csv('catalog.txt', sep='\t', lineterminator='\r', encoding='ISO-8859-1')
# Take out the last data point, which is broken for some reason
df = df.drop([df.shape[0]-1])
# Rip out the URLs
urls = df['URL']


# Create a function to download an image based on its index
def download_image(index):
    url = urls[index]
    image_url = url.replace('html', 'art', 1).replace('html', 'jpg', 1)
    urllib.urlretrieve(image_url, './images/%05d.jpg' % index)


# Multithreaded downloading
pool = multiprocess.Pool(4, maxtasksperchild=1000)
iterator = pool.imap(download_image, range(len(urls)))
list(tqdm(iterator, total=len(urls)))
