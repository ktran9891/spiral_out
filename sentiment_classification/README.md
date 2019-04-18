The contents of the `vso` folder were obtained from
[VSO](http://www.ee.columbia.edu/ln/dvmm/vso/download/flickr_dataset.html). I
downloaded the database using the `download_vso.py` file, which split and saved
the images in their respective `train/` and `test/` folders. Note that the
numbering in each of the folders is specific to that folder---e.g.,
`train/spooky_house_1.jpg` and `test/spooky_house_1.jpg` are different images.
I then used the `train_resnet_on_vso.py` file to... train resnet on the VSO
data. This yielded the `resnet*_vso.h5` model.
