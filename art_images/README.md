The catalog here was downloaded from the [Web Gallery of
Art](https://www.wga.hu/index1.html), and then I used the `download_wga.py`
file to actually download the images. Next I used the `featurize_images.py`
file to use the network created in the `../sentiment_classification` folder to
turn the images into features. Then I used the `encode_features.py` script to
create an encoder to reduce the dimensionality of those features. Both the
ResNet features and the encoded features are inside `features.h5`. Each row in
`features.h5` maps to the rows listed in the `image_mapping.txt` file. The
encoder is stored as the `encoder.h5` file.
