Need to add `export
PYTHONPATH="/home/ktran/spiral_out/dependencies:$PYTHONPATH"` to your
`.bashrc`.

I hacked together a Conda environment to run this stuff, and I forget what
exactly is necessary. Here's what `conda list` gives me:
```
# Name                    Version                   Build  Channel
_tflow_select             2.1.0                       gpu  
absl-py                   0.7.1                    py27_0    conda-forge
astor                     0.7.1                      py_0    conda-forge
atk                       2.25.90           hf2eb9ee_1001    conda-forge
attrs                     19.1.0                     py_0    conda-forge
backcall                  0.1.0                      py_0    conda-forge
backports                 1.0                        py_2    conda-forge
backports.functools_lru_cache 1.5                        py_1    conda-forge
backports.shutil_get_terminal_size 1.0.0                      py_3    conda-forge
backports.weakref         1.0.post1             py27_1000    conda-forge
backports_abc             0.5                        py_1    conda-forge
blas                      2.4                    openblas    conda-forge
bleach                    3.1.0                      py_0    conda-forge
bzip2                     1.0.6             h14c3975_1002    conda-forge
c-ares                    1.15.0            h14c3975_1001    conda-forge
ca-certificates           2019.3.9             hecc5488_0    conda-forge
cairo                     1.14.12           h80bd089_1005    conda-forge
certifi                   2019.3.9                 py27_0    conda-forge
cffi                      1.12.2           py27hf0e25f4_1    conda-forge
configparser              3.7.3                    py27_1    conda-forge
cudatoolkit               10.0.130                      0  
cudnn                     7.3.1                cuda10.0_0  
cupti                     10.0.130                      0  
cycler                    0.10.0                     py_1    conda-forge
dbus                      1.13.2               h714fa37_1  
decorator                 4.4.0                      py_0    conda-forge
defusedxml                0.5.0                      py_1    conda-forge
docutils                  0.14                  py27_1001    conda-forge
entrypoints               0.3                   py27_1000    conda-forge
enum34                    1.1.6                 py27_1001    conda-forge
expat                     2.2.5             hf484d3e_1002    conda-forge
ffmpeg                    4.1.1                h167e202_0    conda-forge
fontconfig                2.13.1            he4413a7_1000    conda-forge
freetype                  2.10.0               he983fc9_0    conda-forge
funcsigs                  1.0.2                      py_3    conda-forge
functools32               3.2.3.2                    py_3    conda-forge
futures                   3.2.0                 py27_1000    conda-forge
gast                      0.2.2                      py_0    conda-forge
gdk-pixbuf                2.36.12           h4f1c04b_1001    conda-forge
gettext                   0.19.8.1          hc5be6a0_1002    conda-forge
giflib                    5.1.7                h516909a_1    conda-forge
glib                      2.56.2            had28632_1001    conda-forge
gmp                       6.1.2             hf484d3e_1000    conda-forge
gnutls                    3.6.5             hd3a4fd2_1002    conda-forge
gobject-introspection     1.56.1          py27h9e29830_1001    conda-forge
graphite2                 1.3.13            hf484d3e_1000    conda-forge
grpcio                    1.16.1           py27hf8bcb03_1  
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb453b48_1  
gtk2                      2.24.31           h5baeb44_1000    conda-forge
h5py                      2.9.0           nompi_py27hf008753_1102    conda-forge
harfbuzz                  1.9.0             he243708_1001    conda-forge
hdf5                      1.10.4          nompi_h3c11f04_1106    conda-forge
icu                       58.2              hf484d3e_1000    conda-forge
intel-openmp              2019.3                      199  
ipaddress                 1.0.22                     py_1    conda-forge
ipykernel                 4.10.0                   py27_1    conda-forge
ipython                   4.2.1                    py27_1    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
ipywidgets                7.4.2                      py_0    conda-forge
jasper                    1.900.1           h07fcdf6_1006    conda-forge
jedi                      0.13.3                   py27_0    conda-forge
jinja2                    2.10                       py_1    conda-forge
jpeg                      9c                h14c3975_1001    conda-forge
jsonschema                3.0.1                    py27_0    conda-forge
jupyter                   1.0.0                      py_2    conda-forge
jupyter_client            5.2.4                      py_3    conda-forge
jupyter_console           5.1.0                    py27_0    conda-forge
jupyter_core              4.4.0                      py_0    conda-forge
keras                     2.1.6                    py27_0    conda-forge
keras-applications        1.0.7                      py_0    conda-forge
keras-preprocessing       1.0.9                      py_0    conda-forge
kiwisolver                1.0.1           py27h6bb024c_1002    conda-forge
lame                      3.100             h14c3975_1001    conda-forge
libblas                   3.8.0                4_openblas    conda-forge
libcblas                  3.8.0                4_openblas    conda-forge
libffi                    3.2.1             he1b5a44_1006    conda-forge
libgcc-ng                 8.2.0                hdf63c60_1  
libgfortran-ng            7.3.0                hdf63c60_0  
libgpuarray               0.7.6             h14c3975_1003    conda-forge
libiconv                  1.15              h516909a_1005    conda-forge
liblapack                 3.8.0                4_openblas    conda-forge
liblapacke                3.8.0                4_openblas    conda-forge
libopenblas               0.2.20               h9ac9557_7  
libpng                    1.6.36            h84994c4_1000    conda-forge
libprotobuf               3.7.0                h8b12597_2    conda-forge
libsodium                 1.0.16            h14c3975_1001    conda-forge
libstdcxx-ng              8.2.0                hdf63c60_1  
libtiff                   4.0.10            h648cc4a_1001    conda-forge
libuuid                   2.32.1            h14c3975_1000    conda-forge
libwebp                   1.0.2                h99fbfcb_2    conda-forge
libxcb                    1.13              h14c3975_1002    conda-forge
libxml2                   2.9.8             h143f9aa_1005    conda-forge
linecache2                1.0.0                      py_1    conda-forge
mako                      1.0.7                      py_1    conda-forge
markdown                  2.6.11                     py_0    conda-forge
markupsafe                1.1.1            py27h14c3975_0    conda-forge
matplotlib                2.2.3            py27h8a2030e_1    conda-forge
matplotlib-base           2.2.3            py27h60b886d_1    conda-forge
mistune                   0.8.4           py27h14c3975_1000    conda-forge
mkl                       2019.1                      144  
mock                      2.0.0                 py27_1001    conda-forge
nbconvert                 5.4.1                      py_2    conda-forge
nbformat                  4.4.0                      py_1    conda-forge
ncurses                   6.1               hf484d3e_1002    conda-forge
nettle                    3.4.1             h1bed415_1002    conda-forge
ninja                     1.9.0                h6bb024c_0    conda-forge
notebook                  5.7.8                    py27_0    conda-forge
numpy                     1.13.3          py27_nomklh2b20989_4  
olefile                   0.46                       py_0    conda-forge
openblas                  0.3.5             h9ac9557_1001    conda-forge
opencv                    2.4.13              np113py27_1    conda-forge
openh264                  1.8.0             hdbcaa40_1000    conda-forge
openssl                   1.1.1b               h14c3975_1    conda-forge
pandas                    0.24.2           py27hf484d3e_0    conda-forge
pandoc                    2.7.1                         0    conda-forge
pandocfilters             1.4.2                      py_1    conda-forge
pango                     1.40.14           hf0c64fd_1003    conda-forge
parso                     0.3.4                      py_0    conda-forge
pathlib2                  2.3.3                 py27_1000    conda-forge
patsy                     0.5.1                      py_0    conda-forge
pbr                       5.1.3                      py_0    conda-forge
pcre                      8.43                 he6710b0_0  
pexpect                   4.6.0                 py27_1000    conda-forge
pickleshare               0.7.5                 py27_1000    conda-forge
pillow                    6.0.0            py27he7afcd5_0    conda-forge
pip                       19.0.3                   py27_0    conda-forge
pixman                    0.34.0            h14c3975_1003    conda-forge
prometheus_client         0.6.0                      py_0    conda-forge
prompt_toolkit            2.0.9                      py_0    conda-forge
protobuf                  3.7.0            py27he1b5a44_1    conda-forge
pthread-stubs             0.4               h14c3975_1001    conda-forge
ptyprocess                0.6.0                 py27_1000    conda-forge
pycparser                 2.19                     py27_1    conda-forge
pygments                  2.3.1                      py_0    conda-forge
pygpu                     0.7.6           py27h3010b51_1000    conda-forge
pyparsing                 2.3.1                      py_0    conda-forge
pyqt                      5.6.0           py27h13b7fb3_1008    conda-forge
pyrsistent                0.14.11          py27h14c3975_0    conda-forge
python                    2.7.15            h721da81_1008    conda-forge
python-dateutil           2.8.0                      py_0    conda-forge
pytorch                   1.0.1           py2.7_cuda10.0.130_cudnn7.4.2_2    pytorch
pytz                      2018.9                     py_0    conda-forge
pyyaml                    5.1              py27h14c3975_0    conda-forge
pyzmq                     18.0.1           py27h0e1adb2_0    conda-forge
qt                        5.6.3                h8bf5577_3  
qtconsole                 4.4.3                      py_0    conda-forge
readline                  7.0               hf8c457e_1001    conda-forge
scandir                   1.10.0           py27h14c3975_0    conda-forge
scikit-learn              0.20.3           py27ha8026db_1    conda-forge
scipy                     1.2.1            py27h09a28d5_1    conda-forge
seaborn                   0.9.0                      py_0    conda-forge
send2trash                1.5.0                      py_0    conda-forge
setuptools                40.8.0                   py27_0    conda-forge
simplegeneric             0.8.1                      py_1    conda-forge
singledispatch            3.4.0.3               py27_1000    conda-forge
sip                       4.18.1          py27hf484d3e_1000    conda-forge
six                       1.12.0                py27_1000    conda-forge
sqlite                    3.26.0            h67949de_1001    conda-forge
statistics                1.0.3.5               py27_1001    conda-forge
statsmodels               0.9.0           py27h3010b51_1000    conda-forge
subprocess32              3.5.3            py27h14c3975_0    conda-forge
tensorboard               1.13.1                   py27_0    conda-forge
tensorflow                1.13.1          gpu_py27hcb41dfa_0  
tensorflow-base           1.13.1          gpu_py27h8d69cac_0  
tensorflow-estimator      1.13.0                     py_0  
tensorflow-gpu            1.13.1               h0d30ee6_0  
termcolor                 1.1.0                      py_2    conda-forge
terminado                 0.8.2                    py27_0    conda-forge
testpath                  0.4.2                   py_1001    conda-forge
theano                    1.0.3                    py27_0    conda-forge
tk                        8.6.9             h84994c4_1001    conda-forge
torchvision               0.2.1                 py27_1000    conda-forge
tornado                   5.1.1           py27h14c3975_1000    conda-forge
tqdm                      4.31.1                     py_0    conda-forge
traceback2                1.4.0                    py27_0    conda-forge
traitlets                 4.3.2                 py27_1000    conda-forge
unittest2                 1.1.0                      py_0    conda-forge
wcwidth                   0.1.7                      py_1    conda-forge
webencodings              0.5.1                      py_1    conda-forge
werkzeug                  0.15.2                     py_0    conda-forge
wheel                     0.33.1                   py27_0    conda-forge
widgetsnbextension        3.4.2                 py27_1000    conda-forge
x264                      1!152.20180806       h14c3975_0    conda-forge
xorg-kbproto              1.0.7             h14c3975_1002    conda-forge
xorg-libice               1.0.9             h516909a_1004    conda-forge
xorg-libsm                1.2.3             h84519dc_1000    conda-forge
xorg-libx11               1.6.7             h14c3975_1000    conda-forge
xorg-libxau               1.0.9                h14c3975_0    conda-forge
xorg-libxdmcp             1.1.3                h516909a_0    conda-forge
xorg-libxext              1.3.4                h516909a_0    conda-forge
xorg-libxrender           0.9.10            h516909a_1002    conda-forge
xorg-libxt                1.1.5             h14c3975_1002    conda-forge
xorg-renderproto          0.11.1            h14c3975_1002    conda-forge
xorg-xextproto            7.3.0             h14c3975_1002    conda-forge
xorg-xproto               7.0.31            h14c3975_1007    conda-forge
xz                        5.2.4             h14c3975_1001    conda-forge
yaml                      0.1.7             h14c3975_1001    conda-forge
zeromq                    4.2.5             hf484d3e_1006    conda-forge
zlib                      1.2.11            h14c3975_1004    conda-forge
```
