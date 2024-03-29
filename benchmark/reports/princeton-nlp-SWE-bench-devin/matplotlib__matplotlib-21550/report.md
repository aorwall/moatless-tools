# matplotlib__matplotlib-21550

| **matplotlib/matplotlib** | `460073b2d9122e276d42c2775bad858e337a51f1` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 1 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/lib/matplotlib/collections.py b/lib/matplotlib/collections.py
--- a/lib/matplotlib/collections.py
+++ b/lib/matplotlib/collections.py
@@ -202,6 +202,18 @@ def __init__(self,
             if offsets.shape == (2,):
                 offsets = offsets[None, :]
             self._offsets = offsets
+        elif transOffset is not None:
+            _api.warn_deprecated(
+                '3.5',
+                removal='3.6',
+                message='Passing *transOffset* without *offsets* has no '
+                        'effect. This behavior is deprecated since %(since)s '
+                        'and %(removal)s, *transOffset* will begin having an '
+                        'effect regardless of *offsets*. In the meantime, if '
+                        'you wish to set *transOffset*, call '
+                        'collection.set_offset_transform(transOffset) '
+                        'explicitly.')
+            transOffset = None
 
         self._transOffset = transOffset
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| lib/matplotlib/collections.py | 205 | 205 | - | - | -


## Problem Statement

```
[Bug]: this example shows ok on matplotlib-3.4.3, but not in matplotlib-3.5.0 master of october 30th
### Bug summary

the display is not working well if swaping matplotlib-3.4.3 with matplotlib-3.5.0.dev2445+gb09aad279b, all the rest being strictly equal.
it was also bad with rc1, so I tested with last master, thanks to the artefact generation

### Code for reproduction
on jupyterlab

\`\`\`python


`
%matplotlib inline
from ipywidgets import interact
import matplotlib.pyplot as plt
import networkx as nx
# wrap a few graph generation functions so they have the same signature

def random_lobster(n, m, k, p):
    return nx.random_lobster(n, p, p / m)

def powerlaw_cluster(n, m, k, p):
    return nx.powerlaw_cluster_graph(n, m, p)

def erdos_renyi(n, m, k, p):
    return nx.erdos_renyi_graph(n, p)

def newman_watts_strogatz(n, m, k, p):
    return nx.newman_watts_strogatz_graph(n, k, p)

@interact(n=(2,30), m=(1,10), k=(1,10), p=(0.0, 1.0, 0.001),
        generator={'lobster': random_lobster,
                   'power law': powerlaw_cluster,
                   'Newman-Watts-Strogatz': newman_watts_strogatz,
                   u'Erdős-Rényi': erdos_renyi,
                   })
def plot_random_graph(n, m, k, p, generator):
    g = generator(n, m, k, p)
    nx.draw(g)
    plt.title(generator.__name__)
    plt.show()
    \`\`\``
\`\`\`


### Actual outcome

![image](https://user-images.githubusercontent.com/4312421/139675032-1c89dac9-9975-4379-b390-8fe7317e8fcb.png)


### Expected outcome

![image](https://user-images.githubusercontent.com/4312421/139675329-980a0007-8533-41a6-9686-bb1b9e835d36.png)


### Operating system

Windows 10

### Matplotlib Version

matplotlib-3.5.0.dev2445+gb09aad279b-cp39-cp39-win_amd64.whl

### Matplotlib Backend

module://matplotlib_inline.backend_inline

### Python version

Python 3.9.7

### Jupyter version

3.2.1

### Other libraries

wheels from cgohlke when binaries, except the  matplotlib-master from https://pipelines.actions.githubusercontent.com/radNkCxZv5fwMgK3hRdEtEflfPA62ntLWJUtB75BrsUZ7MmN7K/_apis/pipelines/1/runs/264026/signedartifactscontent?artifactName=wheels&urlExpires=2021-11-01T10%3A56%3A22.6171739Z&urlSigningMethod=HMACV1&urlSignature=0AaHHaQnK512QOq6OgHWoS%2FvuqsCMZseoyfIWyE6y6c%3D

pip list:
<details>
Package                           Version
--------------------------------- ------------------
adodbapi                          2.6.1.3
affine                            2.3.0
aiofiles                          0.6.0
aiohttp                           3.7.4.post0
aiosqlite                         0.17.0
alabaster                         0.7.12
algopy                            0.5.7
altair                            4.1.0
altair-data-server                0.4.1
altair-transform                  0.2.0
altair-widgets                    0.2.2
altgraph                          0.17.2
amply                             0.1.4
aniso8601                         7.0.0
ansiwrap                          0.8.4
anyio                             3.3.4
appdirs                           1.4.4
argon2-cffi                       21.1.0
arrow                             1.2.1
asciitree                         0.3.3
asgi-csrf                         0.9
asgiref                           3.4.1
asn1crypto                        1.4.0
asteval                           0.9.25
astor                             0.8.1
astroid                           2.6.6
astroML                           1.0.1
astropy                           4.3.1
async-generator                   1.10
async-timeout                     3.0.1
atomicwrites                      1.4.0
attrs                             21.2.0
autopep8                          1.5.7
Babel                             2.9.1
backcall                          0.2.0
backports-abc                     0.5
backports.entry-points-selectable 1.1.0
baresql                           0.7.6
base58                            2.0.0
bcrypt                            3.2.0
beautifulsoup4                    4.10.0
binaryornot                       0.4.4
black                             21.9b0
bleach                            4.1.0
blinker                           1.4
blis                              0.7.5
blosc                             1.10.6
bloscpack                         0.16.0
bokeh                             2.4.1
botorch                           0.4.0
Bottleneck                        1.3.2
bqplot                            0.12.31
branca                            0.4.2
brewer2mpl                        1.4.1
Brotli                            1.0.9
cachelib                          0.3.0
cachetools                        4.2.4
Cartopy                           0.20.1
catalogue                         2.0.6
certifi                           2021.10.8
cffi                              1.15.0
cftime                            1.5.1.1
chardet                           4.0.0
charset-normalizer                2.0.7
click                             7.1.2
click-default-group               1.2.2
click-plugins                     1.1.1
cligj                             0.7.2
cloudpickle                       2.0.0
clrmagic                          0.0.1a2
colorama                          0.4.4
colorcet                          2.0.6
cookiecutter                      1.7.3
coverage                          6.0.2
cramjam                           2.4.0
cryptography                      35.0.0
csvs-to-sqlite                    1.2
cvxopt                            1.2.7
cvxpy                             1.1.15
cx-Freeze                         6.5.3
cycler                            0.11.0
cymem                             2.0.6
Cython                            0.29.24
cytoolz                           0.11.0
dash                              2.0.0
dash-core-components              2.0.0
dash-html-components              2.0.0
dash-table                        5.0.0
dask                              2021.10.0
dask-glm                          0.2.0
dask-image                        0.6.0
dask-labextension                 5.1.0
dask-ml                           2021.10.17
dask-searchcv                     0.2.0
databases                         0.4.1
datasette                         0.59.1
datasette-graphql                 1.5
datashader                        0.13.0
datashape                         0.5.2
dateparser                        1.1.0
dateutils                         0.6.12
db.py                             0.5.4b1
debugpy                           1.5.1
decorator                         4.4.2
defusedxml                        0.7.1
Deprecated                        1.2.13
deprecation                       2.1.0
descartes                         1.1.0
diff-match-patch                  20200713
distlib                           0.3.3
distributed                       2021.10.0
docopt                            0.6.2
docrepr                           0.1.1
docutils                          0.17.1
ecos                              2.0.7.post1
emcee                             3.1.1
entrypoints                       0.3
et-xmlfile                        1.1.0
fast-histogram                    0.10
fastai                            2.5.3
fastapi                           0.70.0
fastcore                          1.3.26
fastdownload                      0.0.5
fasteners                         0.16.3
fastparquet                       0.7.1
fastprogress                      1.0.0
feather-format                    0.4.1
filelock                          3.3.2
Fiona                             1.8.20
flake8                            3.9.2
Flask                             2.0.2
flask-accepts                     0.18.4
Flask-Compress                    1.10.1
Flask-Cors                        3.0.10
Flask-Mail                        0.9.1
flask-restx                       0.5.1
Flask-Session                     0.4.0
Flask-SQLAlchemy                  2.5.1
flaskerize                        0.14.0
flatbuffers                       2.0
flit                              3.4.0
flit_core                         3.4.0
folium                            0.12.1
fonttools                         4.27.1
formlayout                        1.2.1a1
fs                                2.4.13
fsspec                            2021.10.1
future                            0.18.2
fuzzywuzzy                        0.18.0
GDAL                              3.3.3
geographiclib                     1.52
geopandas                         0.10.2
geopy                             2.2.0
geoviews                          1.9.2
gitdb                             4.0.9
GitPython                         3.1.24
gmpy2                             2.0.8
gpytorch                          1.5.1
graphene                          2.1.9
graphql-core                      2.3.1
graphql-relay                     2.0.1
great-expectations                0.13.36
greenlet                          1.1.2
guidata                           1.8.1a0
guiqwt                            3.0.7
h11                               0.12.0
h2                                4.1.0
h5py                              3.5.0
HeapDict                          1.0.1
holoviews                         1.14.6
hpack                             4.0.0
html5lib                          1.1
httpcore                          0.13.7
httpie                            2.6.0
httpx                             0.20.0
hupper                            1.10.3
husl                              4.0.3
hvplot                            0.7.3
Hypercorn                         0.11.2
hyperframe                        6.0.1
hypothesis                        6.24.0
ibis-framework                    1.4.0
idlex                             1.18
idna                              3.1
imageio                           2.10.1
imageio-ffmpeg                    0.4.2
imagesize                         1.2.0
imbalanced-learn                  0.8.1
importlib-metadata                4.8.0
inflection                        0.5.1
iniconfig                         1.1.1
intake                            0.6.2
intervaltree                      3.0.2
ipycanvas                         0.9.1
ipykernel                         6.4.2
ipyleaflet                        0.14.0
ipympl                            0.8.2
ipython                           7.29.0
ipython-genutils                  0.2.0
ipython-sql                       0.4.1b1
ipywidgets                        7.6.5
isort                             5.9.3
itsdangerous                      2.0.1
janus                             0.6.2
jedi                              0.18.0
Jinja2                            3.0.2
jinja2-time                       0.2.0
joblib                            1.1.0
json5                             0.9.6
jsonpatch                         1.32
jsonpointer                       2.1
jsonschema                        4.1.2
julia                             0.5.7
jupyter                           1.0.0
jupyter-bokeh                     3.0.4
jupyter-client                    6.2.0
jupyter-console                   6.4.0
jupyter-core                      4.9.1
jupyter-lsp                       1.5.0
jupyter-packaging                 0.11.0
jupyter-server                    1.11.1
jupyter-server-mathjax            0.2.3
jupyter-server-proxy              3.1.0
jupyter-sphinx                    0.3.2
jupyterlab                        3.2.1
jupyterlab-git                    0.33.0
jupyterlab-launcher               0.13.1
jupyterlab-lsp                    3.9.1
jupyterlab-pygments               0.1.2
jupyterlab-server                 2.8.2
jupyterlab-widgets                1.0.2Note: you may need to restart the kernel to use updated packages.
keyring                           23.2.1
kiwisolver                        1.3.2
lazy-object-proxy                 1.6.0
llvmlite                          0.37.0
lmfit                             1.0.3
locket                            0.2.1
loky                              3.0.0
lxml                              4.6.3
lz4                               3.1.3
Markdown                          3.3.4
MarkupSafe                        2.0.1
marshmallow                       3.12.1
matplotlib                        3.4.3
matplotlib-inline                 0.1.3
maturin                           0.11.5
mccabe                            0.6.1
mercantile                        1.2.1
mergedeep                         1.3.4
metakernel                        0.27.5
mistune                           0.8.4
mizani                            0.7.3
mkl-service                       2.4.0
mlxtend                           0.18.0
moviepy                           1.0.3
mpl-scatter-density               0.7
mpld3                             0.5.5
mpldatacursor                     0.7.1
mpmath                            1.2.1
msgpack                           1.0.2
msvc-runtime                      14.29.30133
multidict                         5.2.0
multipledispatch                  0.6.0
munch                             2.5.0
murmurhash                        1.0.6
mypy                              0.910
mypy-extensions                   0.4.3
mysql-connector-python            8.0.26
nbclassic                         0.3.4
nbclient                          0.5.4
nbconvert                         6.2.0
nbconvert_reportlab               0.2
nbdime                            3.1.1
nbformat                          5.1.3
nbval                             0.9.6
nest-asyncio                      1.5.1
netCDF4                           1.5.8
networkx                          2.6.3
NLopt                             2.7.0
nltk                              3.6.5
notebook                          6.4.5
numba                             0.54.1
numcodecs                         0.9.1
numdifftools                      0.9.40
numexpr                           2.7.3
numpy                             1.20.3+mkl
numpydoc                          1.1.0
oct2py                            5.2.0
octave-kernel                     0.32.0
onnxruntime                       1.9.0
openpyxl                          3.0.9

orjson                            3.6.4
osqp                              0.6.2.post0
outcome                           1.1.0
packaging                         21.2
palettable                        3.3.0
pandas                            1.3.4
pandas-datareader                 0.10.0
pandocfilters                     1.5.0
panel                             0.12.4
papermill                         2.3.3
param                             1.12.0
parambokeh                        0.2.3
paramiko                          2.8.0
paramnb                           2.0.4
parso                             0.8.2
partd                             1.2.0
pathspec                          0.9.0
pathy                             0.6.1
patsy                             0.5.2
pdfrw                             0.4
pdvega                            0.2.1.dev0
pefile                            2021.9.3
pep8                              1.7.1
pexpect                           4.8.0
pg8000                            1.21.1
pickleshare                       0.7.5
Pillow                            8.4.0
PIMS                              0.5
Pint                              0.18
pip                               21.3.1
pipdeptree                        2.2.0
pkginfo                           1.7.1
platformdirs                      2.4.0
plotly                            5.3.1
plotnine                          0.8.0
pluggy                            1.0.0
ply                               3.11
portpicker                        1.4.0
poyo                              0.5.0
ppci                              0.5.8
preshed                           3.0.6
prettytable                       2.2.1
priority                          2.0.0
proglog                           0.1.9
prometheus-client                 0.12.0
promise                           2.3
prompt-toolkit                    3.0.21
protobuf                          4.0.0rc1
psutil                            5.8.0
ptpython                          3.0.20
ptyprocess                        0.7.0
PuLP                              2.3
py                                1.10.0
py-lru-cache                      0.1.4
pyaml                             20.4.0
pyarrow                           6.0.0
PyAudio                           0.2.11
pybars3                           0.9.7
pybind11                          2.8.1
pycodestyle                       2.7.0
pycosat                           0.6.3
pycparser                         2.20
pyct                              0.4.8
pydantic                          1.8.2
pydeck                            0.7.1
pydocstyle                        6.1.1
pyepsg                            0.4.0
pyerfa                            2.0.0
pyflakes                          2.3.1
pyflux                            0.4.17
pygame                            2.0.3
pygbm                             0.1.0
Pygments                          2.10.0
pyhdf                             0.10.3
pyinstaller                       4.5.1
pyinstaller-hooks-contrib         2021.3
pylint                            2.9.6
pyls-spyder                       0.4.0
pymc                              2.3.8
PyMeta3                           0.5.1
pymongo                           3.12.1
PyNaCl                            1.4.0
pynndescent                       0.5.5
pyodbc                            4.0.32
PyOpenGL                          3.1.5
pypandoc                          1.5
pyparsing                         2.4.7
pyproj                            3.2.1
PyQt5                             5.15.1
PyQt5-sip                         12.8.1
pyqt5-tools                       5.15.1.1.7.5.post3
pyqtgraph                         0.12.2
PyQtWebEngine                     5.15.1
pyrsistent                        0.18.0
pyserial                          3.5
pyshp                             2.1.3
PySocks                           1.7.1
pystache                          0.5.4
pytest                            6.2.5
python-baseconv                   1.2.2
python-dateutil                   2.8.2
python-dotenv                     0.19.1
python-hdf4                       0.10.0+dummy
python-Levenshtein                0.12.2
python-lsp-black                  1.0.0
python-lsp-jsonrpc                1.0.0
python-lsp-server                 1.2.4
python-multipart                  0.0.5
python-picard                     0.7
python-slugify                    5.0.2
python-snappy                     0.6.0
pythonnet                         2.5.2
PythonQwt                         0.9.2
pytz                              2021.3
pyviz-comms                       2.1.0
PyWavelets                        1.1.1
pywin32                           302
pywin32-ctypes                    0.2.0
pywinpty                          1.1.5
pywinusb                          0.4.2
PyYAML                            6.0
pyzmq                             22.3.0
pyzo                              4.11.3a1
QDarkStyle                        3.0.2
qdldl                             0.1.5.post0
qpsolvers                         1.7.0
qstylizer                         0.2.1
QtAwesome                         1.0.3
qtconsole                         5.1.1
QtPy                              2.0.0.dev0
quadprog                          0.1.8
quantecon                         0.5.1
Quart                             0.15.1
rasterio                          1.2.10
readme-renderer                   30.0
redis                             3.5.3
regex                             2021.10.23
reportlab                         3.6.2
requests                          2.26.0
requests-toolbelt                 0.9.1
requests-unixsocket               0.2.0
rfc3986                           1.5.0
rise                              5.7.1
rope                              0.21.0
rpy2                              3.4.0.dev0
Rtree                             0.9.7
ruamel.yaml                       0.17.15
ruamel.yaml.clib                  0.2.6
Rx                                1.6.1
scikit-fuzzy                      0.4.1
scikit-image                      0.18.3
scikit-learn                      1.0.1
scikit-optimize                   0.9.0
scilab2py                         0.6.2
scipy                             1.7.1
scramp                            1.4.1
scs                               2.1.4
seaborn                           0.11.2
Send2Trash                        1.8.0
setuptools                        58.3.0
setuptools-scm                    6.3.2
shap                              0.39.0
Shapely                           1.8.0
simpervisor                       0.4
simplegeneric                     0.8.1
simplejson                        3.17.5
simpy                             4.0.1
six                               1.16.0
sklearn-contrib-lightning         0.6.1
slicer                            0.0.7
slicerator                        1.0.0
smart-open                        5.2.1
smmap                             5.0.0
snakeviz                          2.1.0
sniffio                           1.2.0
snowballstemmer                   2.1.0
snuggs                            1.4.7
sortedcontainers                  2.4.0
sounddevice                       0.4.3
soupsieve                         2.2.1
spacy                             3.1.3
spacy-legacy                      3.0.8
Sphinx                            4.2.0
sphinx-rtd-theme                  1.0.0
sphinxcontrib-applehelp           1.0.2
sphinxcontrib-devhelp             1.0.2
sphinxcontrib-htmlhelp            2.0.0
sphinxcontrib-jsmath              1.0.1
sphinxcontrib-qthelp              1.0.3
sphinxcontrib-serializinghtml     1.1.5
spyder                            5.1.5
spyder-kernels                    2.1.3
SQLAlchemy                        1.4.26
sqlite-bro                        0.12.1
sqlite-fts4                       1.0.1
sqlite-utils                      3.17.1
sqlparse                          0.4.2
srsly                             2.4.2
starlette                         0.16.0
statsmodels                       0.13.0
streamlit                         1.1.0
streamz                           0.6.3
supersmoother                     0.4
sympy                             1.9
tables                            3.6.1
tabulate                          0.8.9
tblib                             1.7.0
tenacity                          8.0.1
termcolor                         1.1.0
terminado                         0.12.1
testpath                          0.5.0
text-unidecode                    1.3
textdistance                      4.2.2
textwrap3                         0.9.2
thinc                             8.0.12
threadpoolctl                     3.0.0
three-merge                       0.1.1
thrift                            0.15.0
tifffile                          2021.10.12
tinycss2                          1.1.0
toml                              0.10.2
tomli                             1.2.2
tomli_w                           0.4.0
tomlkit                           0.7.2
toolz                             0.11.1
torch                             1.10.0
torchaudio                        0.10.0
torchvision                       0.11.1
tornado                           6.1
tqdm                              4.62.3
traitlets                         5.1.1
traittypes                        0.2.1
tranquilizer                      0.5.1a1
trio                              0.19.0
trio-asyncio                      0.12.0
twine                             3.4.2
twitter                           1.19.2
typer                             0.4.0
typing-extensions                 3.10.0.2
tzlocal                           2.1
ujson                             4.2.0
umap-learn                        0.5.1
uncertainties                     3.1.6
urllib3                           1.26.7
uvicorn                           0.15.0
validators                        0.18.2
vega                              3.5.0
vega-datasets                     0.9.0
virtualenv                        20.9.0
ViTables                          3.0.2
voila                             0.2.16
voila-gridstack                   0.2.0
wasabi                            0.8.2
wasmer                            1.0.0
wasmer_compiler_cranelift         1.0.0
wasmer_compiler_singlepass        1.0.0
watchdog                          2.1.5
wcwidth                           0.2.5
webencodings                      0.5.1
websocket-client                  1.2.1
Werkzeug                          2.0.2
wheel                             0.37.0
widgetsnbextension                3.5.2
winpython                         4.6.20211017
winrt                             1.0.21033.1
wordcloud                         1.8.1
wrapt                             1.12.1
wsproto                           1.0.0
xarray                            0.19.0
XlsxWriter                        3.0.1
xlwings                           0.24.7
yapf                              0.31.0
yarl                              1.7.0
zarr                              2.10.2
zict                              2.0.0
zipp                              3.6.0
zstandard                         0.16.0
</details>
### Installation

pip

### Conda channel

_No response_

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 setup.py | 273 | 350| 692 | 692 | 2944 | 
| 2 | 2 lib/matplotlib/_cm.py | 1363 | 1441| 870 | 1562 | 31387 | 
| 3 | 3 setupext.py | 360 | 470| 774 | 2336 | 37631 | 
| 4 | 4 lib/matplotlib/pyplot.py | 1 | 85| 643 | 2979 | 64025 | 
| 5 | 5 doc/conf.py | 117 | 193| 760 | 3739 | 68890 | 
| 6 | 6 tutorials/introductory/sample_plots.py | 1 | 363| 2652 | 6391 | 71655 | 
| 7 | 6 lib/matplotlib/pyplot.py | 3013 | 3077| 612 | 7003 | 71655 | 
| 8 | 6 doc/conf.py | 1 | 87| 658 | 7661 | 71655 | 
| 9 | 7 lib/matplotlib/__init__.py | 1 | 115| 717 | 8378 | 83285 | 
| 10 | 7 doc/conf.py | 194 | 303| 729 | 9107 | 83285 | 
| 11 | 8 lib/matplotlib/pylab.py | 1 | 52| 396 | 9503 | 83682 | 
| 12 | 8 setup.py | 221 | 271| 413 | 9916 | 83682 | 
| 13 | 8 setupext.py | 179 | 208| 283 | 10199 | 83682 | 
| 14 | 9 lib/matplotlib/backends/qt_compat.py | 112 | 161| 429 | 10628 | 86171 | 
| 15 | 9 doc/conf.py | 90 | 115| 218 | 10846 | 86171 | 
| 16 | 10 lib/matplotlib/backends/_backend_tk.py | 1 | 57| 368 | 11214 | 94622 | 
| 17 | 11 tools/boilerplate.py | 200 | 312| 742 | 11956 | 97485 | 
| 18 | 12 lib/matplotlib/cbook/__init__.py | 1 | 43| 215 | 12171 | 116306 | 
| 19 | 12 lib/matplotlib/pyplot.py | 2249 | 2324| 767 | 12938 | 116306 | 
| 20 | 12 doc/conf.py | 494 | 622| 828 | 13766 | 116306 | 
| 21 | 13 lib/matplotlib/backends/backend_pdf.py | 7 | 49| 283 | 14049 | 140687 | 
| 22 | 13 lib/matplotlib/backends/qt_compat.py | 1 | 77| 766 | 14815 | 140687 | 
| 23 | 13 doc/conf.py | 304 | 411| 821 | 15636 | 140687 | 
| 24 | 13 lib/matplotlib/_cm.py | 461 | 506| 1095 | 16731 | 140687 | 
| 25 | 14 tutorials/text/usetex.py | 1 | 169| 1734 | 18465 | 142421 | 
| 26 | 14 lib/matplotlib/_cm.py | 158 | 216| 654 | 19119 | 142421 | 


## Missing Patch Files

 * 1: lib/matplotlib/collections.py

### Hint

```
Thanks for testing the RC!  Do you really need the interactive code _and_ networkx to reproduce?  We strongly prefer self-contained issues that don't use downstream libraries.  
I guess the interactive code may be stripped out. will try. 

\`\`\``
# Networks graph Example : https://github.com/ipython/ipywidgets/blob/master/examples/Exploring%20Graphs.ipynb
%matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx

def plot_random_graph(n, m, k, p):
    g = nx.random_lobster(16, 0.5 , 0.5/16)
    nx.draw(g)
    plt.title('lobster')
    plt.show()

plot_random_graph(16, 5 , 5 , 0)
\`\`\``

with Matplotlib-3.4.3
![image](https://user-images.githubusercontent.com/4312421/139744954-1236efdb-7394-4f3d-ba39-f01c4c830a41.png)


with matplotlib-3.5.0.dev2445+gb09aad279b-cp39-cp39-win_amd64.whl
![image](https://user-images.githubusercontent.com/4312421/139745259-057a8e2c-9b4b-4efc-bae1-8dfe156d02e1.png)

code simplified shall be:
\`\`\``
%matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx

g = nx.random_lobster(16, 0.5 , 0.5/16)
nx.draw(g)
plt.title('lobster')
plt.show()
\`\`\``
FWIW the problem seems to be with `LineCollection`, which is used to represent undirected edges in NetworkX's drawing functions.
Bisecting identified 1f4708b310 as the source of the behavior change.
It would still be helpful to have this in pure matplotlib. What does networkx do using line collection that the rc breaks?   Thanks!
Here's the best I could do to boil down `nx_pylab.draw_networkx_edges` to a minimal example:

\`\`\`python
import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
import matplotlib as mpl                                                        
                                                                                
loc = np.array([[[ 1.        ,  0.        ],                                    
        [ 0.30901695,  0.95105657]],                                            
                                                                                
       [[ 1.        ,  0.        ],                                             
        [-0.80901706,  0.58778526]],                                            
                                                                                
       [[ 1.        ,  0.        ],                                             
        [-0.809017  , -0.58778532]],                                            
                                                                                
       [[ 1.        ,  0.        ],                                             
        [ 0.3090171 , -0.95105651]],                                            
                                                                                
       [[ 0.30901695,  0.95105657],                                             
        [-0.80901706,  0.58778526]],                                            
                                                                                
       [[ 0.30901695,  0.95105657],                                             
        [-0.809017  , -0.58778532]],                                            
                                                                                
       [[ 0.30901695,  0.95105657],                                             
        [ 0.3090171 , -0.95105651]],                                            
                                                                                
       [[-0.80901706,  0.58778526],                                             
        [-0.809017  , -0.58778532]],                                            
                                                                                
       [[-0.80901706,  0.58778526],                                             
        [ 0.3090171 , -0.95105651]],                                            
                                                                                
       [[-0.809017  , -0.58778532],                                             
        [ 0.3090171 , -0.95105651]]])                                           
fig, ax = plt.subplots()                                                        
lc = mpl.collections.LineCollection(loc, transOffset=ax.transData)              
ax.add_collection(lc)                                                           
minx = np.amin(np.ravel(loc[..., 0]))                                           
maxx = np.amax(np.ravel(loc[..., 0]))                                           
miny = np.amin(np.ravel(loc[..., 1]))                                           
maxy = np.amax(np.ravel(loc[..., 1]))                                           
w = maxx - minx                                                                 
h = maxy - miny                                                                 
padx, pady = 0.05 * w, 0.05 * h                                                 
corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)                
ax.update_datalim(corners)                                                      
ax.autoscale_view()                                                             
plt.show()
\`\`\`

With 3.4.3 this gives:

![mpl_3 4 3](https://user-images.githubusercontent.com/1268991/139796792-459be85d-cf05-4077-984c-e4762d2d0562.png)

and with 3.5.0rc1:
![mpl_3 5 0rc1](https://user-images.githubusercontent.com/1268991/139796823-6bc62690-dca4-4ec8-b0a3-2f01ff873ca1.png)




The problem is passing `transOffset`, which previously did nothing if you didn't pass `offsets`, but now does all the time. That was a mistake and not supposed to have been changed, I think.
```

## Patch

```diff
diff --git a/lib/matplotlib/collections.py b/lib/matplotlib/collections.py
--- a/lib/matplotlib/collections.py
+++ b/lib/matplotlib/collections.py
@@ -202,6 +202,18 @@ def __init__(self,
             if offsets.shape == (2,):
                 offsets = offsets[None, :]
             self._offsets = offsets
+        elif transOffset is not None:
+            _api.warn_deprecated(
+                '3.5',
+                removal='3.6',
+                message='Passing *transOffset* without *offsets* has no '
+                        'effect. This behavior is deprecated since %(since)s '
+                        'and %(removal)s, *transOffset* will begin having an '
+                        'effect regardless of *offsets*. In the meantime, if '
+                        'you wish to set *transOffset*, call '
+                        'collection.set_offset_transform(transOffset) '
+                        'explicitly.')
+            transOffset = None
 
         self._transOffset = transOffset
 

```

## Test Patch

```diff
diff --git a/lib/matplotlib/tests/test_collections.py b/lib/matplotlib/tests/test_collections.py
--- a/lib/matplotlib/tests/test_collections.py
+++ b/lib/matplotlib/tests/test_collections.py
@@ -1072,8 +1072,13 @@ def test_set_offsets_late():
 
 
 def test_set_offset_transform():
+    with pytest.warns(MatplotlibDeprecationWarning,
+                      match='.transOffset. without .offsets. has no effect'):
+        mcollections.Collection([],
+                                transOffset=mtransforms.IdentityTransform())
+
     skew = mtransforms.Affine2D().skew(2, 2)
-    init = mcollections.Collection([], transOffset=skew)
+    init = mcollections.Collection([], offsets=[], transOffset=skew)
 
     late = mcollections.Collection([])
     late.set_offset_transform(skew)

```


## Code snippets

### 1 - setup.py:

Start line: 273, End line: 350

```python
setup(  # Finally, pass this all along to setuptools to do the heavy lifting.
    name="matplotlib",
    description="Python plotting package",
    author="John D. Hunter, Michael Droettboom",
    author_email="matplotlib-users@python.org",
    url="https://matplotlib.org",
    download_url="https://matplotlib.org/users/installing.html",
    project_urls={
        'Documentation': 'https://matplotlib.org',
        'Source Code': 'https://github.com/matplotlib/matplotlib',
        'Bug Tracker': 'https://github.com/matplotlib/matplotlib/issues',
        'Forum': 'https://discourse.matplotlib.org/',
        'Donate': 'https://numfocus.org/donate-to-matplotlib'
    },
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    license="PSF",
    platforms="any",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Framework :: Matplotlib',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Python Software Foundation License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Visualization',
    ],

    package_dir={"": "lib"},
    packages=find_packages("lib"),
    namespace_packages=["mpl_toolkits"],
    py_modules=["pylab"],
    # Dummy extension to trigger build_ext, which will swap it out with
    # real extensions that can depend on numpy for the build.
    ext_modules=[Extension("", [])],
    package_data=package_data,

    python_requires='>={}'.format('.'.join(str(n) for n in py_min_version)),
    setup_requires=[
        "certifi>=2020.06.20",
        "numpy>=1.17",
        "setuptools_scm>=4",
        "setuptools_scm_git_archive",
    ],
    install_requires=[
        "cycler>=0.10",
        "fonttools>=4.22.0",
        "kiwisolver>=1.0.1",
        "numpy>=1.17",
        "packaging>=20.0",
        "pillow>=6.2.0",
        "pyparsing>=2.2.1",
        "python-dateutil>=2.7",
    ] + (
        # Installing from a git checkout.
        ["setuptools_scm>=4"] if Path(__file__).with_name(".git").exists()
        else []
    ),
    use_scm_version={
        "version_scheme": "release-branch-semver",
        "local_scheme": "node-and-date",
        "write_to": "lib/matplotlib/_version.py",
        "parentdir_prefix_version": "matplotlib-",
        "fallback_version": "0.0+UNKNOWN",
    },
    cmdclass={
        "test": NoopTestCommand,
        "build_ext": BuildExtraLibraries,
        "build_py": BuildPy,
        "sdist": Sdist,
    },
)
```
### 2 - lib/matplotlib/_cm.py:

Start line: 1363, End line: 1441

```python
datad = {
    'Blues': _Blues_data,
    'BrBG': _BrBG_data,
    'BuGn': _BuGn_data,
    'BuPu': _BuPu_data,
    'CMRmap': _CMRmap_data,
    'GnBu': _GnBu_data,
    'Greens': _Greens_data,
    'Greys': _Greys_data,
    'OrRd': _OrRd_data,
    'Oranges': _Oranges_data,
    'PRGn': _PRGn_data,
    'PiYG': _PiYG_data,
    'PuBu': _PuBu_data,
    'PuBuGn': _PuBuGn_data,
    'PuOr': _PuOr_data,
    'PuRd': _PuRd_data,
    'Purples': _Purples_data,
    'RdBu': _RdBu_data,
    'RdGy': _RdGy_data,
    'RdPu': _RdPu_data,
    'RdYlBu': _RdYlBu_data,
    'RdYlGn': _RdYlGn_data,
    'Reds': _Reds_data,
    'Spectral': _Spectral_data,
    'Wistia': _wistia_data,
    'YlGn': _YlGn_data,
    'YlGnBu': _YlGnBu_data,
    'YlOrBr': _YlOrBr_data,
    'YlOrRd': _YlOrRd_data,
    'afmhot': _afmhot_data,
    'autumn': _autumn_data,
    'binary': _binary_data,
    'bone': _bone_data,
    'brg': _brg_data,
    'bwr': _bwr_data,
    'cool': _cool_data,
    'coolwarm': _coolwarm_data,
    'copper': _copper_data,
    'cubehelix': _cubehelix_data,
    'flag': _flag_data,
    'gist_earth': _gist_earth_data,
    'gist_gray': _gist_gray_data,
    'gist_heat': _gist_heat_data,
    'gist_ncar': _gist_ncar_data,
    'gist_rainbow': _gist_rainbow_data,
    'gist_stern': _gist_stern_data,
    'gist_yarg': _gist_yarg_data,
    'gnuplot': _gnuplot_data,
    'gnuplot2': _gnuplot2_data,
    'gray': _gray_data,
    'hot': _hot_data,
    'hsv': _hsv_data,
    'jet': _jet_data,
    'nipy_spectral': _nipy_spectral_data,
    'ocean': _ocean_data,
    'pink': _pink_data,
    'prism': _prism_data,
    'rainbow': _rainbow_data,
    'seismic': _seismic_data,
    'spring': _spring_data,
    'summer': _summer_data,
    'terrain': _terrain_data,
    'winter': _winter_data,
    # Qualitative
    'Accent': {'listed': _Accent_data},
    'Dark2': {'listed': _Dark2_data},
    'Paired': {'listed': _Paired_data},
    'Pastel1': {'listed': _Pastel1_data},
    'Pastel2': {'listed': _Pastel2_data},
    'Set1': {'listed': _Set1_data},
    'Set2': {'listed': _Set2_data},
    'Set3': {'listed': _Set3_data},
    'tab10': {'listed': _tab10_data},
    'tab20': {'listed': _tab20_data},
    'tab20b': {'listed': _tab20b_data},
    'tab20c': {'listed': _tab20c_data},
}
```
### 3 - setupext.py:

Start line: 360, End line: 470

```python
class Matplotlib(SetupPackage):
    name = "matplotlib"

    def get_package_data(self):
        return {
            'matplotlib': [
                'mpl-data/matplotlibrc',
                *_pkg_data_helper('matplotlib', 'mpl-data'),
                *_pkg_data_helper('matplotlib', 'backends/web_backend'),
                '*.dll',  # Only actually matters on Windows.
            ],
        }

    def get_extensions(self):
        # agg
        ext = Extension(
            "matplotlib.backends._backend_agg", [
                "src/py_converters.cpp",
                "src/_backend_agg.cpp",
                "src/_backend_agg_wrapper.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags_and_sources(ext)
        FreeType.add_flags(ext)
        yield ext
        # c_internal_utils
        ext = Extension(
            "matplotlib._c_internal_utils", ["src/_c_internal_utils.c"],
            libraries=({
                "linux": ["dl"],
                "win32": ["ole32", "shell32", "user32"],
            }.get(sys.platform, [])))
        yield ext
        # contour
        ext = Extension(
            "matplotlib._contour", [
                "src/_contour.cpp",
                "src/_contour_wrapper.cpp",
                "src/py_converters.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags(ext)
        yield ext
        # ft2font
        ext = Extension(
            "matplotlib.ft2font", [
                "src/ft2font.cpp",
                "src/ft2font_wrapper.cpp",
                "src/py_converters.cpp",
            ])
        FreeType.add_flags(ext)
        add_numpy_flags(ext)
        add_libagg_flags(ext)
        yield ext
        # image
        ext = Extension(
            "matplotlib._image", [
                "src/_image_wrapper.cpp",
                "src/py_converters.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags_and_sources(ext)
        yield ext
        # path
        ext = Extension(
            "matplotlib._path", [
                "src/py_converters.cpp",
                "src/_path_wrapper.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags_and_sources(ext)
        yield ext
        # qhull
        ext = Extension(
            "matplotlib._qhull", ["src/qhull_wrap.cpp"],
            define_macros=[("MPL_DEVNULL", os.devnull)])
        add_numpy_flags(ext)
        Qhull.add_flags(ext)
        yield ext
        # tkagg
        ext = Extension(
            "matplotlib.backends._tkagg", [
                "src/_tkagg.cpp",
            ],
            include_dirs=["src"],
            # psapi library needed for finding Tcl/Tk at run time.
            libraries={"linux": ["dl"], "win32": ["comctl32", "psapi"],
                       "cygwin": ["comctl32", "psapi"]}.get(sys.platform, []),
            extra_link_args={"win32": ["-mwindows"]}.get(sys.platform, []))
        add_numpy_flags(ext)
        add_libagg_flags(ext)
        yield ext
        # tri
        ext = Extension(
            "matplotlib._tri", [
                "src/tri/_tri.cpp",
                "src/tri/_tri_wrapper.cpp",
            ])
        add_numpy_flags(ext)
        yield ext
        # ttconv
        ext = Extension(
            "matplotlib._ttconv", [
                "src/_ttconv.cpp",
                "extern/ttconv/pprdrv_tt.cpp",
                "extern/ttconv/pprdrv_tt2.cpp",
                "extern/ttconv/ttutil.cpp",
            ],
            include_dirs=["extern"])
        add_numpy_flags(ext)
        yield ext
```
### 4 - lib/matplotlib/pyplot.py:

Start line: 1, End line: 85

```python
# Note: The first part of this file can be modified in place, but the latter
# part is autogenerated by the boilerplate.py script.

"""
`matplotlib.pyplot` is a state-based interface to matplotlib. It provides
an implicit,  MATLAB-like, way of plotting.  It also opens figures on your
screen, and acts as the figure GUI manager.

pyplot is mainly intended for interactive plots and simple cases of
programmatic plot generation::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.plot(x, y)

The explicit (object-oriented) API is recommended for complex plots, though
pyplot is still usually used to create the figure and often the axes in the
figure. See `.pyplot.figure`, `.pyplot.subplots`, and
`.pyplot.subplot_mosaic` to create figures, and
:doc:`Axes API <../axes_api>` for the plotting methods on an axes::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
"""

import functools
import importlib
import inspect
import logging
from numbers import Number
import re
import sys
import time
try:
    import threading
except ImportError:
    import dummy_threading as threading

from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import rcsetup, style
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import docstring
from matplotlib.backend_bases import FigureCanvasBase, MouseButton
from matplotlib.figure import Figure, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcParams, rcParamsDefault, get_backend, rcParamsOrig
from matplotlib.rcsetup import interactive_bk as _interactive_bk
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot
from matplotlib.projections import PolarAxes
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names

from matplotlib import cm
from matplotlib.cm import _colormaps as colormaps, get_cmap, register_cmap

import numpy as np

# We may not need the following imports here:
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import SubplotTool, Button, Slider, Widget

from .ticker import (
    TickHelper, Formatter, FixedFormatter, NullFormatter, FuncFormatter,
    FormatStrFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent,
    LogFormatterMathtext, Locator, IndexLocator, FixedLocator, NullLocator,
    LinearLocator, LogLocator, AutoLocator, MultipleLocator, MaxNLocator)

_log = logging.getLogger(__name__)
```
### 5 - doc/conf.py:

Start line: 117, End line: 193

```python
_check_dependencies()


# Import only after checking for dependencies.
# gallery_order.py from the sphinxext folder provides the classes that
# allow custom ordering of sections and subsections of the gallery
import sphinxext.gallery_order as gallery_order

# The following import is only necessary to monkey patch the signature later on
from sphinx_gallery import gen_rst

# On Linux, prevent plt.show() from emitting a non-GUI backend warning.
os.environ.pop("DISPLAY", None)

autosummary_generate = True

# we should ignore warnings coming from importing deprecated modules for
# autodoc purposes, as this will disappear automatically when they are removed
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='importlib',  # used by sphinx.autodoc.importer
                        message=r'(\n|.)*module was deprecated.*')

autodoc_docstring_signature = True
autodoc_default_options = {'members': None, 'undoc-members': None}

# make sure to ignore warnings that stem from simply inspecting deprecated
# class-level attributes
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='sphinx.util.inspect')

# missing-references names matches sphinx>=3 behavior, so we can't be nitpicky
# for older sphinxes.
nitpicky = sphinx.version_info >= (3,)
# change this to True to update the allowed failures
missing_references_write_json = False
missing_references_warn_unused_ignores = False

intersphinx_mapping = {
    'Pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'cycler': ('https://matplotlib.org/cycler/', None),
    'dateutil': ('https://dateutil.readthedocs.io/en/stable/', None),
    'ipykernel': ('https://ipykernel.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pytest': ('https://pytest.org/en/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'tornado': ('https://www.tornadoweb.org/en/stable/', None),
}


# Sphinx gallery configuration

sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../tutorials', '../plot_types'],
    'filename_pattern': '^((?!sgskip).)*$',
    'gallery_dirs': ['gallery', 'tutorials', 'plot_types'],
    'doc_module': ('matplotlib', 'mpl_toolkits'),
    'reference_url': {
        'matplotlib': None,
        'numpy': 'https://numpy.org/doc/stable/',
        'scipy': 'https://docs.scipy.org/doc/scipy/reference/',
    },
    'backreferences_dir': Path('api') / Path('_as_gen'),
    'subsection_order': gallery_order.sectionorder,
    'within_subsection_order': gallery_order.subsectionorder,
    'remove_config_comments': True,
    'min_reported_time': 1,
    'thumbnail_size': (320, 224),
    # Compression is a significant effort that we skip for local and CI builds.
    'compress_images': ('thumbnails', 'images') if is_release_build else (),
    'matplotlib_animations': True,
    'image_srcset': ["2x"],
    'junit': '../test-results/sphinx-gallery/junit.xml' if CIRCLECI else '',
}

plot_gallery = 'True'
```
### 6 - tutorials/introductory/sample_plots.py:

Start line: 1, End line: 363

```python
"""
==========================
Sample plots in Matplotlib
==========================

Here you'll find a host of example plots with the code that
generated them.

.. _matplotlibscreenshots:

Line Plot
=========

Here's how to create a line plot with text labels using
:func:`~matplotlib.pyplot.plot`.

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_simple_plot_001.png
   :target: ../../gallery/lines_bars_and_markers/simple_plot.html
   :align: center

.. _screenshots_subplot_demo:

Multiple subplots in one figure
===============================

Multiple axes (i.e. subplots) are created with the
:func:`~matplotlib.pyplot.subplot` function:

.. figure:: ../../gallery/subplots_axes_and_figures/images/sphx_glr_subplot_001.png
   :target: ../../gallery/subplots_axes_and_figures/subplot.html
   :align: center

.. _screenshots_images_demo:

Images
======

Matplotlib can display images (assuming equally spaced
horizontal dimensions) using the :func:`~matplotlib.pyplot.imshow` function.

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_image_demo_003.png
   :target: ../../gallery/images_contours_and_fields/image_demo.html
   :align: center

   Example of using :func:`~matplotlib.pyplot.imshow` to display an MRI

.. _screenshots_pcolormesh_demo:


Contouring and pseudocolor
==========================

The :func:`~matplotlib.pyplot.pcolormesh` function can make a colored
representation of a two-dimensional array, even if the horizontal dimensions
are unevenly spaced.  The
:func:`~matplotlib.pyplot.contour` function is another way to represent
the same data:

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_pcolormesh_levels_001.png
   :target: ../../gallery/images_contours_and_fields/pcolormesh_levels.html
   :align: center

.. _screenshots_histogram_demo:

Histograms
==========

The :func:`~matplotlib.pyplot.hist` function automatically generates
histograms and returns the bin counts or probabilities:

.. figure:: ../../gallery/statistics/images/sphx_glr_histogram_features_001.png
   :target: ../../gallery/statistics/histogram_features.html
   :align: center

.. _screenshots_path_demo:

Paths
=====

You can add arbitrary paths in Matplotlib using the
:mod:`matplotlib.path` module:

.. figure:: ../../gallery/shapes_and_collections/images/sphx_glr_path_patch_001.png
   :target: ../../gallery/shapes_and_collections/path_patch.html
   :align: center

.. _screenshots_mplot3d_surface:

Three-dimensional plotting
==========================

The mplot3d toolkit (see :doc:`/tutorials/toolkits/mplot3d` and
:ref:`mplot3d-examples-index`) has support for simple 3D graphs
including surface, wireframe, scatter, and bar charts.

.. figure:: ../../gallery/mplot3d/images/sphx_glr_surface3d_001.png
   :target: ../../gallery/mplot3d/surface3d.html
   :align: center

Thanks to John Porter, Jonathon Taylor, Reinier Heeres, and Ben Root for
the `.mplot3d` toolkit. This toolkit is included with all standard Matplotlib
installs.

.. _screenshots_ellipse_demo:


Streamplot
==========

The :meth:`~matplotlib.pyplot.streamplot` function plots the streamlines of
a vector field. In addition to simply plotting the streamlines, it allows you
to map the colors and/or line widths of streamlines to a separate parameter,
such as the speed or local intensity of the vector field.

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_plot_streamplot_001.png
   :target: ../../gallery/images_contours_and_fields/plot_streamplot.html
   :align: center

   Streamplot with various plotting options.

This feature complements the :meth:`~matplotlib.pyplot.quiver` function for
plotting vector fields. Thanks to Tom Flannaghan and Tony Yu for adding the
streamplot function.


Ellipses
========

In support of the `Phoenix <https://www.jpl.nasa.gov/news/phoenix/main.php>`_
mission to Mars (which used Matplotlib to display ground tracking of
spacecraft), Michael Droettboom built on work by Charlie Moad to provide
an extremely accurate 8-spline approximation to elliptical arcs (see
:class:`~matplotlib.patches.Arc`), which are insensitive to zoom level.

.. figure:: ../../gallery/shapes_and_collections/images/sphx_glr_ellipse_demo_001.png
   :target: ../../gallery/shapes_and_collections/ellipse_demo.html
   :align: center

.. _screenshots_barchart_demo:

Bar charts
==========

Use the :func:`~matplotlib.pyplot.bar` function to make bar charts, which
includes customizations such as error bars:

.. figure:: ../../gallery/statistics/images/sphx_glr_barchart_demo_001.png
   :target: ../../gallery/statistics/barchart_demo.html
   :align: center

You can also create stacked bars
(`bar_stacked.py <../../gallery/lines_bars_and_markers/bar_stacked.html>`_),
or horizontal bar charts
(`barh.py <../../gallery/lines_bars_and_markers/barh.html>`_).

.. _screenshots_pie_demo:


Pie charts
==========

The :func:`~matplotlib.pyplot.pie` function allows you to create pie
charts.  Optional features include auto-labeling the percentage of area,
exploding one or more wedges from the center of the pie, and a shadow effect.
Take a close look at the attached code, which generates this figure in just
a few lines of code.

.. figure:: ../../gallery/pie_and_polar_charts/images/sphx_glr_pie_features_001.png
   :target: ../../gallery/pie_and_polar_charts/pie_features.html
   :align: center

.. _screenshots_table_demo:

Tables
======

The :func:`~matplotlib.pyplot.table` function adds a text table
to an axes.

.. figure:: ../../gallery/misc/images/sphx_glr_table_demo_001.png
   :target: ../../gallery/misc/table_demo.html
   :align: center

.. _screenshots_scatter_demo:

Scatter plots
=============

The :func:`~matplotlib.pyplot.scatter` function makes a scatter plot
with (optional) size and color arguments. This example plots changes
in Google's stock price, with marker sizes reflecting the
trading volume and colors varying with time. Here, the
alpha attribute is used to make semitransparent circle markers.

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_scatter_demo2_001.png
   :target: ../../gallery/lines_bars_and_markers/scatter_demo2.html
   :align: center

.. _screenshots_slider_demo:

GUI widgets
===========

Matplotlib has basic GUI widgets that are independent of the graphical
user interface you are using, allowing you to write cross GUI figures
and widgets.  See :mod:`matplotlib.widgets` and the
`widget examples <../../gallery/index.html#widgets>`_.

.. figure:: ../../gallery/widgets/images/sphx_glr_slider_demo_001.png
   :target: ../../gallery/widgets/slider_demo.html
   :align: center

   Slider and radio-button GUI.


.. _screenshots_fill_demo:

Filled curves
=============

The :func:`~matplotlib.pyplot.fill` function lets you
plot filled curves and polygons:

.. figure:: ../../gallery/lines_bars_and_markers/images/sphx_glr_fill_001.png
   :target: ../../gallery/lines_bars_and_markers/fill.html
   :align: center

Thanks to Andrew Straw for adding this function.

.. _screenshots_date_demo:

Date handling
=============

You can plot timeseries data with major and minor ticks and custom
tick formatters for both.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_date_001.png
   :target: ../../gallery/text_labels_and_annotations/date.html
   :align: center

See :mod:`matplotlib.ticker` and :mod:`matplotlib.dates` for details and usage.


.. _screenshots_log_demo:

Log plots
=========

The :func:`~matplotlib.pyplot.semilogx`,
:func:`~matplotlib.pyplot.semilogy` and
:func:`~matplotlib.pyplot.loglog` functions simplify the creation of
logarithmic plots.

.. figure:: ../../gallery/scales/images/sphx_glr_log_demo_001.png
   :target: ../../gallery/scales/log_demo.html
   :align: center

Thanks to Andrew Straw, Darren Dale and Gregory Lielens for contributions
log-scaling infrastructure.

.. _screenshots_polar_demo:

Polar plots
===========

The :func:`~matplotlib.pyplot.polar` function generates polar plots.

.. figure:: ../../gallery/pie_and_polar_charts/images/sphx_glr_polar_demo_001.png
   :target: ../../gallery/pie_and_polar_charts/polar_demo.html
   :align: center

.. _screenshots_legend_demo:


Legends
=======

The :func:`~matplotlib.pyplot.legend` function automatically
generates figure legends, with MATLAB-compatible legend-placement
functions.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_legend_001.png
   :target: ../../gallery/text_labels_and_annotations/legend.html
   :align: center

Thanks to Charles Twardy for input on the legend function.

.. _screenshots_mathtext_examples_demo:

TeX-notation for text objects
=============================

Below is a sampling of the many TeX expressions now supported by Matplotlib's
internal mathtext engine.  The mathtext module provides TeX style mathematical
expressions using `FreeType <https://www.freetype.org/>`_
and the DejaVu, BaKoMa computer modern, or `STIX <http://www.stixfonts.org>`_
fonts.  See the :mod:`matplotlib.mathtext` module for additional details.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_mathtext_examples_001.png
   :target: ../../gallery/text_labels_and_annotations/mathtext_examples.html
   :align: center

Matplotlib's mathtext infrastructure is an independent implementation and
does not require TeX or any external packages installed on your computer. See
the tutorial at :doc:`/tutorials/text/mathtext`.


.. _screenshots_tex_demo:

Native TeX rendering
====================

Although Matplotlib's internal math rendering engine is quite
powerful, sometimes you need TeX. Matplotlib supports external TeX
rendering of strings with the *usetex* option.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_tex_demo_001.png
   :target: ../../gallery/text_labels_and_annotations/tex_demo.html
   :align: center

.. _screenshots_eeg_demo:

EEG GUI
=======

You can embed Matplotlib into Qt, GTK, Tk, or wxWidgets applications.
Here is a screenshot of an EEG viewer called `pbrain
<https://github.com/nipy/pbrain>`__.

.. image:: ../../_static/eeg_small.png
   :align: center

The lower axes uses :func:`~matplotlib.pyplot.specgram`
to plot the spectrogram of one of the EEG channels.

For examples of how to embed Matplotlib in different toolkits, see:

   * :doc:`/gallery/user_interfaces/embedding_in_gtk4_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_gtk3_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_wx2_sgskip`
   * :doc:`/gallery/user_interfaces/mpl_with_glade3_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_qt_sgskip`
   * :doc:`/gallery/user_interfaces/embedding_in_tk_sgskip`

XKCD-style sketch plots
=======================

Just for fun, Matplotlib supports plotting in the style of `xkcd
<https://xkcd.com/>`_.

.. figure:: ../../gallery/showcase/images/sphx_glr_xkcd_001.png
   :target: ../../gallery/showcase/xkcd.html
   :align: center

Subplot example
===============

Many plot types can be combined in one figure to create
powerful and flexible representations of data.
"""

import matplotlib.pyplot as plt
```
### 7 - lib/matplotlib/pyplot.py:

Start line: 3013, End line: 3077

```python
# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes._sci)
def sci(im):
    return gca()._sci(im)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.set_title)
def title(label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
    return gca().set_title(
        label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.set_xlabel)
def xlabel(xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
    return gca().set_xlabel(
        xlabel, fontdict=fontdict, labelpad=labelpad, loc=loc,
        **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.set_ylabel)
def ylabel(ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
    return gca().set_ylabel(
        ylabel, fontdict=fontdict, labelpad=labelpad, loc=loc,
        **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.set_xscale)
def xscale(value, **kwargs):
    return gca().set_xscale(value, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
@_copy_docstring_and_deprecators(Axes.set_yscale)
def yscale(value, **kwargs):
    return gca().set_yscale(value, **kwargs)


# Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
def autumn(): set_cmap('autumn')
def bone(): set_cmap('bone')
def cool(): set_cmap('cool')
def copper(): set_cmap('copper')
def flag(): set_cmap('flag')
def gray(): set_cmap('gray')
def hot(): set_cmap('hot')
def hsv(): set_cmap('hsv')
def jet(): set_cmap('jet')
def pink(): set_cmap('pink')
def prism(): set_cmap('prism')
def spring(): set_cmap('spring')
def summer(): set_cmap('summer')
def winter(): set_cmap('winter')
def magma(): set_cmap('magma')
def inferno(): set_cmap('inferno')
def plasma(): set_cmap('plasma')
def viridis(): set_cmap('viridis')
def nipy_spectral(): set_cmap('nipy_spectral')


_setup_pyplot_info_docstrings()
```
### 8 - doc/conf.py:

Start line: 1, End line: 87

```python
# Matplotlib documentation build configuration file, created by
# sphinx-quickstart on Fri May  2 12:33:25 2008.
#
# This file is execfile()d with the current directory set to its containing
# dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed
# automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import matplotlib
import sphinx

from datetime import datetime
import time

# Release mode enables optimizations and other related options.
is_release_build = tags.has('release')  # noqa

# are we running circle CI?
CIRCLECI = 'CIRCLECI' in os.environ

# Parse year using SOURCE_DATE_EPOCH, falling back to current time.
# https://reproducible-builds.org/specs/source-date-epoch/
sourceyear = datetime.utcfromtimestamp(
    int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))).year

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('.'))
sys.path.append('.')

# General configuration
# ---------------------

# Unless we catch the warning explicitly somewhere, a warning should cause the
# docs build to fail. This is especially useful for getting rid of deprecated
# usage in the gallery.
warnings.filterwarnings('error', append=True)

# Strip backslashes in function's signature
# To be removed when numpydoc > 0.9.x
strip_signature_backslash = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.inkscapeconverter',
    'sphinxext.custom_roles',
    'sphinxext.github',
    'sphinxext.math_symbol_table',
    'sphinxext.missing_references',
    'sphinxext.mock_gui_toolkits',
    'sphinxext.skip_deprecated',
    'sphinxext.redirect_from',
    'sphinx_copybutton',
    'sphinx_panels',
]

exclude_patterns = [
    'api/prev_api_changes/api_changes_*/*',
]

panels_add_bootstrap_css = False
```
### 9 - lib/matplotlib/__init__.py:

Start line: 1, End line: 115

```python
"""
An object-oriented plotting library.

A procedural interface is provided by the companion pyplot module,
which may be imported directly, e.g.::

    import matplotlib.pyplot as plt

or using ipython::

    ipython

at your terminal, followed by::

    In [1]: %matplotlib
    In [2]: import matplotlib.pyplot as plt

at the ipython shell prompt.

For the most part, direct use of the object-oriented library is encouraged when
programming; pyplot is primarily for working interactively.  The exceptions are
the pyplot functions `.pyplot.figure`, `.pyplot.subplot`, `.pyplot.subplots`,
and `.pyplot.savefig`, which can greatly simplify scripting.

Modules include:

    :mod:`matplotlib.axes`
        The `~.axes.Axes` class.  Most pyplot functions are wrappers for
        `~.axes.Axes` methods.  The axes module is the highest level of OO
        access to the library.

    :mod:`matplotlib.figure`
        The `.Figure` class.

    :mod:`matplotlib.artist`
        The `.Artist` base class for all classes that draw things.

    :mod:`matplotlib.lines`
        The `.Line2D` class for drawing lines and markers.

    :mod:`matplotlib.patches`
        Classes for drawing polygons.

    :mod:`matplotlib.text`
        The `.Text` and `.Annotation` classes.

    :mod:`matplotlib.image`
        The `.AxesImage` and `.FigureImage` classes.

    :mod:`matplotlib.collections`
        Classes for efficient drawing of groups of lines or polygons.

    :mod:`matplotlib.colors`
        Color specifications and making colormaps.

    :mod:`matplotlib.cm`
        Colormaps, and the `.ScalarMappable` mixin class for providing color
        mapping functionality to other classes.

    :mod:`matplotlib.ticker`
        Calculation of tick mark locations and formatting of tick labels.

    :mod:`matplotlib.backends`
        A subpackage with modules for various GUI libraries and output formats.

The base matplotlib namespace includes:

    `~matplotlib.rcParams`
        Default configuration settings; their defaults may be overridden using
        a :file:`matplotlibrc` file.

    `~matplotlib.use`
        Setting the Matplotlib backend.  This should be called before any
        figure is created, because it is not possible to switch between
        different GUI backends after that.

Matplotlib was initially written by John D. Hunter (1968-2012) and is now
developed and maintained by a host of others.

Occasionally the internal documentation (python docstrings) will refer
to MATLAB&reg;, a registered trademark of The MathWorks, Inc.
"""

import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy
from packaging.version import parse as parse_version

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from . import _api, _version, cbook, docstring, rcsetup
from matplotlib.cbook import MatplotlibDeprecationWarning, sanitize_sequence
from matplotlib.cbook import mplDeprecation  # deprecated
from matplotlib.rcsetup import validate_backend, cycler


_log = logging.getLogger(__name__)
```
### 10 - doc/conf.py:

Start line: 194, End line: 303

```python
mathmpl_fontsize = 11.0
mathmpl_srcset = ['2x']

# Monkey-patching gallery signature to include search keywords
gen_rst.SPHX_GLR_SIG = """\n
.. only:: html

 .. rst-class:: sphx-glr-signature

    Keywords: matplotlib code example, codex, python plot, pyplot
    `Gallery generated by Sphinx-Gallery
    <https://sphinx-gallery.readthedocs.io>`_\n"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# This is the default encoding, but it doesn't hurt to be explicit
source_encoding = "utf-8"

# The master toctree document.
master_doc = 'users/index'

# General substitutions.
try:
    SHA = subprocess.check_output(
        ['git', 'describe', '--dirty']).decode('utf-8').strip()
# Catch the case where git is not installed locally, and use the setuptools_scm
# version number instead
except (subprocess.CalledProcessError, FileNotFoundError):
    SHA = matplotlib.__version__

html_context = {
    "sha": SHA,
}

project = 'Matplotlib'
copyright = (
    '2002 - 2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom '
    'and the Matplotlib development team; '
    f'2012 - {sourceyear} The Matplotlib development team'
)


# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.

version = matplotlib.__version__
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True
pygments_style = 'sphinx'

default_role = 'obj'

# Plot directive configuration
# ----------------------------

plot_formats = [('png', 100), ('pdf', 100)]

# GitHub extension

github_project_url = "https://github.com/matplotlib/matplotlib/"

# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
# html_style = 'matplotlib.css'
# html_style = f"mpl.css?{SHA}"
html_css_files = [
    f"mpl.css?{SHA}",
]

html_theme = "mpl_sphinx_theme"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = "_static/logo2.svg"
```
