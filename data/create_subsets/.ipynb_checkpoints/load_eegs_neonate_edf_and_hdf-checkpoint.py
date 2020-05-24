# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python [conda env:mne-2]
#     language: python
#     name: conda-env-mne-2-py
# ---

# %% [markdown] {"colab_type": "text", "id": "SYgxVvZI2r3K"}
# # Loading EEGs 
# The purpose of this notebook is to show how to load EEG files in the eeg.h5 format (eeg-hdfstorage) and in edf format using edflib (or pyedf).
#
# Note that one problem with edf format is it seems that there is enough variance in following it that it is not always loaded easily.
#
# Note that the pip installs need to be done, then the runtime needs restarting to allow for import

# %% {"colab": {}, "colab_type": "code", "id": "_eLRkfQ5FEr6"}
import eeghdf
import eegvis
import eegvis.stacklineplot
import eegvis.stackplot_bokeh as splotb
import eegvis.nb_eegview as nb_eegview
import edflib  # pyedf is also an option

# %% {"colab": {}, "colab_type": "code", "id": "v8aF-fp7Sa9Q"}
from bokeh.io import output_notebook, push_notebook
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)
output_notebook() # note in ordinary jupyter notebook only need this once, in colab needed for each cell with bokeh


# %%
## configuration
from dynaconf import settings

EEGML_TEMPLE_SZv151 = "/mnt/data1/eegdbs/TUH/temple/tuh_eeg_seizure/v1.5.1"

# %%
settings.as_dict()

# %%
settings.COMMENTJSON_ENABLED_FOR_DYNACONF

# %%

# %%

# %%
settings.

# %% {"colab": {}, "colab_type": "code", "id": "iWiqgdeS60E9"}
ef = edflib.EdfReader('/mnt/data1/eegdbs/stevenson_neonatal_eeg/edf/eeg10.edf')

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "aI5PiQ9E60MH", "outputId": "d1191754-f4f6-4948-b453-80d0f6ca7e4f"}
ef.read_annotations()

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 374}, "colab_type": "code", "id": "TuPdjP8h60O6", "outputId": "5bf5dc0c-0d3e-43bd-8661-f3bc78dcb9b1"}
labels = ef.get_signal_text_labels()
labels

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 68}, "colab_type": "code", "id": "gS4apn7160Rs", "outputId": "53090a54-d8b4-4b70-db46-0378dd260d72"}
ef.get_samples_per_signal()

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 51}, "colab_type": "code", "id": "nlGv2EzM60Uh", "outputId": "6ff3acee-28c8-4185-f781-ffa3ef2080d3"}
ef.get_signal_freqs()

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "jmaGFIEY-HA1", "outputId": "3a1b1463-3de5-41db-d33d-a88b721c53bc"}
N=19
fs = int(ef.get_signal_freqs()[0])
assert np.all(ef.get_signal_freqs() == fs)
(N,fs)

# %% {"colab": {}, "colab_type": "code", "id": "mHY-zZVO60YH"}
# this method assumes we have lots of RAM as will load entire EEG into memory, won't work for very long
slist = [ef.get_signal(ii) for ii in range(N)]

## a different method to load a portion of each channel into an array
#slist = []
#L = fs * 50 # num of samples
#start_sample = 0

#for signum in range(N):
#    dest = np.empty((L,), dtype=np.float64) # buffer
#    ef.read_phys_signal(signum, start_sample, L, dest)
#    slist.append(dest)

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 102}, "colab_type": "code", "id": "R5S9zHUt60ec", "outputId": "1d168521-91f7-4c03-bbf7-e3653d29c33f"}
slist[0][:20]

# %% {"colab": {}, "colab_type": "code", "id": "GrzNxQmc9Ysh"}
M = ef.get_samples_per_signal()[0]
S = np.empty((N,M))



# %% {"colab": {}, "colab_type": "code", "id": "-7wN6zPI9Y1I"}
# create a single matrix with all the waveform data
for ii in range(N):
  S[ii,:] = slist[ii]
  

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 502}, "colab_type": "code", "id": "NXVmahRj_mEu", "outputId": "6d733664-a0b6-4ff0-f75b-54472d53c183"}
# plot a single channel in raw form to look at its values
plt.plot(slist[0][256*30:256*(30+10)])

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 514}, "colab_type": "code", "id": "p0qdauHy9Y4Q", "outputId": "1a477f13-e908-419b-8280-eadc80ec1236"}
import eegvis.stacklineplot
t0 =30   # start time for plotting in seconds
T = 15.0 # page length in seconds
eegvis.stacklineplot.stackplot(S[:,t0*fs:fs*(t0+int(15))],seconds=T,start_time=t0,
                                 ylabels=ef.get_signal_text_labels(),
                                 yscale=1.5,
                                 )

# %% [markdown] {"colab_type": "text", "id": "LtmTa2y9CszR"}
# ### Demonstrate how to use eeg hdf5 storage
# This form of storage has multiple advantages. It is well defined and supported by virtually all languages. 
#
# It allows for accessing waveform data without reading in the entirety of the image as if it was a continuous array. Automatic conversion to physical units (usually microvolts) is available as well, again simulating a numpy like array interface.

# %% {"colab": {}, "colab_type": "code", "id": "tEGiMlkNFVSa"}
hf = eeghdf.Eeghdf('/mnt/data1/eegdbs/stevenson_neonatal_eeg/hdf/eeg10.eeg.h5')

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "4sMBuzC6Fa-K", "outputId": "4498d06c-9288-48af-cfdb-b3914327c8ea"}
hf.age_years

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 34}, "colab_type": "code", "id": "QcKawC4AF0ca", "outputId": "ece261b9-e619-47ca-8707-906d6ae2c918"}
hf.duration_seconds_float


# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 374}, "colab_type": "code", "id": "5zuD5kUpGetx", "outputId": "371a3b1b-54a1-4c76-9504-7a116f95c89a"}
hf.physical_dimensions

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 374}, "colab_type": "code", "id": "GqT_DrtEGoOi", "outputId": "8abc51c5-2ade-405f-be0e-c7d8aae1f6ad"}
hf.electrode_labels

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 374}, "colab_type": "code", "id": "Y3CAg-ZSGKc6", "outputId": "58ffa55f-69c1-4860-dc7d-e42dc401a4c4"}
hf.shortcut_elabels 

# %% [markdown] {"colab_type": "text", "id": "ghBCvMBVYDl9"}
#

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 49}, "colab_type": "code", "id": "SR0Vb102GqWy", "outputId": "dfb8c2a6-1820-4bc7-a927-3d35592c8793"}
hf.annotations_contain('seizure')

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 49}, "colab_type": "code", "id": "gl5ldlIxGu0a", "outputId": "bec37bb0-de77-4d83-af66-b437e4856e18"}
hf.edf_annotations_df

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 49}, "colab_type": "code", "id": "cQqpJfpvSNBf", "outputId": "dcf733fb-447d-402b-f0f8-c492a706391b"}
hf._annotations_df

# %% [markdown] {"colab_type": "text", "id": "dMd40Uh5DxWS"}
# ## eegvis has tools for interactive visualization of EEG
# This is based around bokeh and allows for dynamic interaction in the notebook with some limitations in alternative notebooks like colab and azure notebooks.

# %% {"colab": {}, "colab_type": "code", "id": "G6T02EXCZtMc"}
fig=splotb.show_epoch_centered(hf.phys_signals, 20, epoch_width_sec=10, chstart=0, chstop=19,
                               fs=hf.sample_frequency,ylabels=hf.electrode_labels)

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 617}, "colab_type": "code", "id": "Ol4nNixzZtRi", "outputId": "c7525172-84f7-4014-a5f3-744c49010656"}
output_notebook() # needs to be called in the same cells as the widget
from bokeh.plotting import show
show(fig)

# note in google colab the interactive features don't work currently

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 681, "referenced_widgets": ["6499eef4be204e6aa59e47a7536167cf", "bcbabc8dde9f4559a2055f78b83d1a6e", "58565013175342c1ae65403ebf0c941c", "925ee6d9b7a54358b6bbe4e010c5a56f", "e9d55071765d49b38857f3c5fc5f831b", "b68338cd2dc348f59a9f5f4483b464b2", "4700f627519f437b82f8f403ddb5f285", "5bb35764bd52490699c54245bf23d082", "ca298ca324614cfe9f2e03b23be50d9b", "ce47a4470d724d82a43a9db4a8f771fa", "10ad72770fe5461d8665b9fd69fc64a4", "060cad27ba2c459496f8981ea8b7a666", "6d1f9339d6494251aa16e9b895551b09", "c1325ba84df949fbb56a1ce81db5f84a", "7362c286127349cbaee1b00a45ea0203", "ecf56d39d7074f90a2050aa63906a785", "80af4bff132946e28ade253162cff948", "9543a13539694511bdb90c72500bfc8a", "a43d135bf9eb47c09850e9daf01f1511", "54414a4672734f16bbf59c25525d6638", "d28ad710a1734c8891e38e63ec33f1ff", "c6994d7fc93e48719bbcf2e4720f470d", "0548e9c2a02c4d999214f7e2f54c5c41", "6b0ff8a055f64daf8feaaa0ab3f61278", "dc08b0fb1b514f4ba58c6561a7b06c4f", "f877d683433243b7bf10aba571dca13b", "36ecd8ffce464f3ca6c5147246f3472b", "4f0dd9cae1964d408ff4188999cde605", "2a27e71f65a843c7905ddc0cc3a86d7e"]}, "colab_type": "code", "id": "SV38R3CpTZ3g", "outputId": "ef565422-a884-4f72-f9e3-1905a6937dff"}
## An EEG browser (only works in pure jupyter notebook or jupyterlab)
output_notebook()
# current built in montage derivations: 'trace', 'tcp', double banana', laplacian', neonatal', 'DB-REF'
eegplot = nb_eegview.EeghdfBrowser(eeghdf_file=hf, montage='neonatal', yscale=3.0)
eegplot.loc_sec = 50
eegplot.show()

#### NOTE the menu/button interactive features don't seem to work in colab  ####

# %% {"colab": {}, "colab_type": "code", "id": "sPI4UVsfTn53"}
# in pure jupyter notebook can do things like this
eegplot.loc_sec = 100.0
eegplot.update()

# %% {"colab": {}, "colab_type": "code", "id": "9ubW3Ft5KwUu"}
