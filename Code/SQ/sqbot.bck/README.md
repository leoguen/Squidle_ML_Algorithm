# SQBOT #
Library for integrating ML into SQ+

## Setup 
Get code (including submodules):
```shell
git clone --recursive https://ariell@bitbucket.org/ariell/sqbot.git
```


most machines and OS's (for M1 mac, see below)
```bash
virtualenv -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```
For **M1 MACOS**, need todo some tricks
```bash
brew install hdf5   # tensorflow dependency, for some reason fails, if this is not done first
export CPATH="/opt/homebrew/include/"
export HDF5_DIR=/opt/homebrew/
virtualenv -p python3.9 venv
source venv/bin/activate
pip install h5py
pip install tensorflow-macos
# tweak requirements.txt accordingly
pip install -r requirements.txt
```
