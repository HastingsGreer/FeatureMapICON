# FeatureMapICON

training requires getting pykeops to work.
[<img src="https://github.com/uncbiag/FeatureMapICON/actions/workflows/selfhosted-action.yml/badge.svg">](https://github.com/uncbiag/FeatureMapICON/actions)

# How to run our code
```
pip install -e .
git submodule sync
git submodule update --init --recursive

data/download_preprocess_DAVIS.sh

echo foo | python -m unittest discover
#data/eval_osvos_DAVIS.sh
```
