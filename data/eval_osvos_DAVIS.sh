#!/bin/bash

cd davis2017-evaluation/
rm results/semi-supervised/osvos/*results-val.csv
python evaluation_method.py --davis_path ../data_storage/DAVIS --results_path results/semi-supervised/osvos --task semi-supervised
