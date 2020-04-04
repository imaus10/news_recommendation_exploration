#!/usr/bin/env bash

mkdir -p data
curl -o data/adressa_one_week.tar.gz http://reclab.idi.ntnu.no/dataset/one_week.tar.gz
tar xvzf data/adressa_one_week.tar.gz -C data
rm data/adressa_one_week.tar.gz
mv data/one_week data/adressa_one_week
