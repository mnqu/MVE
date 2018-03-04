#!/bin/sh

./mve/mve -network data/toy/view_ -views 3 -label data/toy/label.txt -output node.emb -binary 1 -size 100 -negative 5 -depth 1 -samples 30 -epochs 20 -threads 20 -norm 1

