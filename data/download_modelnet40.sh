#!/bin/bash
curl -L -o ~/paconv/data/modelnet40.zip https://www.kaggle.com/api/v1/datasets/download/cuge1995/modelnet40
unzip ~/paconv/data/modelnet40.zip -d ~/paconv/data/modelnet40_ply_hdf5_2048
rm ~/paconv/data/modelnet40.zip