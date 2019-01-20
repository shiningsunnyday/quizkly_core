# Kvizo Quiz Generation Library

## Overview

`datasets`: code to create datasets for training models\
`filters`: code for filtering gap/distractor candidates\
`models`: core machine learning models\
`proto`: protobufs used for training and web deployment\
`service`: clients to trained models\

## Setting Up
Install protobuf with `brew install protobuf`\
Install required libraries with `pip install -r requirements.txt`\
Build protos and test models/data with `./build.sh`\
Check that all tests pass with `./test.sh`
