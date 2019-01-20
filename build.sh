#!/bin/bash -e
echo "Building protos."
protoc --python_out=. proto/question_candidate.proto

if [ "$1" == "protos" ]; then
	exit 0;
fi

echo "Building testdata."
python -m datasets.create_test_data

echo "Building test models."
python -m models.test_data.save_test_models

echo "Downloading sentence tokenizer data."
python -m nltk.downloader punkt

echo "Downloading wordnet."
python -m nltk.downloader punkt