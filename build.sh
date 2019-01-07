#!/bin/bash -e
echo "Building protos."
protoc --python_out=. proto/question_candidate.proto

if [ "$1" == "protos" ]; then
	exit 0;
fi

echo "Building testdata."
python -m datasets.create_test_data

echo "Downloading sentence tokenizer data."
python -m nltk.downloader punkt