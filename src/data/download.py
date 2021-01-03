import requests
import os
import json
from clint.textui import progress
from pathlib import Path

"""
Downloads training, validation and test data if not available locally and stores them into 
the respective subdirectories in the tape_data directory.
"""

base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "tape_data"

def get_tape_data():
	if not os.path.isfile(data_path / "training" / "train.json"):
	    print("Downloading training set...")
	    url = 'https://polybox.ethz.ch/index.php/s/chy7bf5a4xKvuds/download'
	    r = requests.get(url, allow_redirects=True, stream=True)
	    with open(data_path / "training" / "train.json", 'wb') as f:
	        total_length = int(r.headers.get('content-length'))
	        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
	            if chunk:
	                f.write(chunk)
	                f.flush()
	    print("Finished downloading training set.\n")

	if not os.path.isfile(data_path / "validation" / "val.json"):
	    print("Downloading validation set...")
	    url = 'https://polybox.ethz.ch/index.php/s/4SrHcGRQNX84OF7/download'
	    r = requests.get(url, allow_redirects=True, stream=True)
	    with open(data_path / "validation" / "val.json", 'wb') as f:
	        total_length = int(r.headers.get('content-length'))
	        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
	            if chunk:
	                f.write(chunk)
	                f.flush()
	    print("Finished downloading validation set.\n")

	if not os.path.isfile(data_path / "test" / "test.json"):
	    print("Downloading test set...")
	    url = 'https://polybox.ethz.ch/index.php/s/0DtcjzDfcHL0yAP/download'
	    r = requests.get(url, allow_redirects=True, stream=True)
	    with open(data_path / "test" / "test.json", 'wb') as f:
	        total_length = int(r.headers.get('content-length'))
	        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
	            if chunk:
	                f.write(chunk)
	                f.flush()
	    print("Finished downloading test set.\n")