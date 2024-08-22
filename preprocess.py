import config
import os
import pandas as pd

from scripts.process_frames import get_frame_info
from scripts.process_fulltext import process_fulltext_data
from scripts.tokenize_dataframe import tokenize_dataframe
from scripts.process_os_files import process_os_data

# ================== Frame Info ====================
# Get frame and frame element information from FrameNet dataset and save in json format
print(f'Processing FrameNet frame data...')
frame_info = get_frame_info(f'{config.framenet_path}/frame/')

pd.DataFrame(frame_info).to_json(f'{config.processed_path}/frame_info.json')

# ================= Fulltext Data =================
# Process fulltext files and save in json format
print(f'Processing fulltext data...')
train_fulltext_data = process_fulltext_data(config.OPENSESAME_TRAIN_FILES)
dev_fulltext_data = process_fulltext_data(config.OPENSESAME_DEV_FILES)
test_fulltext_data = process_fulltext_data(config.OPENSESAME_TEST_FILES)

print(f'Saving fulltext data...')
pd.DataFrame(train_fulltext_data).to_json(f'{config.processed_path}/train_fulltext_data.json')
pd.DataFrame(dev_fulltext_data).to_json(f'{config.processed_path}/dev_fulltext_data.json')
pd.DataFrame(test_fulltext_data).to_json(f'{config.processed_path}/test_fulltext_data.json')


# ================== Tokenize Data ====================
# Tokenize fulltext data and add to dataframe

# Tokenize train data
print(f'Tokenizing training data...')
train_fulltext_data = pd.read_json(f'{config.processed_path}/train_fulltext_data.json')
train_fulltext_data = tokenize_dataframe(train_fulltext_data)
train_fulltext_data.to_json(f'{config.processed_path}/train_fulltext_data_tokenized.json')

# Tokenize dev data
print(f'Tokenizing dev data...')
dev_fulltext_data = pd.read_json(f'{config.processed_path}/dev_fulltext_data.json')
dev_fulltext_data = tokenize_dataframe(dev_fulltext_data)
dev_fulltext_data.to_json(f'{config.processed_path}/dev_fulltext_data_tokenized.json')

# Tokenize test data
print(f'Tokenizing test data...')
test_fulltext_data = pd.read_json(f'{config.processed_path}/test_fulltext_data.json')
test_fulltext_data = tokenize_dataframe(test_fulltext_data)
test_fulltext_data.to_json(f'{config.processed_path}/test_fulltext_data_tokenized.json')

# ================== Candidate Targets ====================
# Get candidate targets from fulltext data and save in json format

# train_candidate_dataset = process_candidate_targets('train_fulltext_data_tokenized.json')
# dev_candidate_dataset = process_candidate_targets('dev_fulltext_data_tokenized.json')
# test_candidate_dataset = process_candidate_targets('test_fulltext_data_tokenized.json')

# train_candidate_dataset.to_json(f'{config.processed_path}/train_candidate_targets.json')
# dev_candidate_dataset.to_json(f'{config.processed_path}/dev_candidate_targets.json')
# test_candidate_dataset.to_json(f'{config.processed_path}/test_candidate_targets.json')

# ================== OS Data ====================
# Process OS data and save in json format

assert os.path.exists(f'{config.raw_path}/os_train.pkl'), 'open-sesame data not found'

train_os_data = process_os_data(f'{config.raw_path}/os_train.pkl')
dev_os_data = process_os_data(f'{config.raw_path}/os_dev.pkl')
test_os_data = process_os_data(f'{config.raw_path}/os_test.pkl')

train_os_data.to_json(f'{config.processed_path}/train_os_data.json')
dev_os_data.to_json(f'{config.processed_path}/dev_os_data.json')
test_os_data.to_json(f'{config.processed_path}/test_os_data.json')