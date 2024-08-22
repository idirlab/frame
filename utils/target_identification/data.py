from torch.utils.data import Dataset
from Hydra.utils.data import convert_lists_to_tuples
from Hydra.utils.data import get_framenet_metadata
import Hydra.config as config
import pandas as pd
import torch

def get_candidates(dataset, lu_manager, tokenizer):
    # Get all sentences
    unique_sents = dataset.sentence.unique()
    sent_toks = {sent: tokenizer(sent) for sent in unique_sents}

    # Get all candidates
    candidate_dataset = pd.DataFrame([(sent,span,cand_frames) for sent in unique_sents 
                for span, cand_frames in lu_manager.lookup_lus(sent).items()],
                columns=['sentence', 'target_merged', 'candidate_frames'])

    # Explode candidates into individual rows
    candidate_dataset = candidate_dataset.explode('candidate_frames')
    candidate_dataset = candidate_dataset.reset_index(drop=True)
    candidate_dataset = candidate_dataset.rename(columns={'candidate_frames': 'frame_name'})

    # Merge with dataset
    left_join = candidate_dataset.merge(dataset, how='left', on=['sentence', 'target_merged', 'frame_name'], indicator=True)
    right_join = candidate_dataset.merge(dataset, how='right', on=['sentence', 'target_merged', 'frame_name'], indicator=True)

    # Merge both joins and determine labels
    new_dataset = pd.concat([left_join, right_join])
    new_dataset = new_dataset.drop_duplicates().reset_index(drop=True)
    new_dataset['label'] = new_dataset._merge.apply(lambda x: 0 if x == 'left_only' else 1)

    # Get token spans
    new_dataset['target_token_span'] = new_dataset.apply(lambda x: (sent_toks[x.sentence].char_to_token(x.target_merged[0]), 
                                                        sent_toks[x.sentence].char_to_token(x.target_merged[1]-1)), axis=1)
    new_dataset['target_token_span'] = new_dataset.target_token_span.apply(lambda x: (x[0], x[1]) if x[1] is not None else (x[0], x[0]))
    
    # Drop unnecessary columns
    final_cols = ['sentence', 'target_token_span', 'frame_name', 'label']
    new_dataset = new_dataset[final_cols].drop_duplicates().dropna().reset_index(drop=True)
    
    return new_dataset, sent_toks

def load_dataset(file_path, tokenizer):
    dataset = pd.read_json(f'{config.processed_path}/{file_path}')
    cols_to_convert = ['target_merged', 'target_token_span', 'fe_span', 'fe_token_span']
    dataset = convert_lists_to_tuples(dataset, cols_to_convert)
    relevant_columns = ['frame_name', 'sentence', 
                        'target_merged', 'lu_name', 
                        'target_token_span']

    dataset = dataset[relevant_columns].drop_duplicates().reset_index(drop=True)

    lu_manager, frame_info, frame_lu_defs = get_framenet_metadata()

    dataset, sent_toks = get_candidates(dataset, lu_manager, tokenizer)

    # Get max label for each target_token_span in each sentence (for target identification)
    dataset = dataset.groupby(['sentence', 'target_token_span']).label.max().reset_index()
    
    dataset['input_ids'] = dataset.sentence.apply(lambda x: sent_toks[x]['input_ids'])
    dataset['attention_mask'] = dataset.sentence.apply(lambda x: sent_toks[x]['attention_mask'])

    pad_length = max([len(x['input_ids']) for x in sent_toks.values()])
    dataset['input_ids'] = dataset.input_ids.apply(lambda x: x + [tokenizer.pad_token_id] * (pad_length - len(x)))
    dataset['attention_mask'] = dataset.attention_mask.apply(lambda x: x + [tokenizer.pad_token_id] * (pad_length - len(x)))

    return dataset

class TargetIdentificationDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        self.input_ids = torch.Tensor(dataset.input_ids.values.tolist()).long()
        self.attention_mask = torch.Tensor(dataset.attention_mask.values.tolist()).long()
        self.labels = torch.Tensor(dataset.label.values.tolist()).long()
        self.target_token_span = torch.Tensor(dataset.target_token_span.values.tolist()).long()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx],
                'target_token_span': self.target_token_span[idx]}