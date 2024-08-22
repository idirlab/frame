import pickle
from Hydra.utils.data import get_framenet_metadata
import Hydra.config as config
from transformers import AutoTokenizer


def process_os_data(os_file, tokenizer = AutoTokenizer.from_pretrained('roberta-base')):
    lu_manager, frame_info, frame_lu_defs = get_framenet_metadata()

    with open(os_file, 'rb') as f:
        data = pickle.load(f)

    data = data[data.frame != 'Time_vector']

    data = data.rename(columns={'text': 'sentence', 'frame': 'frame_name', 
                                'lu': 'lu_name', 'target':'target_span',
                                'fe': 'fe_spans'})

    data = data.merge(frame_info[['name', 'definition', 'lu_definitions', 'fes']], 
                        left_on='frame_name', right_on='name', how='left')

    data['lu_definition'] = data.apply(lambda x: x.lu_definitions[x.lu_name], axis=1)

    data.rename(columns={'fes': 'frame_fes'}, inplace=True)
    data['frame_fes'] = data.frame_fes.apply(lambda x: list(x.keys()))

    data = data[['sentence', 'frame_name', 'lu_name', 'target_span', 'fe_spans', 'definition', 'lu_definition', 'frame_fes']]

    sents = data.sentence.unique()
    toks = {sent:tokenizer(sent) for sent in sents}
    data['target_token_span'] = data.apply(lambda x: (toks[x.sentence].char_to_token(x.target_span[0]), 
                                                      toks[x.sentence].char_to_token(x.target_span[1] - 1)), axis=1)

    return data
