import Hydra.config as config
# import config

from transformers import RobertaTokenizerFast

def tokenize_dataframe(dataframe):
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(config.bert_version)

    # Filter out None rows
    out_df = dataframe.dropna()

    # Tokenize sentences
    out_df = out_df.copy()
    out_df['tokenized_sentence'] = out_df['sentence'].apply(
        lambda x: tokenizer(x, add_special_tokens=True))

    # Get token positions for merged target spans
    out_df = out_df.copy()
    out_df['target_token_span'] = out_df.apply(
        lambda x: (x.tokenized_sentence.char_to_token(x.target_merged[0]), 
                x.tokenized_sentence.char_to_token(x.target_merged[1] - 1)), axis=1)

    # Transform list of fe spans
    # Add new row for each span in fe_spans 
    out_df = out_df.explode('fe_spans')
    out_df = out_df[out_df.fe_spans.notna()]
    out_df = out_df[out_df.fe_spans.apply(lambda x: None not in x)]

    # Add 2 new columns for fe_span and fe_name
    out_df['fe_span'] = out_df.fe_spans.apply(lambda x: (x[0], x[1]))
    out_df['fe_name'] = out_df.fe_spans.apply(lambda x: (x[2]))

    # Get token positions for fe spans
    out_df = out_df.copy()
    out_df['fe_token_span'] = out_df.apply(
        lambda x: (x.tokenized_sentence.char_to_token(x.fe_span[0]),
                    x.tokenized_sentence.char_to_token(x.fe_span[1] - 1)), axis=1)
    
    # Get sentence number from index to allow for grouping
    out_df['sentence_number'] = out_df.index

    # Drop unnecessary columns
    out_df.drop(columns=['fe_spans'], inplace=True)

    # Fix non-unique index
    out_df.reset_index(drop=True, inplace=True)

    # Remove the extra info from the tokenized sentence for easier storage
    out_df.tokenized_sentence = out_df.tokenized_sentence.apply(lambda x: x.data)

    return out_df