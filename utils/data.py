import pandas as pd
from typing import List
import Hydra.config as config
from Hydra.target_identification.candidate_identifier import LexicalUnitManager

def convert_lists_to_tuples(df: pd.DataFrame, columns: List[str]):
    """
    Convert lists in a dataframe column to tuples

    Args:
        df (pd.DataFrame): Dataframe to convert
        columns (List[str]): Columns to convert
    
    Returns:
        pd.DataFrame: Converted dataframe
    """

    for column in columns:
        df[column] = df[column].apply(lambda x: tuple(x))
        
    return df

def get_framenet_metadata():
    # Load candidate target identifier
    lu_manager = LexicalUnitManager()
    lu_manager.load_lus()

    # Load frame info
    frame_info = pd.read_json(f'{config.processed_path}/frame_info.json')

    # Get LU definitions
    frame_lu_defs = pd.DataFrame([(frame_name, lu_name, lu_def) for frame_name, lu_defs in frame_info[['name','lu_definitions']].values
                                                                    for lu_name, lu_def in lu_defs.items()],
                                columns=['frame_name', 'lu', 'lu_def'])
    frame_lu_defs['lu_no_pos'] = frame_lu_defs.lu.apply(lambda x: x.split('.')[0])

    return lu_manager, frame_info, frame_lu_defs

