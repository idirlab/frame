import xml.etree.ElementTree as ET
import Hydra.config as config
# import config

def process_fulltext_data(files):
    """ 
    Get fulltext data from FrameNet dataset.

    Args:
        files (list): List of file names.

    Returns:
        fulltext_data (list): List of dictionaries containing fulltext data.
    """

    # Get fulltext data from FrameNet dataset
    fulltext_data = []
    fn_prefix = '{http://framenet.icsi.berkeley.edu}'

    # Iterate over all fulltext xml files
    for x in files:
        # Check if file is xml file
        if x.endswith(".xml"):
            # Parse xml file
            fulltext_root = ET.parse(f'{config.framenet_path}/fulltext/{x}').getroot()
            sentences = fulltext_root.findall(f'{fn_prefix}sentence')

            # Get fulltext data
            for sentence in sentences:
                sent_text = sentence.find(f'{fn_prefix}text').text.strip()

                sent_annotations = sentence.findall(f".//{fn_prefix}annotationSet[@status='MANUAL']")

                for anno in sent_annotations:
                    # Get frame and frame element information
                    frame_name = anno.attrib.get('frameName', '')
                    lu_name = anno.attrib.get('luName', '')

                    anno_target_nodes = anno.findall(f".//{fn_prefix}label[@name='Target']")
                    anno_targets_disjoint = [(int(x.attrib['start']), int(x.attrib['end'])+1) for x in anno_target_nodes] if anno_target_nodes else None
                    anno_targets_merged = (min([x[0] for x in anno_targets_disjoint]), max([x[1] for x in anno_targets_disjoint])) if anno_targets_disjoint else None

                    fe_spans = []
                    
                    anno_fe_nodes = anno.find(f".//{fn_prefix}layer[@name='FE'][@rank='1']")

                    if anno_fe_nodes:
                        for fe in anno_fe_nodes:
                            start = int(fe.attrib['start']) if 'start' in fe.attrib else None
                            end = int(fe.attrib['end']) + 1 if 'end' in fe.attrib else None
                            name = fe.attrib.get('name', None)

                            fe_spans.append((start, end, name))

                    fulltext_data.append({
                        'frame_name': frame_name,
                        'lu_name': lu_name,
                        'sentence': sent_text,
                        'fe_spans': fe_spans,
                        'target_spans': anno_targets_disjoint,
                        'target_merged': anno_targets_merged
                        })
                    
    return fulltext_data

# Process all of the raw data
# train_fulltext_data = process_fulltext_data(config.OPENSESAME_TRAIN_FILES)
# dev_fulltext_data = process_fulltext_data(config.OPENSESAME_DEV_FILES)
# test_fulltext_data = process_fulltext_data(config.OPENSESAME_TEST_FILES)

# Save in json format
# pd.DataFrame(train_fulltext_data).to_json(f'{config.processed_path}/train_fulltext_data.json')
# pd.DataFrame(dev_fulltext_data).to_json(f'{config.processed_path}/dev_fulltext_data.json')
# pd.DataFrame(test_fulltext_data).to_json(f'{config.processed_path}/test_fulltext_data.json')