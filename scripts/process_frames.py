import os
import xml.etree.ElementTree as ET
import Hydra.config as config
# import config

config.processed_path = './data/processed/' # Add this to config.py

def get_frame_info(path):
    """ 
    Get frame and frame element information from FrameNet dataset.

    Args:
        path (str): Path to FrameNet dataset.

    Returns:
        frame_info (list): List of dictionaries containing frame and frame element information.
    """

    # Get frame and frame element information from FrameNet dataset
    frame_info = []
    fn_prefix = '{http://framenet.icsi.berkeley.edu}'

    # Iterate over all frame xml files
    for file_name in os.listdir(path):
        # Check if file is xml file
        if file_name.endswith(".xml") and file_name not in set(["Time_vector.xml"]):
            # Parse xml file
            with open(f'{config.framenet_path}/frame/{file_name}', 'r', encoding='utf8') as file:
                xml_string = file.read().replace('&gt;', '>').replace('&lt;', '<')
            
            frame_root = ET.fromstring(xml_string)

            # Get FE names
            # fes = {name: None for name in [x.attrib["name"] for x in frame_root.findall(f'{fn_prefix}FE')]}
            fes = {}
            for fe_node in frame_root.findall(f'{fn_prefix}FE'):
                fe_name = fe_node.attrib.get('name', None)
                fe_def_node = fe_node.find(f'{fn_prefix}definition')
                fe_def = "".join(fe_def_node.itertext())

                if fe_name is not None:
                    fes[fe_name] = fe_def

            # Get frame definition
            frame_def_node = frame_root.find(f'{fn_prefix}definition/{fn_prefix}def-root')
            frame_def = "".join(frame_def_node.itertext())

            # Get frame name
            frame_name = frame_root.attrib.get('name', None)

            # Get LU definitions
            lu_definitions = {}
            for lu_node in frame_root.findall(f'{fn_prefix}lexUnit'):
                lu_name = lu_node.attrib.get('name', None)
                # lu_name = '.'.join(lu_name.split('.')[:-1]) # remove POS tag
                lu_def_node = lu_node.find(f'{fn_prefix}definition')
                lu_def = "".join(lu_def_node.itertext())

                if lu_name is not None:
                    lu_definitions[lu_name] = lu_def

            # Append frame and frame element information to frame_info
            if frame_name is not None:
                frame_info.append({"name":frame_name, "fes":fes, "definition":frame_def, 
                                   "num_elements":len(fes), "lu_definitions":lu_definitions})

    # Sort frame_info by frame name
    return sorted(frame_info, key=lambda x: x["name"])


# Get frame and frame element information from FrameNet dataset
# frame_info = get_frame_info(f'{config.framenet_path}/frame/')

# Save as json
# pd.DataFrame(frame_info).to_json(f'{config.processed_path}/frame_info.json')

# Load json
# x = pd.read_json(f'{config.processed_path}/frame_info.json')