# Frame-semantic parsing pipeline

Robust Frame-Semantic Models with Lexical Unit Trees and Negative Samples

https://aclanthology.org/2024.acl-long.374/

--------

## Important files
- `setup.sh` prepares the repo by creating needed folders, extracting the FrameNet dataset, and tokenizing the dataset. 
    - Note: you must register for access to FrameNet at their website. Please do so before using this code. We include the relevant portions of the dataset for convenience, but it may be removed in the future. 
    - You may also need to add the repo's parent directory to your (hopefully virtual) environment's PATH variable. 

- New evaluation datasets can be found at `frame_identification/data/new_datasets.zip`

- `examples/` contains several examples for using the code in this repository

    - `candidate_target.ipynb` shows how to use the (very efficient) candidate generation algorithm to extract possible LUs from a sentence
    - `create_test_1cf.ipynb` and `create_test_uu.ipynb` show how we created the two new evaluation sets. 
    - `train_candidate_filter.ipynb` shows how to train the candidate target filtering model
    - `train_frame_identification.ipynb` shows how we use FIDO's training script. Our modifications of FIDO can be found primarily in the main.py file.

