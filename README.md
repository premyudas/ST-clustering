# ST-clustering
This repository contains the code used to benchmark the seven clustering algorithms (spatialMNN, BISON, SpaRTaCo, Novae, HyperGCN, STMSGAL, SCOIGET) and generate plots.

Download each dataset and place in `app/data/{folder}`. Your datasets and code should be organized as: \
app/ \
        |__data/ \
                |__Human dorsolateral prefrontal cortex (DLPFC)/ \
                |__HER2+ breast cancer/ \
                |__Stereo-seq MOSTA/ \
notebooks/ \
        |__benchmarking/

To run each Python notebook in `benchmarking`:
1. Create virtual environment
    
    `python3 -m venv .venv`

2. Download libraries from requirements.txt

    `pip install -r requirements.txt`

3. Activate virtual environment
    
    `source .venv/bin/activate`
