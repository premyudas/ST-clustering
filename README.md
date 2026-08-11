# ST-clustering
This repository contains the code used to benchmark the seven clustering algorithms (spatialMNN, BISON, SpaRTaCo, Novae, HyperGCN, STMSGAL, SCOIGET) and generate plots.

To run each notebook in benchmarking:
1. Create virtual environment
    
    `python3 -m venv .venv`

2. Download libraries from requirements.txt

    `pip install -r requirements.txt`

3. Activate virtual environment
    
    `source .venv/bin/activate`

4. Download each dataset and place in `app/data/{folder}`
