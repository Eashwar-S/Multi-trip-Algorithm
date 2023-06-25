# Multi-trip-Algorithm


## Information on the dataset:
The dataset consists of undirected weighted multi-graphs stored in .pkl or .net formats. These undirected graphs form instances for the multi-trip multi-depot rural postman problem which is solved using the multi-trip algortihm. The dataset consists of instances obtained from the literature and also from real-world road networks.

## Dataset structure
```
dataset/
    dearmon graph files/   -- .net or .pkl graph files
    dearmon graph info/    -- .npy files containing depot and
                                requirededge information
    real world graph files/
        icy road weather instance graph/       --road network with considering wind conditions
        icy road instance graph/        --road network without considering wind conditions
    real world graph info/    -- .npy files containing depot and
                                requirededge information

```


## Software Required
1. [Python >=3.9]()

## Steps to run the repository
```bash
git clone https://github.com/Eashwar-S/Multi-trip-Algorithm.git
cd Multi-trip-Algorithm/src
```
## Install Required Python Packages
```bash
pip install -r requirements.txt
```

## To run multi-trip algorithm
```bash
python multi-trip-algorithm.py
```
