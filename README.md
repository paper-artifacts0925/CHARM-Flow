# CHARM-FLOW

## Environment

```bash
conda create -n charm-flow python=3.11 -y
conda activate charm-flow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export CHARM_TRAIN_PYTHON=python
export CHARM_EVAL_PYTHON=python
```

## Data

```bash
./fetch_data.sh all
./prepare_data.sh all build
```

## Train

```bash
GPU_SET=0 ./train.sh replogle formal14000
GPU_SET=0,1 ./train.sh pbmc formal14000
GPU_SET=0,1,2,3 ./train.sh tahoe formal14000
```

## Test

```bash
GPU=0 ./test.sh replogle
GPU_IDS=0,1,2,3,4,5 ./test.sh pbmc
GPU=0 ./test.sh tahoe
```
