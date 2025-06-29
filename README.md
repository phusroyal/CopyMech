# Patch-Copying Repeated Tokens for Efficient Generation

## 1. Setup
1. Create conda environment:

```bash
conda create -n copy_mech python=3.10
```

2. Install requirements to conda:
```bash
conda install --file requirements.txt
```

3. [optional] Install Jupiter kernel if your notebook cannot find your env (1) and install ipykernel package (2):
```bash
conda install jupyter
conda install -n copy_mech ipykernel --update-deps --force-reinstall
```

## Experiments

1. Get `wiki_atomic_edits` data by running `utils/wiki_loader.py`

2. Running Schemas
- Copy mode: `copy_mode.py`
- Turning points: `turning_points.py`
- Continuation generation: `continous_generating.py`

