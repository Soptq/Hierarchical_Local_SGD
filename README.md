# HLSGD

Official Pytorch implementation of "HLSGD: Hierarchical Local SGD With Stale Gradients Featuring".

## Quickstart
### Cloning
```
git clone https://github.com/soptq/Hierarchical_Local_SGD
cd Hierarchical_Local_SGD
```

### Installation
```
pip install -r requirements.txt
```

### Dataset Preparation
```
python prepare_data.py
```

### Run DBS
Here we run HLSGD with DenseNet-121 in 4 workers' distributed environment.

Additionally, the total batchsize of the entire cluster is set to 128, other arguments remain default.

```
python hlsgd.py -d false -ws 4 -b 128 -m densenet
```

Details of other arguments can be referred in `parser.py`

## Citation
```
@inproceedings{Zhou2020HLSGD,
  title={HLSGD: Hierarchical Local SGD With Stale Gradients Featuring},
  author={Yuhao Zhou and Qing Ye and Hailun Zhang and Jiancheng Lv},
  year={2020}
}
```