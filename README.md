# NN-Performance-Test

## Test Results

* Transformer: 5 Batch
* CNN: All Batch

### Laptop
| Model | CPU     | NPU    | MPS  |
| ----- | ------- | ------ | ---- | 
| BERT  | 112.34s | 30.60s |      | 
| CNN   | 30.41s  | 40.87s |      | 

### Desktop
CPU: 12700F
GPU: 3080

| Model | CPU    | 3080  |
| ----- | ------ | ----- |
| BERT  | 94.19s | 2.22s |
| CNN   | 12.68s | 1.67s |


Usage:

Modify the DEBUG_ARGS in main.py to set the parameters.
```python
DEBUG_ARGS = {
    'train': False,
    'test': True,
    'device': 'cpu',
    'model': 'transformer',
    'batch_size': 64,
    'epochs': 2,
    'max_train_batches': 5,
    'max_test_batches': 5,
}
```

If you want to run the model on CPU, you can use the following command:
```bash
python main.py --model transformer --device cpu --batch_size 64 --epochs 2 --max_train_batches 5 --max_test_batches 5 --debug=False
```