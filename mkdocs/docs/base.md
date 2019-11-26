### download\_dataset
```python
def download_dataset(save_path, dataset_name, file_name, dataset_url, cache_dir)
```

### load\_dataset
```python
def load_dataset(dataset_dir, processor)
```

### text\_information
```python
def text_information(data, single_text, language, char_level, tokenizer)
```

### label\_information
```python
def label_information(data)
```

## class Dataset_Base
### \_\_init\_\_
```python
def __init__(save_path)
```

### dataset\_information
```python
def dataset_information()
```

## class TSV_Processor
### get\_example\_from\_tensor\_dict
```python
def get_example_from_tensor_dict(tensor_dict)
```

### get\_train\_examples
```python
def get_train_examples(data_dir)
```

### get\_dev\_examples
```python
def get_dev_examples(data_dir)
```

### get\_test\_examples
```python
def get_test_examples(data_dir)
```

### get\_labels
```python
def get_labels()
```

## class CSV_Processor
### get\_example\_from\_tensor\_dict
```python
def get_example_from_tensor_dict(tensor_dict)
```

### get\_train\_examples
```python
def get_train_examples(data_dir)
```

### get\_dev\_examples
```python
def get_dev_examples(data_dir)
```

### get\_test\_examples
```python
def get_test_examples(data_dir)
```

### get\_labels
```python
def get_labels()
```

