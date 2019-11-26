### convert\_tf\_checkpoint\_to\_pytorch
```python
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)
```
param tf_checkpoint_path: Path to the TensorFlow checkpoint path. :param bert_config_file: The config json file corresponding to the pre-trained BERT model. :param pytorch_dump_path: Path to the output PyTorch model. :return



### convert\_pytorch\_checkpoint\_to\_tf
```python
def convert_pytorch_checkpoint_to_tf(model, ckpt_dir, model_name)
```
param model:BertModel Pytorch model instance to be converted :param ckpt_dir: Tensorflow model directory :param model_name: model name :return: Currently supported HF models: Y BertModel N BertForMaskedLM N BertForPreTraining N BertForMultipleChoice N BertForNextSentencePrediction N BertForSequenceClassification N BertForQuestionAnswering



