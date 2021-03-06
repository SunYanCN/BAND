# BAND：BERT Application aNd Deployment

A simple and efficient BERT model training and deployment framework.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/SunYanCN/BAND">
    <img src="figures/logo.png" alt="Logo" width="100" height="100">
  </a>
  <p align="center">
    BAND：BERT Application aNd Deployment
    <br />
    <a href="https://sunyancn.github.io/BAND/"><strong> Documents »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SunYanCN/BAND/tree/master/examples">Examples</a>
    ·
    <a href="https://github.com/SunYanCN/BAND/issues/new?assignees=&labels=&template=bug_report.md&title=">Report Bug</a>
    ·
    <a href="https://github.com/SunYanCN/BAND/issues/new?assignees=&labels=&template=feature_request.md&title=">Feature Request</a>
        ·
    <a href="https://github.com/SunYanCN/BAND/issues/new?assignees=&labels=&template=custom.md&title=">Questions</a>
  </p>

</p>

<h2 align="center">What is it</h3>  
  
**Encoding/Embedding** is a upstream task of encoding any inputs in the form of text, image, audio, video, transactional data to fixed length vector. Embeddings are quite popular in the field of NLP, there has been various Embeddings models being proposed in recent years by researchers, some of the famous one are bert, xlnet, word2vec etc. The goal of this repo is to build one stop solution for all embeddings techniques available, here we are starting with popular text embeddings for now and later on we aim  to add as much technique for image, audio, video inputs also.  
**Finally**, **`embedding-as-service`** help you to encode any given text to fixed length vector from supported embeddings and models.  
  
<h2 align="center">💾 Installation</h2>  
  
Install the band via `pip`.   
```bash  
$ pip install band -U
```  
> Note that the code MUST be running on **Python >= 3.6**. Again module does not support Python 2!  
  
<h2 align="center">⚡ ️Getting Started</h2> 

### Text Classification Example

```python
import time
import tensorflow as tf
from transformers import BertConfig, BertTokenizer
from band.model import TFBertForSequenceClassification
from band.dataset import ChnSentiCorp
from band.progress import classification_convert_examples_to_features

USE_XLA = False
USE_AMP = False

EPOCHS = 1
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5
SAVE_MODEL = False
pretrained_dir = "/home/band/models"

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

dataset = ChnSentiCorp(save_path="/tmp/band")
data, label = dataset.data, dataset.label
dataset.dataset_information()

train_number, eval_number, test_number = dataset.train_examples_num, dataset.eval_examples_num, dataset.test_examples_num

tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
train_dataset = classification_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            label_list=label,
                                                            output_mode="classification")
valid_dataset = classification_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            label_list=label,
                                                            output_mode="classification")
test_dataset = classification_convert_examples_to_features(data['test'], tokenizer, max_length=MAX_SEQ_LEN,
                                                           label_list=label,
                                                           output_mode="classification")

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

config = BertConfig.from_pretrained(pretrained_dir, num_labels=dataset.num_labels)
model = TFBertForSequenceClassification.from_pretrained(pretrained_dir, config=config, from_pt=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
if USE_AMP:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=train_number // BATCH_SIZE,
                    validation_data=valid_dataset,
                    validation_steps=eval_number // EVAL_BATCH_SIZE)

loss, accuracy = model.evaluate(test_dataset, steps=test_number // TEST_BATCH_SIZE)
print(loss, accuracy)

if SAVE_MODEL:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")
```
### Named Entity Recognition

```python
import time
import tensorflow as tf
from transformers import BertTokenizer, BertConfig
from band.dataset import MSRA_NER
from band.seqeval.callbacks import F1Metrics
from band.model import TFBertForTokenClassification
from band.utils import TrainConfig
from band.progress import NER_Dataset

pretrained_dir = '/home/band/models'

train_config = TrainConfig(epochs=3, train_batch_size=32, eval_batch_size=32, test_batch_size=1, max_length=128,
                           learning_rate=3e-5, save_model=False)

dataset = MSRA_NER(save_path="/tmp/band")

config = BertConfig.from_pretrained(pretrained_dir, num_labels=dataset.num_labels, return_unused_kwargs=True)
tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
model = TFBertForTokenClassification.from_pretrained(pretrained_dir, config=config, from_pt=True)

ner = NER_Dataset(dataset=dataset, tokenizer=tokenizer, train_config=train_config)
model.compile(optimizer=ner.optimizer, loss=ner.loss, metrics=[ner.metric])

f1 = F1Metrics(dataset.get_labels(), validation_data=ner.valid_dataset, steps=ner.valid_steps)

history = model.fit(ner.train_dataset, epochs=train_config.epochs, steps_per_epoch=ner.test_steps, callbacks=[f1])

loss, accuracy = model.evaluate(ner.test_dataset, steps=ner.test_steps)

if train_config.save_model:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")
```

### Question Answering
```python
import time
import tensorflow as tf
from transformers import BertConfig, BertTokenizer
from band.model import TFBertForQuestionAnswering
from band.dataset import Squad
from band.progress import squad_convert_examples_to_features, parallel_squad_convert_examples_to_features

USE_XLA = False
USE_AMP = False

EPOCHS = 1
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LEARNING_RATE = 3e-5
SAVE_MODEL = False
pretrained_dir = "/home/band/models"

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

dataset = Squad(save_path="/tmp/band")
data, label = dataset.data, dataset.label

train_number, eval_number = dataset.train_examples_num, dataset.eval_examples_num

tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
train_dataset = parallel_squad_convert_examples_to_features(data['train'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            doc_stride=128, is_training=True, max_query_length=64)
valid_dataset = parallel_squad_convert_examples_to_features(data['validation'], tokenizer, max_length=MAX_SEQ_LEN,
                                                            doc_stride=128, is_training=False, max_query_length=64)

train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

config = BertConfig.from_pretrained(pretrained_dir)
model = TFBertForQuestionAnswering.from_pretrained(pretrained_dir, config=config, from_pt=True, max_length=MAX_SEQ_LEN)

print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
if USE_AMP:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')

loss = {'start_position': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'end_position': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
metrics = {'start_position': tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
           'end_position': tf.keras.metrics.SparseCategoricalAccuracy('accuracy')}

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit(train_dataset, epochs=EPOCHS,
                    steps_per_epoch=train_number // BATCH_SIZE,
                    validation_data=valid_dataset,
                    validation_steps=eval_number // EVAL_BATCH_SIZE)

if SAVE_MODEL:
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    model.save(saved_model_path, save_format="tf")

```

## Dataset 
For more information about dataset, see

| Dataset Name | Language |             TASK              |        Description         |
| :----------: | :------: | :---------------------------: | :------------------------: |
| ChnSentiCorp |    CN    |      Text Classification      |   Binary Classification    |
|    LCQMC     |    CN    |     Question Answer Match     |   Binary Classification    |
|   MSRA_NER   |    CN    |   Named Entity Recognition    |     Sequence Labeling      |
|    Toxic     |    EN    |      Text Classification      |  Multi-label Multi-label   |
|   Thucnews   |    CN    |      Text Classification      | Multi-class Classification |
|    SQUAD     |    EN    | Machine Reading Comprehension |            Span            |
|     DRCD     |    CN    | Machine Reading Comprehension |            Span            |
|     CMRC     |    CN    | Machine Reading Comprehension |            Span            |
|     GLUE     |    EN    |                               |                            |

<h2 align="center" href="#supported-models">✅ Supported Embeddings and Models</h2> 

For more information about pretrained models, see
<!-- links -->
[your-project-path]: SunYanCN/BAND
[contributors-shield]: https://img.shields.io/github/contributors/SunYanCN/BAND.svg?style=flat-square
[contributors-url]: https://github.com/SunYanCN/BAND/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SunYanCN/BAND.svg?style=flat-square
[forks-url]: https://github.com/SunYanCN/BAND/network/members
[stars-shield]: https://img.shields.io/github/stars/SunYanCN/BAND.svg?style=flat-square
[stars-url]: https://github.com/SunYanCN/BAND/stargazers
[issues-shield]: https://img.shields.io/github/issues/SunYanCN/BAND.svg?style=flat-square
[issues-url]: https://github.com/SunYanCN/BAND/issues
[license-shield]: https://img.shields.io/github/license/SunYanCN/BAND.svg?style=flat-square
[license-url]: https://github.com/SunYanCN/BAND/blob/master/LICENSE

## Stargazers over time

[![Stargazers over time](https://starchart.cc/SunYanCN/BAND.svg)](https://starchart.cc/SunYanCN/BAND)