"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: model.py
@time: 2019-12-02 08:07:30
"""

import tensorflow as tf
from transformers import TFBertModel, BertConfig


class BertModel(object):

    def __init__(self):
        input_ids = tf.keras.layers.Input(shape=(None,), dtype='int32', name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(None,), dtype='int32', name='attention_mask')
        token_type_ids = tf.keras.layers.Input(shape=(None,), dtype='int32', name='token_type_ids')
        inputs = [input_ids, attention_mask, token_type_ids]
        self.inputs = inputs

    def build_model(self, bert):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_dir, config, from_pt):
        bert = TFBertModel.from_pretrained(pretrained_dir, config=config, from_pt=from_pt)
        inputs, outputs = cls(config=config).build_model(bert)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class TFBertForSequenceClassification(BertModel):

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, name='classifier')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return inputs, outputs


class TFBertForTokenClassification(BertModel):

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, name='classifier')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        sequence_output = self.dropout(sequence_output)
        outputs = self.classifier(sequence_output)
        return inputs, outputs


class TFBertForQuestionAnswering(BertModel):

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, name='classifier')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        outputs = (start_logits, end_logits)
        return inputs, outputs


if __name__ == '__main__':
    pretrained_path = "/home/band/models"
    bert_config = BertConfig.from_pretrained(pretrained_path)
    model = TFBertForQuestionAnswering.from_pretrained(pretrained_path, bert_config, from_pt=True)
    print(model.summary())
