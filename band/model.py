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
from transformers.modeling_tf_utils import get_initializer


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
        config, other_config = config
        bert = TFBertModel.from_pretrained(pretrained_dir, config=config, from_pt=from_pt)
        inputs, outputs = cls(config=config, other_config=other_config).build_model(bert)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class TFBertForSequenceClassification(BertModel):

    def __init__(self, config, other_config):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return inputs, outputs


class TFBertForTokenClassification(BertModel):

    def __init__(self, config, other_config):
        super().__init__()
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        sequence_output = self.dropout(sequence_output)
        outputs = self.classifier(sequence_output)
        return inputs, outputs


class TFBertForQuestionAnswering(BertModel):

    def __init__(self, config, other_config):
        super().__init__()
        self.unique_id = tf.keras.layers.Input(shape=(), dtype='int32', name='unique_id')
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, name='qa_outputs')
        self.start_pos_classifier = tf.keras.layers.Dense(other_config['max_length'],
                                                          kernel_initializer=get_initializer(config.initializer_range),
                                                          name='start_position')
        self.end_pos_classifier = tf.keras.layers.Dense(other_config['max_length'],
                                                        kernel_initializer=get_initializer(config.initializer_range),
                                                        name='end_position')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        pooled_output = self.dropout(pooled_output)
        start_logits = self.start_pos_classifier(pooled_output)
        end_logits = self.end_pos_classifier(pooled_output)
        outputs = (start_logits, end_logits)
        inputs = [self.unique_id] + self.inputs
        return inputs, outputs


class TFBertForQuestionAnsweringWithAnswerType(BertModel):

    def __init__(self, config):
        super().__init__()
        self.unique_id = tf.keras.layers.Input(shape=(), dtype='int32', name='unique_id')
        self.num_labels = config.num_labels
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, name='qa_outputs')
        self.start_pos_classifier = tf.keras.layers.Dense(config.max_length,
                                                          kernel_initializer=get_initializer(config.initializer_range),
                                                          name='start_position')
        self.end_pos_classifier = tf.keras.layers.Dense(config.max_length,
                                                        kernel_initializer=get_initializer(config.initializer_range),
                                                        name='end_position')

        self.answer_type_classifier = tf.keras.layers.Dense(config.answer_types_num,
                                                            kernel_initializer=get_initializer(
                                                                config.initializer_range),
                                                            name='answer_type_classifier')

    def build_model(self, bert):
        inputs = self.inputs
        sequence_output, pooled_output = bert(inputs)
        pooled_output = self.dropout(pooled_output)
        start_logits = self.start_pos_classifier(pooled_output)
        end_logits = self.end_pos_classifier(pooled_output)
        answer_type_logits = self.answer_type_classifier(pooled_output)
        outputs = (start_logits, end_logits, answer_type_logits)
        inputs = [self.unique_id] + self.inputs
        return inputs, outputs


if __name__ == '__main__':
    pretrained_path = "/home/band/models"
    bert_config = BertConfig.from_pretrained(pretrained_path)
    model = TFBertForQuestionAnswering.from_pretrained(pretrained_path, bert_config, from_pt=True)
    print(model.summary())
