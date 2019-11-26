"""
@author: SunYanCN
@contact: sunyanhust@163.com
@blog: https://sunyancn.github.io
@version: 1.0
@license: MIT Licence
@file: covert.py, copy from https://github.com/huggingface/transformers/blob/master/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py
                 copy from https://github.com/huggingface/transformers/blob/master/transformers/convert_bert_pytorch_checkpoint_to_original_tf.py
@time: 2019-11-22 19:14:13
"""

import logging
import os

import numpy as np
import tensorflow as tf
import torch
from transformers import BertModel, BertConfig, BertForPreTraining, load_tf_weights_in_bert

logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    """
    :param tf_checkpoint_path: Path to the TensorFlow checkpoint path.
    :param bert_config_file: The config json file corresponding to the pre-trained BERT model.
    :param pytorch_dump_path: Path to the output PyTorch model.
    :return:
    """
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):
    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:
    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('token_type_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('LayerNorm/weight', 'LayerNorm/gamma'),
        ('LayerNorm/bias', 'LayerNorm/beta'),
        ('weight', 'kernel')
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return 'bert/{}'.format(name)

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


if __name__ == '__main__':
    convert_tf_checkpoint_to_pytorch(
        tf_checkpoint_path="C:/Users/lenovo/Downloads/chinese_L-12_H-768_A-12/bert_model.ckpt",
        bert_config_file="C:/Users/lenovo/Downloads/chinese_L-12_H-768_A-12/bert_config.json",
        pytorch_dump_path="C:/Users/lenovo/Downloads/chinese_L-12_H-768_A-12/pytorch_model.bin"
    )
