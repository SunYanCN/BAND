### glue\_convert\_examples\_to\_features
```python
def glue_convert_examples_to_features(examples, tokenizer, max_length, label_list, output_mode, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero)
```
Loads a data file into a list of ``InputFeatures``


##### Args
* **examples**: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.

* **tokenizer**: Instance of a tokenizer that will tokenize the examples

* **max_length**: Maximum example length

* **task**: GLUE task

* **label_list**: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method

* **output_mode**: String indicating the output mode. Either ``regression`` or ``classification``

* **pad_on_left**: If set to ``True``, the examples will be padded on the left rather than on the right (default)

* **pad_token**: Padding token

* **pad_token_segment_id**: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)

* **mask_padding_with_zero**: If set to ``True``, the attention mask will be filled by ``1`` for actual values
    and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
    actual values)

##### Returns

### ner\_convert\_examples\_to\_features
```python
def ner_convert_examples_to_features(examples, tokenizer, label_list, max_length, cls_token_at_end, cls_token, cls_token_segment_id, sep_token, sep_token_extra, pad_on_left, pad_token, pad_token_segment_id, pad_token_label_id, sequence_a_segment_id, mask_padding_with_zero)
```
param examples: :param tokenizer: :param label_list: :param max_length: :param cls_token_at_end: :param cls_token: :param cls_token_segment_id: :param sep_token: :param sep_token_extra: :param pad_on_left: :param pad_token: :param pad_token_segment_id: :param pad_token_label_id: :param sequence_a_segment_id: :param mask_padding_with_zero: :return



