### scaled\_dot\_product\_attention
```python
def scaled_dot_product_attention(q, k, v, mask)
```
计算注意力权重。 q, k, v 必须具有匹配的前置维度。 k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。 虽然 mask 根据其类型（填充或前瞻）有不同的形状， 但是 mask 必须能进行广播转换以便求和。

参数: q: 请求的形状 == (..., seq_len_q, depth) k: 主键的形状 == (..., seq_len_k, depth) v: 数值的形状 == (..., seq_len_v, depth_v) mask: Float 张量，其形状能转换成 (..., seq_len_q, seq_len_k)。默认为None。

返回值: 输出，注意力权重

### create\_padding\_mask
```python
def create_padding_mask(x)
```

### create\_look\_ahead\_mask
```python
def create_look_ahead_mask(x)
```

### point\_wise\_feed\_forward\_network
```python
def point_wise_feed_forward_network(d_model, dff)
```

## class PositionalEncoding
### \_\_init\_\_
```python
def __init__(vocab_size, d_model)
```

### get\_angles
```python
def get_angles(position, i, d_model)
```

### positional\_encoding
```python
def positional_encoding(position, d_model)
```

### call
```python
def call(inputs)
```

## class MultiHeadAttention
### \_\_init\_\_
```python
def __init__(d_model, num_heads)
```

### split\_heads
```python
def split_heads(x, batch_size)
```
分拆最后一个维度到 (num_heads, depth). 转置结果使得形状为 (batch_size, num_heads, seq_len, depth)



### call
```python
def call(v, k, q, mask)
```

## class EncoderLayer
### \_\_init\_\_
```python
def __init__(d_model, num_heads, dff, rate)
```

### call
```python
def call(x, training, mask)
```

## class DecoderLayer
### \_\_init\_\_
```python
def __init__(d_model, num_heads, dff, rate)
```

### call
```python
def call(x, enc_output, training, look_ahead_mask, padding_mask)
```

## class Encoder
### \_\_init\_\_
```python
def __init__(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate)
```

### call
```python
def call(inputs, training, mask)
```

## class Decoder
### \_\_init\_\_
```python
def __init__(num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate)
```

### call
```python
def call(x, enc_output, training, look_ahead_mask, padding_mask)
```

## class Transformer
### \_\_init\_\_
```python
def __init__(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate)
```

### call
```python
def call(inp, tar, training)
```

