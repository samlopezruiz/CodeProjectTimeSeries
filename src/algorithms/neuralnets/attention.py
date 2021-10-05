import tensorflow as tf
from tensorflow.keras.layers import Attention
from algorithms.tft2.libs.tft_model import ScaledDotProductAttention, get_decoder_mask

if __name__ == '__main__':
    x = tf.random.normal(shape=[1, 257, 5])
    mask = get_decoder_mask(x)

    scaled_dot_attn = ScaledDotProductAttention()
    attn = Attention(causal=True)

    context, self_att = scaled_dot_attn(x, x, x, mask=mask)
    print(f'Input batch, shape (batch): {x.shape}')
    print(f'Attention result shape: (batch_size, query_seq_length, units):           {context.shape}')
    print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {self_att.shape}')

    #%%
    context, self_att = attn([x, x, x], return_attention_scores=True)
    print(f'Input batch, shape (batch): {x.shape}')
    print(f'Attention result shape: (batch_size, query_seq_length, units):           {context.shape}')
    print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {self_att.shape}')