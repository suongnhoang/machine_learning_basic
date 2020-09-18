import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM

inputs = tf.random.normal([3, 5, 16])

lstm = SimpleRNN(4)
output = lstm(inputs)
print(output)
print('='*50)

simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences=True, return_state=True)
whole_sequence_output, final_state = simple_rnn(inputs)
print(whole_sequence_output)
print(final_state)
print('='*50)