import tensorflow as tf

inputs = tf.random.normal([3, 5, 16])

lstm = tf.keras.layers.SimpleRNN(4)
output = lstm(inputs)
print(output)

simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences=False, return_state=True)
whole_sequence_output, final_state = simple_rnn(inputs)
print(whole_sequence_output)
print(final_state)