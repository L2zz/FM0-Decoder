import tensorflow as tf



def loss(model, input, answer):
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(input), answer)))
  return reconstruction_error



def train(model, input, answer):
  optimizer = tf.optimizers.Adam(learning_rate=0.001)
  with tf.GradientTape() as tape:
    train_loss = loss(model, input, answer)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)
  return train_loss
