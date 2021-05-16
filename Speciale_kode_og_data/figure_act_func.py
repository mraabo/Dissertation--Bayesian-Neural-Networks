import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-10., 10., 200)
y_sigmoid = tf.keras.activations.sigmoid(x)
y_tanh = tf.keras.activations.tanh(x)
y_relu = tf.keras.activations.relu(x)
y_elu = tf.keras.activations.elu(x)
y_step = x > 0

plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_tanh, label=r'$tanh$')
plt.plot(x, y_relu, ':', label='ReLU', linewidth=3)
plt.plot(x, y_elu, label='ELU')
plt.plot(x, y_step, label='Step')
axes = plt.gca()
axes.set_ylim([-2, 2])
plt.xlabel('a')
plt.ylabel('g(a)')
plt.legend()
plt.savefig('act_func_fig.pdf')
plt.show()
