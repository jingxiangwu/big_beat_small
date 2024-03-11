import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the right-hand side of the ODE
def my_func(x, y):
    return -2 * x * y

# Define the neural network model with He initialization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='tanh', input_shape=(1,),
                          kernel_initializer='he_normal'),  # He initialization
    tf.keras.layers.Dense(1, kernel_initializer='he_normal')  # He initialization for output layer as well
])

optimizer = tf.optimizers.Adam(learning_rate=0.01)

def train_step(x, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        
        # Compute dy/dx analytically
        with tf.GradientTape() as nested_tape:
            nested_tape.watch(x)
            y_pred_nested = model(x, training=True)
        dy_dx = nested_tape.gradient(y_pred_nested, x)
    
        # Compute the ODE loss component
        ode_loss = tf.reduce_mean(tf.square(dy_dx - my_func(x, y_pred)))
        
        # Compute the initial condition loss component
        y_at_zero_pred = model(tf.convert_to_tensor([[0.0]], dtype=tf.float32))
        initial_condition_loss = tf.reduce_mean(tf.square(y_at_zero_pred - 1))
        
        # Combine the losses
        loss = ode_loss + initial_condition_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Prepare the training data
x_train = np.linspace(-2, 2, 400).reshape(-1, 1).astype('float32')
x_train_tensor = tf.convert_to_tensor(x_train)

# Train the model
epochs = 1000
for epoch in range(epochs):
    loss = train_step(x_train_tensor, optimizer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Predictions for plotting
x_test = np.linspace(-2, 2, 400).reshape(-1, 1).astype('float32')
y_pred = model.predict(x_test)

# Analytical solution for comparison
y_true = np.exp(-x_test**2)

# Plotting the results and error
fig = plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_test, y_true, label='Analytical Solution', linewidth=4)
plt.plot(x_test, y_pred, label='NN Approximation', linestyle='None', marker = '*', markersize=3, color='red', alpha=0.7)
plt.legend()
plt.title('Solution Comparison')

plt.subplot(1, 2, 2)
plt.plot(x_test, np.abs(y_true - y_pred))
plt.title('Approximation Error')
plt.show()

fig.savefig('solving_ode.pdf')