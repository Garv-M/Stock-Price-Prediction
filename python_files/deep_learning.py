import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
file_path = '/home/slade/AAPL.csv'
df = pd.read_csv(file_path)

# Normalize the data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Function to split the dataset for clients
def split_dataset_for_clients(dataset, num_clients):
    client_data = np.array_split(dataset, num_clients)
    return [tf.data.Dataset.from_tensor_slices((client.iloc[:, :-1].values, client.iloc[:, -1].values)).batch(20).repeat(10) for client in client_data]

# Split the dataset into subsets for each client
num_clients = 3
train_data = split_dataset_for_clients(df_scaled, num_clients)

# Wrap a Keras model for use with TFF
def model_fn():
    # Adjust the input shape to match the number of features in your dataset (excluding the target column)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.softmax, input_shape=(df_scaled.shape[1] - 1,)),
        tf.keras.layers.Dense(1)  # Predicting a single value (e.g., Close price)
    ])
    return tff.learning.models.from_keras_model(
        model,
        input_spec=train_data[0].element_spec,  # Set the input_spec based on the train_data format
        loss=tf.keras.losses.MeanSquaredError(),  # Suitable for regression tasks
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

# The federated learning setup
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01))  # Adjust learning rate as needed
state = trainer.initialize()

# Train the model with the simulated client data
for _ in range(5):
    result = trainer.next(state, train_data)
    state = result.state
    metrics = result.metrics
    print(metrics['client_work']['train']['loss'])
