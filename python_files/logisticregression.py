import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess your dataset
df = pd.read_csv('path_to_your_dataset.csv', usecols=["Open", "High", "Low", "Close"])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Function to split dataset for clients
def split_dataset_for_clients(dataset, num_clients):
    client_data = np.array_split(dataset, num_clients)
    return [tf.data.Dataset.from_tensor_slices((client.iloc[:, :-1].values, client.iloc[:, -1].values)).batch(20).repeat(10) for client in client_data]

# Split the dataset into subsets for each client
num_clients = 5
train_data = split_dataset_for_clients(df_scaled, num_clients)

# Define a Logistic Regression model
def create_logistic_regression_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=input_shape)
    ])
    return model

# Wrap the model for use with TFF
def model_fn():
    sample_batch = next(iter(train_data[0]))
    input_shape = sample_batch[0].shape[1:]
    model = create_logistic_regression_model(input_shape)
    return tff.learning.models.from_keras_model(
        model,
        input_spec=sample_batch.element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()]
    )

# Federated learning setup
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01))
state = trainer.initialize()

# Federated training loop
for _ in range(5):
    result = trainer.next(state, train_data)
    state = result.state
    metrics = result.metrics
    print(metrics['client_work']['train']['loss'])
