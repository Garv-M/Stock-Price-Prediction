import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  # Using SVR (Support Vector Regression)

class FederatedSVMModel(tf.keras.Model):
    def __init__(self):
        super(FederatedSVMModel, self).__init__()
        # This is a placeholder; actual SVM logic will need to be TensorFlow compatible
        self.svm = SVR()

    def call(self, inputs):
        # Custom logic to integrate SVM goes here
        # Note: This will not work as expected in a federated context
        # SVM is not differentiable and doesn't fit well with TensorFlow's backpropagation
        # The following line is just a placeholder and needs appropriate modification
        return self.svm.fit(inputs, np.zeros(inputs.shape[0]))

# Load your dataset
df = pd.read_csv('/home/slade/AAPL.csv', usecols=["Open", "High", "Low", "Close"])

# Normalize the data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Function to split the dataset for clients
def split_dataset_for_clients(dataset, num_clients):
    client_data = np.array_split(dataset, num_clients)
    return [tf.data.Dataset.from_tensor_slices(client.values).batch(20).repeat(10) for client in client_data]

# Split the dataset into subsets for each client
num_clients = 5
train_data = split_dataset_for_clients(df_scaled, num_clients)

def model_fn():
    # Create an instance of your custom model
    model = FederatedSVMModel()

    # Dummy input_spec; this should ideally match the structure and data types of your actual dataset
    dummy_input_spec = tf.TensorSpec(shape=[None, 4], dtype=tf.float32)  # Example for 4 features

    return tff.learning.models.from_keras_model(
        model,
        input_spec=dummy_input_spec,
        loss=tf.keras.losses.MeanSquaredError(),  # Placeholder loss function
        metrics=[tf.keras.metrics.MeanAbsoluteError()]  # Placeholder metric
    )

# Federated learning setup
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))  # Optimizer may not be relevant for SVM
state = trainer.initialize()

# Federated training loop
for _ in range(5):
    result = trainer.next(state, train_data)
    state = result.state
    metrics = result.metrics
    print(metrics['client_work']['train']['loss'])
