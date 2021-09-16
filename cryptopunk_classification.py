import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import multilabel_confusion_matrix

import json
import os
import numpy as np
import pandas as pd


import wandb
from wandb.keras import WandbCallback
wandb.login()

# Read in image and label arrays
X = np.load('x_punk.npy')
Y = np.load('y_punk.npy', allow_pickle=True)

# Split into train validation and test.
# No need to shuffle as Punks are randomly generated.
x_train = X[:8000]
y_train = Y[:8000].astype('float32')

x_val = X[8000:9000]
y_val = Y[8000:9000].astype('float32')

x_test = X[9000:]
y_test = Y[9000:].astype('float32')

# Form a baseline prediction for prediction punk type & accessory
baseline_prediction = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,0] for i in range(1000)]).astype('float32')

accuracy = 0
for i in range(1000):
  accuracy += 92-abs(y_test[i]-baseline_prediction[i]).sum()/92
baseline_acc = accuracy/1000

# Define train and test steps
def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_value

    
def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value

def train(train_dataset,
          val_dataset, 
          model,
          optimizer,
          loss_fn,
          train_acc_metric,
          val_acc_metric,
          epochs=10, 
          log_step=200, 
          val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # Log metrics using wandb.log
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})

# Read in sweep config
with open("config.json", "r") as jsonfile:
    sweep_config = json.load(jsonfile)

# Define model & sweep
def sweep_train():
    # Specify the hyperparameter to be tuned along with
    # an initial value
    config_defaults = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.1,
        'filters': 32,
        'metalayers':1,
        'logits':92
    }

    # Initialize wandb with a sample project name
    wandb.init(config=config_defaults)  # this gets over-written in the Sweep

    # Specify the other hyperparameters to the configuration, if any
    wandb.config.epochs = 14
    wandb.config.log_step = wandb.config.batch_size #int(20000/wandb.config.batch_size)
    wandb.config.val_log_step = wandb.config.batch_size #int(5000/wandb.config.batch_size)
    wandb.config.architecture_name = "CNN"
    wandb.config.dataset_name = "CryptoPunks"

    # build input pipeline using tf.data
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train[:,:wandb.config.logits]))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(wandb.config.batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val[:,:wandb.config.logits]))
    val_dataset = val_dataset.batch(wandb.config.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test[:,:wandb.config.logits]))
    test_dataset = test_dataset.batch(wandb.config.batch_size)

    # initialize model
    inputs = keras.Input((42, 42,3))

    x = layers.Conv2D(filters=wandb.config.filters, kernel_size=(3,3), activation="relu")(inputs)
    x = layers.Conv2D(filters=wandb.config.filters, kernel_size=(3,3), activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(wandb.config.dropout)(x)
    for i in range(wandb.config.metalayers):
        x = layers.Conv2D(filters=wandb.config.filters, kernel_size=(3,3), activation="relu")(x)
        x = layers.Conv2D(filters=wandb.config.filters, kernel_size=(3,3), activation="relu")(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(wandb.config.dropout)(x)
        

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(units=wandb.config.logits, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
    # Instantiate a loss function.
    loss_fn = keras.losses.BinaryCrossentropy()

    # Prepare the metrics.
    train_acc_metric = keras.metrics.AUC()
    val_acc_metric = keras.metrics.AUC()

    train(train_dataset,
          val_dataset, 
          model,
          optimizer,
          loss_fn,
          train_acc_metric,
          val_acc_metric,
          epochs=wandb.config.epochs, 
          log_step=wandb.config.log_step, 
          val_log_step=wandb.config.val_log_step)
    
    model.save('model')

project_name = "cryptopunk-classification"
sweep_id = wandb.sweep(sweep_config, project=project_name)
wandb.agent(sweep_id, function=sweep_train, count=10)


# Predict labels on unseen test data
model = keras.models.load_model('model', compile=False)
preds = model.predict(x_test)

# Turn predictions into binary True / False predictions for confusion matrix
preds = (preds > 0.5)
matrix = multilabel_confusion_matrix(y_test, preds)

# Read in label dataframe
df = pd.read_csv('punks_dummies.csv')

# Create results dataframe 
results = pd.DataFrame(columns = ['feature', 'acc', 'baseline', 'true_class_abs_error'])
for i in range(92):
  acc = (matrix[i][0][0] + matrix[i][1][1]) / 1000
  baseline = 1-y_test[:,i].sum() / 1000
  true_class_abs_error = abs(preds[:,i].sum()/y_test[:,i].sum()-1)
  metrics = {'feature':df.columns[i+1], 'acc':acc, 'baseline':baseline, 'true_class_abs_error':true_class_abs_error}
  results = results.append(metrics, ignore_index=True)
results.sort_values(by='true_class_abs_error', ascending=False).iloc[:20]

