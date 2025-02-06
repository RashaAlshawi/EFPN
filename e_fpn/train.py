import os
import tensorflow as tf
from model import build_fpn  
from dataset import load_data, prepare_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters
input_shape = (128, 128, 3)  # Adjust this to match your image size and channels
num_classes = 9
batch_size = 5
num_epochs = 10
learning_rate = 0.0009

def train_model():
    # Load and prepare dataset
    X_datas, y_datas = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(X_datas, y_datas)

    # Create data generators for training, validation, and testing
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(1000)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Build the model
    model = build_fpn(input_shape=input_shape, num_classes=num_classes)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # Model directory for saving checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callback for saving the best model during training
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Callback for early stopping (optional)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Training the model
    history = model.fit(
        train_data,
        epochs=num_epochs,
        validation_data=val_data,
        callbacks=[checkpoint_callback, early_stopping_callback],
        verbose=1
    )

 

    # Save the final model (optional)
    model.save('/Users/rashaalshawi/Documents/Research_PhD23/E-FPN-Segmentation/experiments/final_model.h5')


if __name__ == "__main__":
    train_model()
