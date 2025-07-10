import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model

from models.SegResNet.data_preparation import prepare_data_splits
from models.SegResNet.data_generator import BraTSDataGenerator
from models.SegResNet.model_architecture import build_segresnet
from models.SegResNet.loss_metrics import combined_loss, DiceCoefficient

def train_model(train_files, val_files, input_shape=(128, 128, 96, 4), num_classes=4):
    """Train the SegResNet model"""
    
    # Create data generators
    train_generator = BraTSDataGenerator(
        train_files, 
        batch_size=2, 
        dim=(128, 128, 96),  
        n_channels=4, 
        n_classes=4, 
        shuffle=True
    )
    
    val_generator = BraTSDataGenerator(
        val_files, 
        batch_size=1, 
        dim=(128, 128, 96),
        n_channels=4, 
        n_classes=4, 
        shuffle=False
    )
    
    # Build model
    model = build_segresnet(input_shape=input_shape, num_classes=num_classes)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=['accuracy',
            DiceCoefficient(class_id=1, name="dice_class1"),
            DiceCoefficient(class_id=2, name="dice_class2"),
            DiceCoefficient(class_id=3, name="dice_class3"), 
            tf.keras.metrics.MeanIoU(num_classes=4)
        ]
    )
    
    # Plot model architecture
    plot_model(
        model,
        to_file="model_summary.png",
        show_shapes=False,
        show_layer_names=True,
        rankdir="TB",
        dpi=300
    )
    
    # Define callbacks
    checkpoint_filepath = './best_model_mas_epocas_patience.keras'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=30,
        min_lr=1e-6
    )
    
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[model_checkpoint_callback, early_stopping, reduce_lr, csv_logger]
    )
    
    return model, history

if __name__ == "__main__":
    # Ejemplo de uso
    # train_files, val_files, test_files = prepare_data_splits("train_data.csv", data_list)
    # model, history = train_model(train_files, val_files)
    pass
