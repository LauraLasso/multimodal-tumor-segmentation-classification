import numpy as np
import tensorflow as tf

from models.SegResNet.data_preprocessing import process_scan, process_scany, center_crop_3d

class BraTSDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_list, batch_size=1, dim=(128, 128, 96), n_channels=4, 
                 n_classes=4, shuffle=True, mode='Train', for_inference=False):
        self.data_list = data_list
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mode = mode
        self.for_inference = for_inference
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.data_list) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find list of IDs
        data_temp = [self.data_list[i] for i in indexes]
        X, y = self.__data_generation(data_temp)
        
        return X, y
    
    def __data_generation(self, data_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))
        
        for i, data in enumerate(data_temp):
            t1 = process_scan(data['t1'], for_inference=self.for_inference)
            t1ce = process_scan(data['t1ce'], for_inference=self.for_inference)
            t2 = process_scan(data['t2'], for_inference=self.for_inference)
            flair = process_scan(data['flair'], for_inference=self.for_inference)
            
            seg = process_scany(data['seg'], for_inference=self.for_inference)
            
            t1 = center_crop_3d(t1, self.dim)
            t1ce = center_crop_3d(t1ce, self.dim)
            t2 = center_crop_3d(t2, self.dim)
            flair = center_crop_3d(flair, self.dim)
            seg = center_crop_3d(seg, self.dim)
            
            img = np.stack([t1, t1ce, t2, flair], axis=-1)
            
            # Convert BraTS labels (0, 1, 2, 4) to (0, 1, 2, 3)
            seg = np.where(seg == 4, 3, seg).astype(np.uint8)
            
            # Convert segmentation to one-hot encoding
            seg_one_hot = tf.one_hot(seg, depth=self.n_classes).numpy()
            X[i,] = img
            y[i,] = seg_one_hot
            
        return X, y
