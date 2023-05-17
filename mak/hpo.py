import tensorflow as tf
from keras.callbacks import CSVLogger

class CSVLoggerWithLr(CSVLogger):
    def __init__(self, filename, separator=',', append=False,server_round=1):
        self.server_round = server_round
        super().__init__(filename, separator=separator, append=append)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['round'] = self.server_round
        logs['lr'] = self.model.optimizer.lr.numpy() # Get the current learning rate from the optimizer
        super().on_epoch_end(epoch + 1, logs)



es = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                min_delta=0.01, 
                                patience=3, 
                                verbose=1, 
                                mode='auto', 
                                baseline=None, 
                                restore_best_weights=False)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                     factor=0.3, 
                                     patience=2, 
                                     verbose=0, 
                                     mode='auto',     
                                     min_delta=0.01, 
                                     cooldown=0, 
                                     min_lr=0.0001)
