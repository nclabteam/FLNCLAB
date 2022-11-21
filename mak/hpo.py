import tensorflow as tf
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
