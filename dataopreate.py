#### 建模部分 ####
model = keras.Sequential()
model.add(layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(5))

model.compile(optimizer=keras.optimizers.Adam(), loss='mae',metrics=['accuracy'])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7, min_lr=0.000000005)

history = model.fit(x_train, y_train,
                    batch_size = 128,
                    epochs=70,
                    validation_data=(x_valid, y_valid),
                    callbacks=[learning_rate_reduction])

# loss变化趋势可视化
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.show()