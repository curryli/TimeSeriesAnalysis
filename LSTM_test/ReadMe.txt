运行成功
https://yq.aliyun.com/articles/68463
http://blog.csdn.net/a819825294/article/details/54376781
https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(timesteps, data_dim),  #(50,1)
        output_dim=50, #第一层LSTM神经元个数，随便定义，不一定是50 ，可以是64 128...
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],  #100, 第一层LSTM神经元个数
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3])) #1 一个输出做预测
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model
	
	
	LSTM(128, input_shape=(timesteps, data_dim)