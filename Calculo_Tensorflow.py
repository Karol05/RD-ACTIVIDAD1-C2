import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Tensor:

    def calcular(epocas, tasaAprendizaje):
        print("----------- Iniciando metodo con Tensorflow -----------")
        X= np.loadtxt('201219.csv',unpack=True,usecols=[0,1,2,3,4], delimiter = ',', skiprows=1)
        Y = np.loadtxt('201219.csv',unpack=True,usecols=[5], delimiter = ',', skiprows=1)
        X = np.stack(X, axis = 1)
        print(X)
        print(Y)

        capa = tf.keras.layers.Dense(units=1,input_dim=5)
        modelo = tf.keras.Sequential([capa])

        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(tasaAprendizaje),
            loss='mean_squared_error'
        )

        historial = modelo.fit(X,Y,epochs=epocas,verbose=False)
        resultados = modelo.predict(X)
        fig2 = plt.figure(figsize=(12,7))
        plt.xlabel('# Epocas')
        plt.ylabel("loss")
        plt.plot(historial.history["loss"])
        plt.show()

        results_df = pd.DataFrame(data=resultados, columns=["Y calculada"])
        print(results_df)
        print("PESOS")
        print(capa.get_weights())      
