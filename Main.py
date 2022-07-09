from cProfile import label
import csv
import math
import matplotlib.pyplot as plt
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np

from Calculo_Tensorflow import Tensor

y = []
x = []
error = 0


class Neuronal(QMainWindow):
    ARRAY_W = np.array([[94,46,95,53,56,88]])
    E = list()
    EPOCAS = list()

    def __init__(self):
        super().__init__()
        uic.loadUi("interfaz.ui", self)
        self.btn_comenzar.clicked.connect(self.validar_datos)

    def validar_datos(self):
            tasa_aprendizaje = float(self.input_tasa.text())
            error_permisible = float(self.input_error.text())
            epocas = int(self.input_epocas.text())
            self.leer_CSV(epocas,tasa_aprendizaje,error_permisible)

    def leer_CSV(self, epocas,tasa_aprendizaje,error_permisible):
        matriz_X = []
        matriz_Y = []
        #auxMax = 0.000006

        with open('201219.csv') as f:
            reader = csv.reader(f)
            conta_aux = 0
            for j in reader:
                if conta_aux > 0:
                    matriz_X.append([1, int(j[0]),int (j[1]),int (j[2]),int (j[3]),int (j[4])])
                    matriz_Y.append([int (j[5])])
                conta_aux = 1
        matriz_X = np.array(matriz_X)
        matriz_Y = np.array(matriz_Y)

        print("-----------Matrices Iniciales-----------")
        print(matriz_X)
        print(matriz_Y)

        for i in range(epocas):
            u=self.calculo_u(matriz_X)
            y_calculada=self.fa_lineal(u)
            error=self.calcular_error(y_calculada,matriz_Y)
            delta_W = self.delta_w(matriz_X,error,tasa_aprendizaje)
            self.ARRAY_W = self.doble_w(delta_W)
            e = self.calcular_e(error)
            self.E.append(e)
            self.EPOCAS.append(i)

        self.graficar()
        tensor = Tensor
        tensor.calcular(epocas,error_permisible)

        metodo_5_veces = Comparacion5Tasas()
        metodo_5_veces.tasas_diferentes(epocas,matriz_X,matriz_Y,tasa_aprendizaje)

    def calculo_u(self, matriz_X):

        print("----------- Entro al método calculo de U -----------")
        matriz_WT = np.transpose(self.ARRAY_W)
        print(matriz_WT)
        u = np.dot(matriz_X,matriz_WT)
        print(u)

        return u

    def fa_lineal(self,u):
        y_calculada = u
        return y_calculada

    def calcular_error(self, y_calculada,matriz_Y):
        print("----------- Entro al metodo de Y caculada -----------")
        error = matriz_Y - y_calculada
        print(error)

        return error

    def delta_w(self,matriz_X,error,tasa_aprendizaje):
        print("----------- Entro al metodo de delta W -----------")
        e_transpuesta = np.transpose(error)
        con_matrizX = np.dot(e_transpuesta,matriz_X)
        delta_W = np.dot(tasa_aprendizaje,con_matrizX)
        print(delta_W)

        return delta_W

    def doble_w(self, delta_W):
        print("----------- Entro al metodo doble W -----------")
        nueva_w = delta_W + self.ARRAY_W
        print(nueva_w)
        return nueva_w

    def calcular_e(self, error):
        print("----------- Entro al metodo de calcular E -----------")
        e = 0
        for i in error:
            e = e + (i[0]**2)

        e = math.sqrt(e)
        print(e)

        return e

    def graficar(self):
        print("----------- Entro al metodo de graficar -----------")
        fig = plt.figure(figsize=(12,7))
        plt.plot(self.EPOCAS, self.E, color = "blue")
        plt.show()

class Comparacion5Tasas():
        def tasas_diferentes(self,epocas,matriz_X,matriz_Y,tasa_aprendizaje):
            e_dos = []
            epocas2 = []
            auxMin =  [0.000013, 0.000014, 0.000016, 0.000017, 0.000018]
            neuronal = Neuronal()
            vueltas = 5
            x = []
            for j in range(vueltas):
                tasa_aprendizaje = auxMin[j]
                e_dos.clear()
                neuronal.ARRAY_W = np.array([[94,46,95,53,56,88]])
                for i in range(epocas):
                    u=neuronal.calculo_u(matriz_X)
                    y_calculada=neuronal.fa_lineal(u)
                    error=neuronal.calcular_error(y_calculada,matriz_Y)
                    delta_W = neuronal.delta_w(matriz_X,error,tasa_aprendizaje)
                    neuronal.ARRAY_W = neuronal.doble_w(delta_W)
                    e = neuronal.calcular_e(error)
                    e_dos.append(e)
                    # if len(x) < 100:
                    #     x.append(i)
                epocas2.append(e_dos)
            for i in range(epocas):
                x.append(i)
            #print("Se cocatena: ",len(epocas2[0]))
            fig3 = plt.figure(figsize=(12,7))
            plt.plot(x,epocas2[0], color="red",label="1")
            plt.plot(x,epocas2[1], color="black",label="2")
            plt.plot(x,epocas2[2], color="green",label="3")
            plt.plot(x,epocas2[3], color="orange",label="4")
            plt.plot(x,epocas2[4], color="pink",label="5")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = Neuronal()
    GUI.show()
    sys.exit(app.exec_())

