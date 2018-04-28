import sys
import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from sklearn.metrics import confusion_matrix
import math


class InputForm(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Multi Layer Neural Networks - Task3'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500
        self.label1 = QLabel("Enter Number Of Hidden Layers :", self)
        self.label2 = QLabel("Enter Number Of Neurons in Each Hidden Layer :", self)
        self.label3 = QLabel("Enter learning Rate :", self)
        self.label4 = QLabel("Enter Number of Epochs :", self)
        self.label5 = QLabel("Choose The Activation Function Type :", self)
        self.label6 = QLabel("Choose The Stopping Criteria :", self)
        self.label7 = QLabel("( In Case Of MSE Selected ) Enter MSE Threshold :", self)
        self.CheckBox = QCheckBox("Bias", self)
        self.textboxHiddenLayers = QLineEdit(self)
        self.textboxNeuronsPerLayer = QLineEdit(self)
        self.textboxLr = QLineEdit(self)
        self.textboxEp = QLineEdit(self)
        self.textboxMSE = QLineEdit(self)
        self.button = QPushButton('Run', self)
        self.button.setToolTip('Run The Program')
        self.ActivationFunctionType = QComboBox(self)
        self.StoppingCriteria = QComboBox(self)
        # input variables
        self.bias = 0
        self.NumberOfLayers = 1
        self.NumberOfNeurons = 1
        self.learning_rate = 0
        self.no_epochs = 1
        self.ActivationFunction = "Sigmoid"
        self.StoppingCondition = "Fix The Number Of Epochs"
        self.MSEThreshold = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.move(10, 20)

        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.move(10, 60)

        self.label3.setAlignment(Qt.AlignCenter)
        self.label3.move(10, 100)

        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.move(10, 140)

        self.label5.setAlignment(Qt.AlignCenter)
        self.label5.move(10, 220)

        self.label6.setAlignment(Qt.AlignCenter)
        self.label6.move(10, 260)

        self.label7.setAlignment(Qt.AlignCenter)
        self.label7.move(30, 300)

        self.CheckBox.move(10, 180)

        self.textboxHiddenLayers.move(220, 20)
        self.textboxHiddenLayers.resize(40, 20)

        self.textboxNeuronsPerLayer.move(310, 60)
        self.textboxNeuronsPerLayer.resize(40, 20)

        self.textboxLr.move(180,100)
        self.textboxLr.resize(40, 20)

        self.textboxEp.move(180, 140)
        self.textboxEp.resize(40, 20)

        self.button.move(200, 400)
        self.button.clicked.connect(self.on_click)

        self.ActivationFunctionType.move(250, 220)
        self.ActivationFunctionType.addItem("Sigmoid")
        self.ActivationFunctionType.addItem("Hyperbolic")

        self.StoppingCriteria.move(250, 260)
        self.StoppingCriteria.addItem("Fix The Number Of Epochs")
        self.StoppingCriteria.addItem("MSE")

        self.textboxMSE.resize(40, 20)
        self.textboxMSE.move(350, 300)
        self.show()

    @pyqtSlot()
    def on_click(self):
        self.NumberOfLayers = self.textboxHiddenLayers.text()
        self.NumberOfNeurons = self.textboxNeuronsPerLayer.text()
        self.learning_rate = self.textboxLr.text()
        self.no_epochs = self.textboxEp.text()
        if self.CheckBox.isChecked():
            self.bias = 1
        else:
            self.bias = 0
        self.ActivationFunction = self.ActivationFunctionType.currentText()
        if self.StoppingCriteria.currentText() == "MSE":
            self.StoppingCondition = "MSE"
            self.MSEThreshold = self.textboxMSE.text()
        else:
            self.StoppingCondition = "Fix The Number Of Epochs"
        MyClass = MultiLayerNN(self.NumberOfLayers, self.NumberOfNeurons, self.bias, self.learning_rate,
                               self.no_epochs, self.ActivationFunction, self.StoppingCondition, self.MSEThreshold)
        self.close()


class MultiLayerNN:
    def __init__(self, no_layers, no_neu, b, lr, no_ep, af, sc, mse):
        #System Variables
        self.bias = int(b)
        self.NumberOfLayers = int(no_layers)
        x = no_neu.split(',')
        z = []
        for i in x:
            z.append(int(i))
        arr = np.array(z)
        if arr.shape[0] == 1:
            self.NumberOfNeurons = np.full(self.NumberOfLayers, arr[0])
            self.MaxNeuron = int(arr[0])
        elif arr.shape[0] == self.NumberOfLayers:
            self.NumberOfNeurons = np.zeros(self.NumberOfLayers,int)
            self.MaxNeuron = 0
            for i in range(0, self.NumberOfLayers):
                self.NumberOfNeurons[i] = arr[i]
                self.MaxNeuron = max(self.MaxNeuron, arr[i])
        else:
            return
        self.learning_rate = float(lr)
        self.ActivationFunction = af
        self.StoppingCondition = sc
        if sc != "MSE":
            self.no_epochs = int(no_ep)
        self.MSEThreshold = float(mse)
        self.NumberOfFeatures = 4
        self.NumberOfClasses = 3
        self.Weights = np.zeros((self.NumberOfLayers+1, max(self.MaxNeuron, 4), max(self.MaxNeuron, 4)))
        self.biasList = np.zeros((self.NumberOfLayers+1, 1))
        self.Out = np.zeros((self.NumberOfLayers+1, max(self.MaxNeuron, 4)))
        self.Error = np.zeros((self.NumberOfLayers+1, max(self.MaxNeuron, 4)))
        self.OutError1 = np.zeros((90, 1))
        self.OutError2 = np.zeros((90, 1))
        self.OutError3 = np.zeros((90, 1))
        self.Epochs = []
        # Assign Labels To Samples
        self.TrainingLabels = np.zeros((90, 1))
        self.TestingLabels = np.zeros((60, 1))
        self.TrainingLabels[0:30] = 1
        self.TrainingLabels[30:60] = 2
        self.TrainingLabels[60:90] = 3
        self.TestingLabels[0:20] = 1
        self.TestingLabels[20:40] = 2
        self.TestingLabels[40:60] = 3
        # Loading Data ......
        data = np.genfromtxt("Iris Data.txt", delimiter=',')
        self.TrainingData = np.concatenate((np.concatenate((data[1:31, 0:4], data[51:81, 0:4])), data[101:131, 0:4]))
        self.TestingData = np.concatenate((np.concatenate((data[31:51, 0:4], data[81:101, 0:4])), data[131:151, 0:4]))
        self.Normalize()
        self.initialize()
        self.train()
        self.Run()

    def Normalize(self):
        # Get Mean and Maximum
        mean1 = np.mean(self.TrainingData[:, 0])
        mean2 = np.mean(self.TrainingData[:, 1])
        mean3 = np.mean(self.TrainingData[:, 2])
        mean4 = np.mean(self.TrainingData[:, 3])
        max1 = np.max(self.TrainingData[:, 0])
        max2 = np.max(self.TrainingData[:, 1])
        max3 = np.max(self.TrainingData[:, 2])
        max4 = np.max(self.TrainingData[:, 3])

        # Subtract The Mean
        self.TrainingData[:, 0] -= mean1
        self.TrainingData[:, 1] -= mean2
        self.TrainingData[:, 2] -= mean3
        self.TrainingData[:, 3] -= mean4
        self.TestingData[:, 0] -= mean1
        self.TestingData[:, 1] -= mean2
        self.TestingData[:, 2] -= mean3
        self.TestingData[:, 3] -= mean4

        # Divide By Maximum
        self.TrainingData[:, 0] /= max1
        self.TrainingData[:, 1] /= max2
        self.TrainingData[:, 2] /= max3
        self.TrainingData[:, 3] /= max4
        self.TestingData[:, 0] /= max1
        self.TestingData[:, 1] /= max2
        self.TestingData[:, 2] /= max3
        self.TestingData[:, 3] /= max4

    def initialize(self):
        np.random.seed(0)
        for i in range(0, self.Weights.shape[0]):
            for j in range(0, self.Weights.shape[1]):
                self.Weights[i, j] = np.random.uniform(-1, 1)
        if self.bias == 1:
            for i in range(0, self.biasList.shape[0]):
                self.biasList[i] = np.random.uniform(-1, 1)
    
    def ActFunction(self, vnet):
        if self.ActivationFunction == "Sigmoid":
            return 1/(1+math.exp(vnet*-1))
        else:
            return (1 - math.exp(vnet * -1)) / (1 + math.exp(vnet * -1))
            #try:
              #  return (1 - math.exp(vnet * -1))/(1+math.exp(vnet*-1))
            #except OverflowError:
               # return float('inf')

    def train(self):
        Epoch_Number = 0
        OK = True
        while OK:
            for index in range(0, self.TrainingData.shape[0]):
                X = self.TrainingData[index]
                D = self.TrainingLabels[index]
                # Forward Step....
                for Level in range(0, self.NumberOfLayers+1):
                    if Level == 0:
                        Weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                            Vnet = np.sum(Weight[NeuronIndx, 0:4] * X) + self.biasList[Level] * self.bias
                            self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

                    elif Level == self.NumberOfLayers:
                        Weight = self.Weights[Level]
                        X = self.Out[Level-1, 0:self.NumberOfNeurons[Level-1]]

                        Vnet = np.sum(Weight[0, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                        Out1 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[1, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                        Out2 = self.ActFunction(Vnet)

                        Vnet = np.sum(Weight[2, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                        Out3 = self.ActFunction(Vnet)

                        if D == 1:
                            D1 = 1
                            D2 = 0
                            D3 = 0
                        elif D == 2:
                            D1 = 0
                            D2 = 1
                            D3 = 0
                        else:
                            D1 = 0
                            D2 = 0
                            D3 = 1
                        self.OutError1[index] = (D1 - Out1)
                        self.OutError2[index] = (D2 - Out2)
                        self.OutError3[index] = (D3 - Out3)
                        E1 = (D1 - Out1) * Out1 * (1-Out1)
                        E2 = (D2 - Out2) * Out2 * (1-Out2)
                        E3 = (D3 - Out3) * Out3 * (1-Out3)
                        self.Error[Level, 0] = E1
                        self.Error[Level, 1] = E2
                        self.Error[Level, 2] = E3
                    else:
                        Weight = self.Weights[Level]
                        X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level-1]]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                            Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                            self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)
                #Backward Step .......
                for Level in range(self.NumberOfLayers, -1, -1):
                    if Level == self.NumberOfLayers:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level-1]):
                            f = self.Out[Level-1, NeuronIndx]
                            self.Error[Level-1, NeuronIndx] = f * (1-f) * np.sum(self.Error[Level, 0:3]*weight[0:3, NeuronIndx])
                            Temp = weight[0:3, NeuronIndx] + self.learning_rate * self.Error[Level, 0:3] * f
                            self.Weights[Level, 0:3, NeuronIndx] = Temp[0:3]
                        self.biasList[Level] = self.biasList[Level] + np.sum(self.learning_rate * self.Error[Level, 0:3] * self.bias)
                    elif Level == 0:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, 4):
                            X = self.TrainingData[index]
                            NTemp = weight[0:self.NumberOfNeurons[Level], NeuronIndx] + self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * X[NeuronIndx]
                            self.Weights[Level, 0:self.NumberOfNeurons[Level], NeuronIndx] = NTemp[0:self.NumberOfNeurons[Level]]
                        self.biasList[Level] = self.biasList[Level] + np.sum(self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * self.bias)

                    else:
                        weight = self.Weights[Level]
                        for NeuronIndx in range(0, self.NumberOfNeurons[Level-1]):
                            f = self.Out[Level-1, NeuronIndx]
                            self.Error[Level-1, NeuronIndx] = f * (1-f) * np.sum(self.Error[Level, 0:self.NumberOfNeurons[Level]]*weight[0:self.NumberOfNeurons[Level], NeuronIndx])
                            Temp = weight[0:self.NumberOfNeurons[Level], NeuronIndx] + self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * f
                            self.Weights[Level, 0:self.NumberOfNeurons[Level], NeuronIndx] = Temp[0:self.NumberOfNeurons[Level]]
                        self.biasList[Level] = self.biasList[Level] + np.sum(self.learning_rate * self.Error[Level, 0:self.NumberOfNeurons[Level]] * self.bias)

            MSE1 = 0.5 * np.mean((self.OutError1 ** 2))
            MSE2 = 0.5 * np.mean((self.OutError2 ** 2))
            MSE3 = 0.5 * np.mean((self.OutError3 ** 2))
            TotalMSE = MSE1 + MSE2 + MSE3
            print(TotalMSE)
            self.Epochs.append(TotalMSE)
            if self.StoppingCondition == "MSE":
                if TotalMSE <= self.MSEThreshold:
                    break
            else:
                if Epoch_Number == self.no_epochs-1:
                    break

            Epoch_Number += 1

    def confusion(self, predicted, real):
        con = confusion_matrix(real, predicted)
        print(con)
        acc = 0
        for i in range(0, 3):
            acc += con[i, i]
        return (acc / len(real)) * 100

    def test(self, X):
        # Forward Step....
        for Level in range(0, self.NumberOfLayers + 1):
            if Level == 0:
                Weight = self.Weights[Level]
                for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                    Vnet = np.sum(Weight[NeuronIndx, 0:4] * X) + self.biasList[Level] * self.bias
                    self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

            elif Level == self.NumberOfLayers:
                Weight = self.Weights[Level]
                X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level-1]]

                Vnet = np.sum(Weight[0, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                Out1 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[1, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                Out2 = self.ActFunction(Vnet)

                Vnet = np.sum(Weight[2, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                Out3 = self.ActFunction(Vnet)

                if Out1 > Out2 and Out1 > Out3:
                    D1 = 1
                    D2 = 0
                    D3 = 0
                elif Out2 > Out3 and Out2 > Out1:
                    D1 = 0
                    D2 = 1
                    D3 = 0
                else:
                    D1 = 0
                    D2 = 0
                    D3 = 1
                if D1 == 1:
                    return 1
                elif D2 == 1:
                    return 2
                elif D3 == 1:
                    return 3
            else:
                Weight = self.Weights[Level]
                X = self.Out[Level - 1, 0:self.NumberOfNeurons[Level-1]]
                for NeuronIndx in range(0, self.NumberOfNeurons[Level]):
                    Vnet = np.sum(Weight[NeuronIndx, 0:self.NumberOfNeurons[Level-1]] * X) + self.biasList[Level] * self.bias
                    self.Out[Level, NeuronIndx] = self.ActFunction(Vnet)

    def Run(self):
        predicted = []
        for i in range(0, self.TestingData.shape[0]):
            predicted.append(self.test(self.TestingData[i]))
        pre = np.array(predicted)
        print(self.confusion(pre, self.TestingLabels))
        ep = np.array(self.Epochs)
        epoch = np.zeros((ep.shape[0], 1))
        for i in range(0, ep.shape[0]):
            epoch[i] = i + 1
        plt.plot(epoch, ep)
        plt.xlabel("epoch number ")
        plt.ylabel("MSE")
        plt.title("Learning Curve")
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InputForm()
    sys.exit(app.exec_())
