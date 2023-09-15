import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
from matplotlib import patches, animation,pyplot as plt

from game import *
from  PyQt5.QtGui import QMovie
from UI import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.filename = None
        self.filename_trail =None
        self.movie = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.openFileButton.clicked.connect(self.open_file)
        self.ui.openFileButton_2.clicked.connect(self.open_trail)
        self.ui.trainButton.clicked.connect(self.run)

    def open_file(self):
        self.filename, filetype = QFileDialog.getOpenFileNames(self, "Open file", "./", "*.txt")
        self.ui.selectedFileURLTextEdit.setText(str(self.filename))
        print("filename", self.filename)
        if len(self.filename) == 0:
            return
    def open_trail(self):
        self.filename_trail, filetype = QFileDialog.getOpenFileNames(self, "Open file", "./", "*.txt")
        self.ui.selectedFileURLTextEdit_2.setText(str(self.filename_trail))
        print("filename", self.filename_trail)
        if len(self.filename_trail) == 0:
            return


    def run(self):
        self.ui.label.setText("Animation will appear here once training is completed.")
        self.ui.trackLabel.setText("Track History will appear here once \ntraining is completed")
        k = self.ui.kSpinbox.value()
        epsilon = self.ui.epsilonDoubleSpinBox.value()
        lr = self.ui.lrDoubleSpinBox.value()
        epoch = self.ui.epochSpinbox.value()
        gamma = self.ui.gammaDoubleSpinBox.value()
        is_pseudo_inverse = self.ui.pseudoInverseCheckBox.isChecked()
        print("train start")
        weights, bias, rbfns_dict, loss_history, input_dim = RBFNTrain(self.filename, k, epsilon, lr,epoch, gamma,
                                                            pseudo_inverse=is_pseudo_inverse).train_rbfn()
        print(weights)
        print(loss_history)
        print(loss_history.shape)
        rbfn = RBFN(weights, bias, rbfns_dict)

        if input_dim == 5:
            is6d = True
        else:
            is6d = False

        car = Car(self.filename_trail)

        status = car.probe(is6d)

        position_history = []
        output_history = []
        while not (car.is_success or car.is_failed):
            # print every state and position of the car
            print("-------------------")
            print(status, car.x, car.y)
            print("car angle", car.angle)
            position_history.append([car.x,car.y,car.angle])
            # select action randomly
            # you can predict your action according to the state here
            predicted_control = rbfn.predict(status)

            print("predict:")
            print(predicted_control)
            print("-------------------")

            output_history.append([s for s in status] + [predicted_control])
            # take action
            car.move(predicted_control)
            status = car.probe(is6d)
        position_history = np.array(position_history)
        output_history = np.array(output_history)

        def ani_iter(position_history):
            ax.scatter(position_history[0], position_history[1], color='black', marker=".")
            circle = plt.Circle((position_history[0], position_history[1]), 3, fill=False)
            ax.add_artist(circle)
            x = [position_history[0], position_history[0] + 3 * np.cos(position_history[2] * np.pi / 180)]
            y = [position_history[1], position_history[1] + 3 * np.sin(position_history[2] * np.pi / 180)]
            plt.plot(x, y, color='green')

        def draw_boundary():
            x0, y0 = [-6, 6], [0, 0]
            x1, y1 = [-6, 6], [-3, -3]
            x2, y2 = [6, 6], [-3, 10]
            x3, y3 = [6, 30], [10, 10]
            x4, y4 = [30, 30], [10, 50]
            x5, y5 = [18, 30], [50, 50]
            x6, y6 = [18, 18], [22, 50]
            x7, y7 = [-6, 18], [22, 22]
            x8, y8 = [-6, -6], [-3, 22]
            plt.plot(x0, y0,color='black')
            plt.plot(x1, y1, x2, y2, color='black')
            plt.plot(x1, y1, x2, y2, color='black')
            plt.plot(x3, y3, x4, y4, color='black')
            plt.plot(x5, y5, x6, y6, color='black')
            plt.plot(x7, y7, x8, y8, color='black')
            rect = patches.Rectangle((18, 40), 12, 3, linewidth=2, edgecolor='r', facecolor='none', fill=True)
            ax.add_patch(rect)

        fig, ax = plt.subplots()
        ani = animation.FuncAnimation(fig, ani_iter, frames=position_history, interval=10, init_func=draw_boundary,repeat=False)

        ani.save("track.gif", fps=10)
        np.savetxt("track.txt",output_history)
        self.ui.trackLabel.setText(str(output_history))
        self.movie = QMovie("track.gif")
        self.ui.label.setMovie(self.movie)
        self.movie.start()











if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

