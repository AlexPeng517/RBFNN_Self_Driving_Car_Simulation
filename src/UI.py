# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1142, 711)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(480, 40, 621, 591))
        font = QtGui.QFont()
        font.setFamily("Garamond")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.QFileDialog = QtWidgets.QWidget(self.centralwidget)
        self.QFileDialog.setGeometry(QtCore.QRect(20, 280, 421, 91))
        self.QFileDialog.setObjectName("QFileDialog")
        self.openFileButton = QtWidgets.QPushButton(self.QFileDialog)
        self.openFileButton.setGeometry(QtCore.QRect(10, 10, 141, 28))
        self.openFileButton.setObjectName("openFileButton")
        self.selectedFileURLTextEdit = QtWidgets.QTextEdit(self.QFileDialog)
        self.selectedFileURLTextEdit.setGeometry(QtCore.QRect(170, 10, 221, 61))
        self.selectedFileURLTextEdit.setObjectName("selectedFileURLTextEdit")
        self.trainButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainButton.setGeometry(QtCore.QRect(30, 620, 93, 28))
        self.trainButton.setObjectName("trainButton")
        self.lrDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.lrDoubleSpinBox.setGeometry(QtCore.QRect(120, 570, 62, 22))
        self.lrDoubleSpinBox.setObjectName("lrDoubleSpinBox")
        self.kSpinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.kSpinbox.setGeometry(QtCore.QRect(90, 480, 42, 22))
        self.kSpinbox.setObjectName("kSpinbox")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 480, 31, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 570, 91, 16))
        self.label_3.setObjectName("label_3")
        self.epsilonDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.epsilonDoubleSpinBox.setGeometry(QtCore.QRect(240, 530, 131, 22))
        self.epsilonDoubleSpinBox.setDecimals(10)
        self.epsilonDoubleSpinBox.setSingleStep(1e-06)
        self.epsilonDoubleSpinBox.setProperty("value", 0.0)
        self.epsilonDoubleSpinBox.setObjectName("epsilonDoubleSpinBox")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(180, 530, 58, 15))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 520, 41, 16))
        self.label_5.setObjectName("label_5")
        self.epochSpinbox = QtWidgets.QSpinBox(self.centralwidget)
        self.epochSpinbox.setGeometry(QtCore.QRect(90, 520, 42, 22))
        self.epochSpinbox.setObjectName("epochSpinbox")
        self.gammaDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.gammaDoubleSpinBox.setGeometry(QtCore.QRect(230, 480, 141, 22))
        self.gammaDoubleSpinBox.setDecimals(14)
        self.gammaDoubleSpinBox.setSingleStep(1e-06)
        self.gammaDoubleSpinBox.setObjectName("gammaDoubleSpinBox")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(180, 480, 71, 21))
        self.label_6.setObjectName("label_6")
        self.pseudoInverseCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.pseudoInverseCheckBox.setGeometry(QtCore.QRect(160, 610, 121, 51))
        self.pseudoInverseCheckBox.setObjectName("pseudoInverseCheckBox")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(20, 20, 421, 251))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 419, 249))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")
        self.trackLabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setFamily("Garamond")
        font.setPointSize(12)
        self.trackLabel.setFont(font)
        self.trackLabel.setObjectName("trackLabel")
        self.verticalLayout.addWidget(self.trackLabel)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.QFileDialog_2 = QtWidgets.QWidget(self.centralwidget)
        self.QFileDialog_2.setGeometry(QtCore.QRect(20, 380, 421, 91))
        self.QFileDialog_2.setObjectName("QFileDialog_2")
        self.openFileButton_2 = QtWidgets.QPushButton(self.QFileDialog_2)
        self.openFileButton_2.setGeometry(QtCore.QRect(10, 10, 141, 28))
        self.openFileButton_2.setObjectName("openFileButton_2")
        self.selectedFileURLTextEdit_2 = QtWidgets.QTextEdit(self.QFileDialog_2)
        self.selectedFileURLTextEdit_2.setGeometry(QtCore.QRect(170, 10, 221, 61))
        self.selectedFileURLTextEdit_2.setObjectName("selectedFileURLTextEdit_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1142, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Animation will appear here once training is completed."))
        self.openFileButton.setText(_translate("MainWindow", "Open Train File"))
        self.trainButton.setText(_translate("MainWindow", "Train"))
        self.label_2.setText(_translate("MainWindow", "K"))
        self.label_3.setText(_translate("MainWindow", "Learning rate"))
        self.label_4.setText(_translate("MainWindow", "Epsilon"))
        self.label_5.setText(_translate("MainWindow", "Epoch"))
        self.label_6.setText(_translate("MainWindow", "Gamma"))
        self.pseudoInverseCheckBox.setText(_translate("MainWindow", "Pseudo Inverse"))
        self.trackLabel.setText(_translate("MainWindow", "Track History will appear here once \n"
"training is completed"))
        self.openFileButton_2.setText(_translate("MainWindow", "Open Trail File"))

