# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'advices.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 770)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushbutton_import_student = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_import_student.setGeometry(QtCore.QRect(10, 30, 93, 28))
        self.pushbutton_import_student.setObjectName("pushbutton_import_student")
        self.pushbutton_expert_import = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_expert_import.setGeometry(QtCore.QRect(10, 100, 93, 28))
        self.pushbutton_expert_import.setObjectName("pushbutton_expert_import")
        self.pushbutton_compute_diff = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_compute_diff.setGeometry(QtCore.QRect(320, 360, 93, 28))
        self.pushbutton_compute_diff.setObjectName("pushbutton_compute_diff")
        self.listwidget_diff = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_diff.setGeometry(QtCore.QRect(10, 390, 781, 181))
        self.listwidget_diff.setObjectName("listwidget_diff")
        self.listwidget_advices = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_advices.setGeometry(QtCore.QRect(10, 580, 781, 131))
        self.listwidget_advices.setObjectName("listwidget_advices")
        self.combobox_datatype = QtWidgets.QComboBox(self.centralwidget)
        self.combobox_datatype.setGeometry(QtCore.QRect(310, 30, 211, 22))
        self.combobox_datatype.setObjectName("combobox_datatype")
        self.label_datatype = QtWidgets.QLabel(self.centralwidget)
        self.label_datatype.setGeometry(QtCore.QRect(310, 10, 61, 16))
        self.label_datatype.setObjectName("label_datatype")
        self.pushbutton_add_combination = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_add_combination.setGeometry(QtCore.QRect(310, 170, 111, 28))
        self.pushbutton_add_combination.setObjectName("pushbutton_add_combination")
        self.pushbutton_remove_combination = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_remove_combination.setGeometry(QtCore.QRect(310, 200, 131, 28))
        self.pushbutton_remove_combination.setObjectName("pushbutton_remove_combination")
        self.listwidget_joints = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_joints.setGeometry(QtCore.QRect(530, 30, 261, 191))
        self.listwidget_joints.setObjectName("listwidget_joints")
        self.label_joints = QtWidgets.QLabel(self.centralwidget)
        self.label_joints.setGeometry(QtCore.QRect(530, 10, 55, 16))
        self.label_joints.setObjectName("label_joints")
        self.pushbutton_import = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_import.setGeometry(QtCore.QRect(10, 170, 93, 28))
        self.pushbutton_import.setObjectName("pushbutton_import")
        self.listwidget_combination = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_combination.setGeometry(QtCore.QRect(310, 230, 481, 91))
        self.listwidget_combination.setObjectName("listwidget_combination")
        self.label_student = QtWidgets.QLabel(self.centralwidget)
        self.label_student.setGeometry(QtCore.QRect(110, 30, 131, 31))
        self.label_student.setText("")
        self.label_student.setObjectName("label_student")
        self.label_expert = QtWidgets.QLabel(self.centralwidget)
        self.label_expert.setGeometry(QtCore.QRect(110, 100, 131, 31))
        self.label_expert.setText("")
        self.label_expert.setObjectName("label_expert")
        self.pushbutton_load_datatypes = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_load_datatypes.setGeometry(QtCore.QRect(640, 330, 151, 28))
        self.pushbutton_load_datatypes.setObjectName("pushbutton_load_datatypes")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
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
        self.pushbutton_import_student.setText(_translate("MainWindow", "Student"))
        self.pushbutton_expert_import.setText(_translate("MainWindow", "Expert"))
        self.pushbutton_compute_diff.setText(_translate("MainWindow", "Compare"))
        self.label_datatype.setText(_translate("MainWindow", "Data type"))
        self.pushbutton_add_combination.setText(_translate("MainWindow", "Add combination"))
        self.pushbutton_remove_combination.setText(_translate("MainWindow", "Remove combination"))
        self.label_joints.setText(_translate("MainWindow", "Joints"))
        self.pushbutton_import.setText(_translate("MainWindow", "Import"))
        self.pushbutton_load_datatypes.setText(_translate("MainWindow", "Load data types"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())