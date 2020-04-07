# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'advices.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from data_import import json_specific_import
from advices_gui_funcs import data_gathering, compare

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

        self.student_path = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/data/darts_all_fake/me_5Char00'
        self.expert_path = r'C:/Users/quentin/Documents/Programmation/Python/ml_mla/data/darts_all_fake/me_15Char00'

        self.student = None
        self.expert = None

        self.features = None

        self.compared = None

        self.link()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Motion Learning Analytics"))
        self.pushbutton_import_student.setText(_translate("MainWindow", "Student"))
        self.pushbutton_expert_import.setText(_translate("MainWindow", "Expert"))
        self.pushbutton_compute_diff.setText(_translate("MainWindow", "Compare"))
        self.label_datatype.setText(_translate("MainWindow", "Data type"))
        self.pushbutton_add_combination.setText(_translate("MainWindow", "Add combination"))
        self.pushbutton_remove_combination.setText(_translate("MainWindow", "Remove combination"))
        self.label_joints.setText(_translate("MainWindow", "Joints"))
        self.pushbutton_import.setText(_translate("MainWindow", "Import"))
        self.label_student.setText(_translate("MainWindow", ""))
        self.label_expert.setText(_translate("MainWindow", ""))
        self.pushbutton_load_datatypes.setText(_translate("MainWindow", "Load data types"))

    def link(self):
        self.pushbutton_import_student.clicked.connect(self.select_folder_student)
        self.pushbutton_expert_import.clicked.connect(self.select_folder_expert)
        self.pushbutton_import.clicked.connect(self.import_data)
        self.pushbutton_compute_diff.clicked.connect(self.compare_values)
        self.pushbutton_add_combination.clicked.connect(self.add_combination)
        self.pushbutton_load_datatypes.clicked.connect(self.load_datatypes)
        self.pushbutton_remove_combination.clicked.connect(self.remove_combination)
        self.combobox_datatype.currentTextChanged.connect(self.re_set_possible_joints)

    def select_folder_student(self):
        self.student_path = QtWidgets.QFileDialog.getExistingDirectory()
        self.label_student.setText(self.student_path.split('/')[-1])

    def select_folder_expert(self):
        self.expert_path = QtWidgets.QFileDialog.getExistingDirectory()
        self.label_expert.setText(self.expert_path.split('/')[-1])

    def import_data(self):
        self.student = json_specific_import('/'.join(self.student_path.split('/')[0:-1]), self.student_path.split('/')[-1])[0]
        self.expert = json_specific_import('/'.join(self.expert_path.split('/')[0:-1]), self.expert_path.split('/')[-1])[0]

        self.set_values()

    def set_values(self):
        datatypes_names = set()

        for datatype in self.student.get_datatypes_names():
            datatypes_names.add(datatype)

        datatypes_names = sorted(list(datatypes_names))

        self.combobox_datatype.clear()
        for datatype in datatypes_names:
            self.combobox_datatype.addItem(datatype)

        self.listwidget_joints.clear()

        joint_list = sorted(self.student.get_datatype(self.combobox_datatype.currentText()).get_joint_list())
        for joint in joint_list:
            self.listwidget_joints.addItem(joint)

    def re_set_possible_joints(self):
        self.listwidget_joints.clear()
        joint_list = sorted(self.student.get_datatype(self.combobox_datatype.currentText()).get_joint_list())
        for joint in joint_list:
            self.listwidget_joints.addItem(joint)

    def compare_values(self):
        self.compared = compare(self.features)
        self.listwidget_diff.clear()
        self.listwidget_diff.addItem(f'Difference between {self.label_student.text()} and {self.label_expert.text()}:')
        self.listwidget_diff.addItem(f' ')
        for joint_and_datatype, value in self.compared.items():
            if len(joint_and_datatype) > 20:
                datatype = joint_and_datatype.split(' ')[1]
                joint_and_datatype = joint_and_datatype[:20] + '... ' + datatype
            self.listwidget_diff.addItem(f'{joint_and_datatype}: {value[0]:.5f}')

        self.listwidget_advices.addItem('Conseils sur le geste :')
        self.listwidget_advices.addItem('Votre coude ne doit pas bouger lors du lancer.')
        self.listwidget_advices.addItem('Votre bras doit rester aligné (de la main à l\'épaule) lorsque vous lancez.')


    def add_combination(self):
        if self.combobox_datatype.currentText() and len(self.listwidget_joints.selectedItems()) > 0:
            joints = [x.text() for x in self.listwidget_joints.selectedItems()]
            self.listwidget_combination.addItem(f'{self.combobox_datatype.currentText()}: {",".join([j for j in joints])}')

    def remove_combination(self):
        for to_delete in self.listwidget_combination.selectedItems():
            self.listwidget_combination.takeItem(self.listwidget_combination.row(to_delete))

    def load_datatypes(self):
        datatype_joints = {}
        combinations = [str(self.listwidget_combination.item(i).text()) for i in range(self.listwidget_combination.count())]
        for combi in combinations:
            splitted = combi.split(':')
            if splitted[0] in datatype_joints:
                datatype_joints[splitted[0]].extend(splitted[1].strip().split(','))
            else:
                datatype_joints[splitted[0]] = splitted[1].strip().split(',')


        self.features = data_gathering(self.student, self.expert, datatype_joints)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())