# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'base.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import ui_funcs
import constants as cst

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        ######################################################################################
        # AUTO-GENERATED STUFF
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_folder_path = QtWidgets.QLabel(self.centralwidget)
        self.label_folder_path.setGeometry(QtCore.QRect(120, 10, 211, 21))
        self.label_folder_path.setObjectName("label_folder_path")
        self.pushbutton_folder = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_folder.setGeometry(QtCore.QRect(10, 10, 93, 28))
        self.pushbutton_folder.setObjectName("pushbutton_folder")
        self.lineedit_person_to_add = QtWidgets.QLineEdit(self.centralwidget)
        self.lineedit_person_to_add.setGeometry(QtCore.QRect(10, 50, 151, 22))
        self.lineedit_person_to_add.setObjectName("lineedit_person_to_add")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 90, 101, 16))
        self.label.setObjectName("label")
        self.pushbutton_add_person = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_add_person.setGeometry(QtCore.QRect(170, 50, 93, 28))
        self.pushbutton_add_person.setObjectName("pushbutton_add_person")
        self.pushbutton_remove_selected = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_remove_selected.setGeometry(QtCore.QRect(152, 80, 111, 28))
        self.pushbutton_remove_selected.setObjectName("pushbutton_remove_selected")
        self.listwidget_people_to_process = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_people_to_process.setGeometry(QtCore.QRect(10, 110, 256, 192))
        self.listwidget_people_to_process.setObjectName("listwidget_people_to_process")
        self.combobox_datatype = QtWidgets.QComboBox(self.centralwidget)
        self.combobox_datatype.setGeometry(QtCore.QRect(360, 30, 151, 22))
        self.combobox_datatype.setObjectName("combobox_datatype")
        self.listwidget_joints = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_joints.setGeometry(QtCore.QRect(530, 30, 256, 91))
        self.listwidget_joints.setObjectName("listwidget_joints")
        self.label_datatype = QtWidgets.QLabel(self.centralwidget)
        self.label_datatype.setGeometry(QtCore.QRect(360, 10, 61, 16))
        self.label_datatype.setObjectName("label_datatype")
        self.label_joints = QtWidgets.QLabel(self.centralwidget)
        self.label_joints.setGeometry(QtCore.QRect(530, 10, 55, 16))
        self.label_joints.setObjectName("label_joints")
        self.combobox_clustering_algo = QtWidgets.QComboBox(self.centralwidget)
        self.combobox_clustering_algo.setGeometry(QtCore.QRect(10, 400, 251, 22))
        self.combobox_clustering_algo.setObjectName("combobox_clustering_algo")
        self.pushbutton_import_data = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_import_data.setGeometry(QtCore.QRect(10, 310, 151, 28))
        self.pushbutton_import_data.setObjectName("pushbutton_import_data")
        self.label_clustering_algo = QtWidgets.QLabel(self.centralwidget)
        self.label_clustering_algo.setGeometry(QtCore.QRect(10, 380, 121, 16))
        self.label_clustering_algo.setObjectName("label_clustering_algo")
        self.pushbutton_load_datatypes = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_load_datatypes.setGeometry(QtCore.QRect(360, 310, 151, 28))
        self.pushbutton_load_datatypes.setObjectName("pushbutton_load_datatypes")
        self.listwidget_combination = QtWidgets.QListWidget(self.centralwidget)
        self.listwidget_combination.setGeometry(QtCore.QRect(360, 130, 421, 171))
        self.listwidget_combination.setObjectName("listwidget_combination")
        self.pushbutton_add_combination = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_add_combination.setGeometry(QtCore.QRect(360, 70, 111, 28))
        self.pushbutton_add_combination.setObjectName("pushbutton_add_combination")
        self.pushbutton_remove_combination = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_remove_combination.setGeometry(QtCore.QRect(360, 100, 131, 28))
        self.pushbutton_remove_combination.setObjectName("pushbutton_remove_combination")
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
        ######################################################################################

        self.listwidget_people_to_process.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.link()

        self.label_folder_path.setText(r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/redo_all_students')
        self.original_data = []
        self.person_to_process = []

        self.features = []

        self.clustering_algo = ''

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_folder_path.setText(_translate("MainWindow", "Folder path"))
        self.pushbutton_folder.setText(_translate("MainWindow", "Folder"))
        self.label.setText(_translate("MainWindow", "People to process"))
        self.pushbutton_add_person.setText(_translate("MainWindow", "Add person"))
        self.pushbutton_remove_selected.setText(_translate("MainWindow", "Remove selected"))
        self.label_datatype.setText(_translate("MainWindow", "Data type"))
        self.label_joints.setText(_translate("MainWindow", "Joints"))
        self.pushbutton_import_data.setText(_translate("MainWindow", "Import data"))
        self.label_clustering_algo.setText(_translate("MainWindow", "Clustering algorithm"))
        self.pushbutton_load_datatypes.setText(_translate("MainWindow", "Load data types"))
        self.pushbutton_add_combination.setText(_translate("MainWindow", "Add combination"))
        self.pushbutton_remove_combination.setText(_translate("MainWindow", "Remove combination"))

    def link(self):
        self.pushbutton_folder.clicked.connect(self.select_folder)

        self.pushbutton_add_person.clicked.connect(self.add_person)
        self.pushbutton_remove_selected.clicked.connect(self.remove_person)

        self.pushbutton_import_data.clicked.connect(self.import_data)

        self.pushbutton_add_combination.clicked.connect(self.add_combination)

        self.pushbutton_load_datatypes.clicked.connect(self.load_datatypes)

        self.pushbutton_remove_combination.clicked.connect(self.remove_combination)

        for algo_name in cst.implemented_algo:
            self.combobox_clustering_algo.addItem(algo_name.capitalize())

    def select_folder(self):
        self.label_folder_path.setText(QtWidgets.QFileDialog.getExistingDirectory())

    def add_person(self):
        if self.lineedit_person_to_add.text():
            self.person_to_process.append(self.lineedit_person_to_add.text())
            self.listwidget_people_to_process.addItem(self.lineedit_person_to_add.text())
            self.lineedit_person_to_add.clear()

    def remove_person(self):
        for to_delete in self.listwidget_people_to_process.selectedItems():
            self.listwidget_people_to_process.takeItem(self.listwidget_people_to_process.row(to_delete))

    def import_data(self):
        names = [str(self.listwidget_people_to_process.item(i).text()) for i in range(self.listwidget_people_to_process.count())]

        if names:
            try:
                self.original_data = ui_funcs.import_data(self.label_folder_path.text(), names)
            except FileNotFoundError:
                print('ERROR: path does not contains the specified names')

        datatypes_names = set()
        for motion in self.original_data:
            for datatype in motion.get_datatypes_names():
                datatypes_names.add(datatype)

        for datatype in datatypes_names:
            self.combobox_datatype.addItem(datatype)

        for joint in self.original_data[0].get_joint_list():
            self.listwidget_joints.addItem(joint)

    def add_combination(self):
        if self.combobox_datatype.currentText() and self.listwidget_joints.count() > 0:
            joints = [str(self.listwidget_joints.item(i).text()) for i in range(self.listwidget_joints.count())]
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


        self.features = ui_funcs.data_gathering(self.original_data, datatype_joints)

    def run_clustering(self):
        pass

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())