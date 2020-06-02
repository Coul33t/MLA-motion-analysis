# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from interface_fonctions import *
from data_import import json_import
from tools import Person

class Ui_MainWindow(object):

    student = Person(r'', 'RannouP', 'Right', 'Pierre_Rannou')
    liste_descripteurs = []
    expert_data_repartion = {}
    
# Fonction liées aux boutons
    
    def activer_add_button(self) :
        if (self.defaut_name_edit.text() == "") :
            self.add_button.setEnabled(False)
        else : 
            self.add_button.setEnabled(True)

    def activer_articulation_button(self) :
        self.add_articulation_button.setEnabled(True)

    def on_btn_valider_clicked(self):
        expert = Person(r'', 'aurel', 'Right')
        path = self.data_path_edit_2.text()
        path_is_valid = check_path_validity(path)
        if (path_is_valid) :
            datatype_joints_list = []
            root = self.defauts_treeView.invisibleRootItem()
            for defaut in range (root.childCount()) :
                defaut_actuel = root.child(defaut) 
                descripteurs = {}
                for descripteur in range (defaut_actuel.childCount()) :
                    articulations = []
                    descripteur_actuel = defaut_actuel.child(descripteur)
                    nom = descripteur_actuel.text(0)

                    for articulation in range (descripteur_actuel.childCount()) :
                        articulation_actuelle = descripteur_actuel.child(articulation)
                        if (articulation_actuelle.text(1) == "False") :
                            articulations.append({'joint' : articulation_actuelle.text(0), 'laterality' : False})
                        else :
                            articulations.append({'joint' : articulation_actuelle.text(0), 'laterality' : True})
                    descripteurs[nom.replace(" ", "")] = articulations

                if (str.lower(defaut_actuel.text(0)) == "leaning" and not self.pb_leaning_check.isChecked()) :
                    break
                elif (str.lower(defaut_actuel.text(0)) == "elbow_move" and not self.pb_elbow_check.isChecked()) :
                    break
                elif (str.lower(defaut_actuel.text(0)) == "align_arm" and not self.pb_arm_check.isChecked()) :
                    break
                elif (str.lower(defaut_actuel.text(0)) == "javelin" and not self.pb_javelin_check.isChecked()) :
                    break
                else :
                    datatype_joints_list.append([str.lower(defaut_actuel.text(0)), descripteurs])    

            param = self.gather_parameters_info()    
            only_feedback_new_descriptors(expert, self.student, path, datatype_joints_list, self.expert_data_repartion, param)
        else : 
            error_msg = QtWidgets.QMessageBox()
            error_msg.setIcon(QtWidgets.QMessageBox.Critical)
            error_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            error_msg.setText("Le chemin spécifié est incorrect")
            error_msg.setInformativeText("Le chemin doit indiquer l'emplacement du dossier 'mixed' ")
            error_msg.setWindowTitle("Erreur de chemin")
            error_msg.exec_()
            return

    def load_joints_data(self) :
        path = self.data_path_edit.text()
        datatypes_list = get_datatypes_names(path, self.student.name)
        datatype = datatypes_list[0]
        joints_list = get_joint_list(path, self.student.name, datatype)

        self.descriptor_label.show()
        self.descriptor_box.show()
        self.articulation_display.show()
        self.add_articulation_button.show()
        self.lateralite_check.show()

        self.descriptor_box.addItems(datatypes_list)
        self.articulation_display.addItems(joints_list)

    def add_articulation(self) :
        self.defaut_name_label.show()
        self.defaut_name_edit.show()
        self.default_tree_recap.show()
        self.add_button.show()
        self.expert_data_label.show()
        self.expert_data_index_label.show()
        self.label_6.show()
        self.expert_data_min_index.show()
        self.expert_data_max_index.show()

        self.add_articulation_button.setEnabled(False)

        datatype = self.descriptor_box.currentText()
        articulation = self.articulation_display.currentItem().text()
        for descripteur in self.liste_descripteurs :
            if (descripteur[0] == datatype) :
                articulation = self.articulation_display.currentItem().text()
                QtWidgets.QTreeWidgetItem(descripteur[1], [articulation, str(self.lateralite_check.isChecked())])
                self.articulation_display.currentItem().setFlags(QtCore.Qt.NoItemFlags)
                return  

        datatype_display = QtWidgets.QTreeWidgetItem(self.default_tree_recap, [datatype])
        articulation = self.articulation_display.currentItem().text()
        QtWidgets.QTreeWidgetItem(datatype_display, [articulation, str(self.lateralite_check.isChecked())])    
        self.liste_descripteurs.append([datatype, datatype_display])   
        self.articulation_display.currentItem().setFlags(QtCore.Qt.NoItemFlags)

    def add_new_default(self):
        self.tabSecondaire.setCurrentIndex(2)
        new_default = QtWidgets.QTreeWidgetItem(self.defauts_treeView, [self.defaut_name_edit.text()])
        root = self.default_tree_recap.invisibleRootItem()
        for i in range (root.childCount()) :
            enfant = root.child(i) 
            descripteur = QtWidgets.QTreeWidgetItem(new_default, [enfant.text(0)])
            for j in range (enfant.childCount()) :
                QtWidgets.QTreeWidgetItem(descripteur, [enfant.child(j).text(0), enfant.child(j).text(1)])
        self.expert_data_repartion[str.lower(self.defaut_name_edit.text())] = [x+1 for x in range(self.expert_data_min_index.value() -1, self.expert_data_max_index.value() -1)]
        self.default_tree_recap.clear()
        self.defaut_name_edit.clear()
        self.articulation_display.clear()
        self.descriptor_box.setCurrentIndex(0)
        self.load_joints_data()
    
        return

    def remove_defaut(self) :
        root = self.defauts_treeView.invisibleRootItem()
        if (len(self.defauts_treeView.selectedItems()) == 0) :
            error_msg = QtWidgets.QMessageBox()
            error_msg.setIcon(QtWidgets.QMessageBox.Critical)
            error_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            error_msg.setText("Veuillez sélectionner un défaut à supprimer")
            error_msg.setWindowTitle("Erreur lors de la suppression")
            error_msg.exec_()
        else :
            for defaut in self.defauts_treeView.selectedItems():
                (defaut.parent() or root).removeChild(defaut)

# Fonction de récupération de données

    def gather_parameters_info(self):

            if (not self.pb_arm_check.isChecked() and not self.pb_elbow_check.isChecked() and not self.pb_javelin_check.isChecked() and not self.pb_leaning_check.isChecked()) :
                error_msg = QtWidgets.QMessageBox()
                error_msg.setIcon(QtWidgets.QMessageBox.Critical)
                error_msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                error_msg.setText("Aucun défaut n'est sélectionné")
                error_msg.setInformativeText('Veuillez cocher au moins un défaut dans la partie "Défauts à observer" de l\'onglet "Basique"')
                error_msg.setWindowTitle("Erreur de lancement")
                error_msg.exec_()
                return

            else :
                param = {}
                param['display'] = self.display_check.isChecked()
                param['scale'] = self.scale_check.isChecked()
                param['normalise'] = self.normalize_check.isChecked()
                param['algos'] = {self.algo_comboBox.currentText() : {'n_clusters' : self.centroids_spinBox.value()}}
                return param

    def save_defauts(self):
        data = []
        root = self.defauts_treeView.invisibleRootItem()
        for defaut in range (root.childCount()) :
            defaut_actuel = root.child(defaut) 
            descripteurs = {}
            for descripteur in range (defaut_actuel.childCount()) :
                articulations = []
                descripteur_actuel = defaut_actuel.child(descripteur)
                nom = descripteur_actuel.text(0)

                for articulation in range (descripteur_actuel.childCount()) :
                    articulation_actuelle = descripteur_actuel.child(articulation)
                    if (articulation_actuelle.text(1) == "False") :
                        articulations.append({'joint' : articulation_actuelle.text(0), 'laterality' : False})
                    else :
                        articulations.append({'joint' : articulation_actuelle.text(0), 'laterality' : True})
                descripteurs[nom.replace(" ", "")] = articulations
            data.append([defaut_actuel.text(0), descripteurs])
        with open('defauts.json', "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_defauts(self):
        with open('defauts.json', "r") as f:
            data = json.load(f)
        self.defauts_treeView.clear()

        for defaut in data :
            defautWidget = QtWidgets.QTreeWidgetItem(self.defauts_treeView, [defaut[0]])
            for descripteur in defaut[1] :
                descripteurWidget = QtWidgets.QTreeWidgetItem(defautWidget, [descripteur])
                for articulation in defaut[1][descripteur] :
                    QtWidgets.QTreeWidgetItem(descripteurWidget, [articulation['joint'], str(articulation['laterality'])])

# Fonction d'affichage

    def hide_descriptor_page(self) :
            self.descriptor_label.hide()
            self.descriptor_box.hide()
            self.articulation_display.hide()
            self.add_articulation_button.hide()
            self.lateralite_check.hide()
            self.defaut_name_label.hide()
            self.defaut_name_edit.hide()
            self.default_tree_recap.hide()
            self.add_button.hide()
            self.expert_data_label.hide()
            self.expert_data_index_label.hide()
            self.label_6.hide()
            self.expert_data_max_index.hide()
            self.expert_data_min_index.hide()

            self.load_button.setEnabled(False)
            self.add_button.setEnabled(False)
            self.add_articulation_button.setEnabled(False)

    def data_path_explorer(self) :
            explorer = QtWidgets.QFileDialog()
            explorer.setFileMode(QtWidgets.QFileDialog.Directory)
            explorer.exec_()
            if (len(explorer.selectedFiles()) > 0):
                self.data_path_edit.setText(explorer.selectedFiles()[0])
                self.data_path_edit_2.setText(explorer.selectedFiles()[0])
                self.load_button.setEnabled(True)
            return

    def refresh_joints_display(self) :
        path = self.data_path_edit.text()
        datatype = self.descriptor_box.currentText()
        joints_list = get_joint_list(path, self.student.name, datatype)
        self.articulation_display.clear()
        self.articulation_display.addItems(joints_list)

# Fonction d'initialisation de l'interface

    def initialise(self) :
            leaning = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Leaning"])
            meanSpeed = QtWidgets.QTreeWidgetItem(leaning, ["MeanSpeed"])
            QtWidgets.QTreeWidgetItem(meanSpeed, ["LeftShoulder", "False"])
            QtWidgets.QTreeWidgetItem(meanSpeed, ["RightShoulder", "False"])

            elbowMove = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Elbow_Move"])
            meanSpeed = QtWidgets.QTreeWidgetItem(elbowMove, ["MeanSpeed"])
            QtWidgets.QTreeWidgetItem(meanSpeed, ["LeftArm", "True"])
            QtWidgets.QTreeWidgetItem(meanSpeed, ["LeftShoulder", "True"])

            javelin = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Javelin"])
            distanceX = QtWidgets.QTreeWidgetItem(javelin, ["DistanceX"])
            QtWidgets.QTreeWidgetItem(distanceX, ["distanceRightHandHead", "True"])
            distanceY = QtWidgets.QTreeWidgetItem(javelin, ["Distance Y"])
            QtWidgets.QTreeWidgetItem(distanceY, ["distanceRightHandHead", "True"])
            distanceZ = QtWidgets.QTreeWidgetItem(javelin, ["Distance Z"])
            QtWidgets.QTreeWidgetItem(distanceZ, ["distanceRightHandHead", "True"])

            alignArm = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Align_Arm"])
            boundingBowWithMean = QtWidgets.QTreeWidgetItem(alignArm, ["BoundingBoxWidthMean"])
            QtWidgets.QTreeWidgetItem(boundingBowWithMean, ["HeadRightShoulderRightArmRightForeArmRightHand", "True"])
            boundingBoxWidthStd = QtWidgets.QTreeWidgetItem(alignArm, ["BoundingBoxWidthStd"])
            QtWidgets.QTreeWidgetItem(boundingBoxWidthStd, ["HeadRightShoulderRightArmRightForeArmRightHand", "True"])

            self.centroids_spinBox.setValue(2)

            self.pb_arm_check.setChecked(True)
            self.pb_elbow_check.setChecked(True)
            self.pb_javelin_check.setChecked(True)
            self.pb_leaning_check.setChecked(True)
            self.display_check.setChecked(True)

            self.pb_leaning_check.setToolTip('Le lanceur est-il trop penché lors du lancer ?')
            self.pb_elbow_check.setToolTip('Le coude du lanceur reste-t-il toujours à la même hauteur lors du lancer ?')
            self.pb_arm_check.setToolTip('Le lanceur garde-t-il le bras bien aligné lors du lancer ?')
            self.pb_javelin_check.setToolTip('La main du lanceur passe-t-elle à côté ou derrière sa tête lors du lancer ?')

            self.btn_valider.clicked.connect(self.on_btn_valider_clicked)
            self.btn_parcourir.clicked.connect(self.data_path_explorer)
            self.path_explore_button.clicked.connect(self.data_path_explorer)
            self.load_button.clicked.connect(self.load_joints_data)
            self.descriptor_box.currentIndexChanged.connect(self.refresh_joints_display)
            self.add_articulation_button.clicked.connect(self.add_articulation)
            self.add_button.clicked.connect(self.add_new_default)
            self.defaut_name_edit.textEdited.connect(self.activer_add_button)
            self.articulation_display.currentItemChanged.connect(self.activer_articulation_button)
            self.delete_defaut_button.clicked.connect(self.remove_defaut)
            self.save_defaut_button.clicked.connect(self.save_defauts)
            self.load_defauts_button.clicked.connect(self.load_defauts)
            self.hide_descriptor_page()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(459, 532)
        self.tabPrincipal = QtWidgets.QTabWidget(MainWindow)
        self.tabPrincipal.setGeometry(QtCore.QRect(0, 0, 461, 491))
        self.tabPrincipal.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabPrincipal.setObjectName("tabPrincipal")
        self.basicsTab = QtWidgets.QWidget()
        self.basicsTab.setObjectName("basicsTab")
        self.pb_box = QtWidgets.QGroupBox(self.basicsTab)
        self.pb_box.setGeometry(QtCore.QRect(0, 20, 451, 121))
        self.pb_box.setObjectName("pb_box")
        self.pb_leaning_check = QtWidgets.QCheckBox(self.pb_box)
        self.pb_leaning_check.setGeometry(QtCore.QRect(30, 20, 70, 17))
        self.pb_leaning_check.setObjectName("pb_leaning_check")
        self.pb_elbow_check = QtWidgets.QCheckBox(self.pb_box)
        self.pb_elbow_check.setGeometry(QtCore.QRect(30, 40, 101, 17))
        self.pb_elbow_check.setObjectName("pb_elbow_check")
        self.pb_arm_check = QtWidgets.QCheckBox(self.pb_box)
        self.pb_arm_check.setGeometry(QtCore.QRect(30, 60, 101, 17))
        self.pb_arm_check.setObjectName("pb_arm_check")
        self.pb_javelin_check = QtWidgets.QCheckBox(self.pb_box)
        self.pb_javelin_check.setGeometry(QtCore.QRect(30, 80, 70, 17))
        self.pb_javelin_check.setObjectName("pb_javelin_check")
        self.data_box = QtWidgets.QGroupBox(self.basicsTab)
        self.data_box.setGeometry(QtCore.QRect(0, 150, 451, 91))
        self.data_box.setObjectName("data_box")
        self.data_path_edit = QtWidgets.QLineEdit(self.data_box)
        self.data_path_edit.setGeometry(QtCore.QRect(20, 30, 261, 20))
        self.data_path_edit.setObjectName("data_path_edit")
        self.btn_parcourir = QtWidgets.QPushButton(self.data_box)
        self.btn_parcourir.setGeometry(QtCore.QRect(290, 30, 75, 23))
        self.btn_parcourir.setDefault(False)
        self.btn_parcourir.setFlat(False)
        self.btn_parcourir.setObjectName("btn_parcourir")
        self.msg_data_path = QtWidgets.QLabel(self.data_box)
        self.msg_data_path.setGeometry(QtCore.QRect(120, 60, 151, 16))
        self.msg_data_path.setObjectName("msg_data_path")
        self.tabPrincipal.addTab(self.basicsTab, "")
        self.advancedTab = QtWidgets.QWidget()
        self.advancedTab.setObjectName("advancedTab")
        self.tabSecondaire = QtWidgets.QTabWidget(self.advancedTab)
        self.tabSecondaire.setGeometry(QtCore.QRect(0, 0, 451, 471))
        self.tabSecondaire.setObjectName("tabSecondaire")
        self.algorithmTab = QtWidgets.QWidget()
        self.algorithmTab.setObjectName("algorithmTab")
        self.data_treatment_box = QtWidgets.QGroupBox(self.algorithmTab)
        self.data_treatment_box.setGeometry(QtCore.QRect(0, 10, 441, 141))
        self.data_treatment_box.setObjectName("data_treatment_box")
        self.scale_check = QtWidgets.QCheckBox(self.data_treatment_box)
        self.scale_check.setGeometry(QtCore.QRect(30, 20, 131, 31))
        self.scale_check.setObjectName("scale_check")
        self.normalize_check = QtWidgets.QCheckBox(self.data_treatment_box)
        self.normalize_check.setGeometry(QtCore.QRect(30, 50, 131, 31))
        self.normalize_check.setObjectName("normalize_check")
        self.algorithms_box = QtWidgets.QGroupBox(self.algorithmTab)
        self.algorithms_box.setGeometry(QtCore.QRect(0, 160, 441, 171))
        self.algorithms_box.setObjectName("algorithms_box")
        self.algo_comboBox = QtWidgets.QComboBox(self.algorithms_box)
        self.algo_comboBox.setGeometry(QtCore.QRect(140, 30, 121, 22))
        self.algo_comboBox.setObjectName("algo_comboBox")
        self.algo_comboBox.addItem("")
        self.algo_comboBox.addItem("")
        self.algo_comboBox.addItem("")
        self.algo_comboBox.addItem("")
        self.algo_label = QtWidgets.QLabel(self.algorithms_box)
        self.algo_label.setGeometry(QtCore.QRect(30, 30, 111, 16))
        self.algo_label.setObjectName("algo_label")
        self.centroids_label = QtWidgets.QLabel(self.algorithms_box)
        self.centroids_label.setGeometry(QtCore.QRect(30, 70, 111, 16))
        self.centroids_label.setObjectName("centroids_label")
        self.centroids_spinBox = QtWidgets.QSpinBox(self.algorithms_box)
        self.centroids_spinBox.setGeometry(QtCore.QRect(150, 70, 42, 22))
        self.centroids_spinBox.setObjectName("centroids_spinBox")
        self.display_box = QtWidgets.QGroupBox(self.algorithmTab)
        self.display_box.setGeometry(QtCore.QRect(0, 340, 441, 91))
        self.display_box.setObjectName("display_box")
        self.display_check = QtWidgets.QCheckBox(self.display_box)
        self.display_check.setGeometry(QtCore.QRect(30, 30, 131, 31))
        self.display_check.setObjectName("display_check")
        self.tabSecondaire.addTab(self.algorithmTab, "")
        self.descriptorsTab = QtWidgets.QWidget()
        self.descriptorsTab.setObjectName("descriptorsTab")
        self.data_path_label = QtWidgets.QLabel(self.descriptorsTab)
        self.data_path_label.setGeometry(QtCore.QRect(10, 10, 47, 16))
        self.data_path_label.setObjectName("data_path_label")
        self.path_explore_button = QtWidgets.QPushButton(self.descriptorsTab)
        self.path_explore_button.setGeometry(QtCore.QRect(210, 10, 75, 23))
        self.path_explore_button.setObjectName("path_explore_button")
        self.load_button = QtWidgets.QPushButton(self.descriptorsTab)
        self.load_button.setGeometry(QtCore.QRect(290, 10, 111, 23))
        self.load_button.setObjectName("load_button")
        self.data_path_edit_2 = QtWidgets.QLineEdit(self.descriptorsTab)
        self.data_path_edit_2.setGeometry(QtCore.QRect(80, 10, 113, 20))
        self.data_path_edit_2.setObjectName("data_path_edit_2")
        self.descriptor_label = QtWidgets.QLabel(self.descriptorsTab)
        self.descriptor_label.setGeometry(QtCore.QRect(10, 50, 61, 21))
        self.descriptor_label.setObjectName("descriptor_label")
        self.descriptor_box = QtWidgets.QComboBox(self.descriptorsTab)
        self.descriptor_box.setGeometry(QtCore.QRect(80, 50, 101, 22))
        self.descriptor_box.setObjectName("descriptor_box")
        self.add_articulation_button = QtWidgets.QPushButton(self.descriptorsTab)
        self.add_articulation_button.setGeometry(QtCore.QRect(100, 110, 75, 23))
        self.add_articulation_button.setObjectName("add_articulation_button")
        self.lateralite_check = QtWidgets.QCheckBox(self.descriptorsTab)
        self.lateralite_check.setGeometry(QtCore.QRect(250, 200, 70, 17))
        self.lateralite_check.setObjectName("lateralite_check")
        self.defaut_name_edit = QtWidgets.QLineEdit(self.descriptorsTab)
        self.defaut_name_edit.setGeometry(QtCore.QRect(250, 260, 181, 20))
        self.defaut_name_edit.setObjectName("defaut_name_edit")
        self.add_button = QtWidgets.QPushButton(self.descriptorsTab)
        self.add_button.setGeometry(QtCore.QRect(250, 360, 75, 23))
        self.add_button.setObjectName("add_button")
        self.defaut_name_label = QtWidgets.QLabel(self.descriptorsTab)
        self.defaut_name_label.setGeometry(QtCore.QRect(250, 240, 81, 16))
        self.defaut_name_label.setObjectName("defaut_name_label")
        self.articulation_display = QtWidgets.QListWidget(self.descriptorsTab)
        self.articulation_display.setGeometry(QtCore.QRect(190, 40, 251, 151))
        self.articulation_display.setObjectName("articulation_display")
        self.default_tree_recap = QtWidgets.QTreeWidget(self.descriptorsTab)
        self.default_tree_recap.setGeometry(QtCore.QRect(10, 230, 231, 191))
        self.default_tree_recap.setObjectName("default_tree_recap")
        self.expert_data_label = QtWidgets.QLabel(self.descriptorsTab)
        self.expert_data_label.setGeometry(QtCore.QRect(250, 290, 111, 16))
        self.expert_data_label.setObjectName("expert_data_label")
        self.expert_data_min_index = QtWidgets.QSpinBox(self.descriptorsTab)
        self.expert_data_min_index.setGeometry(QtCore.QRect(310, 310, 42, 22))
        self.expert_data_min_index.setMinimum(1)
        self.expert_data_min_index.setObjectName("expert_data_min_index")
        self.expert_data_max_index = QtWidgets.QSpinBox(self.descriptorsTab)
        self.expert_data_max_index.setGeometry(QtCore.QRect(370, 310, 42, 22))
        self.expert_data_max_index.setMinimum(2)
        self.expert_data_max_index.setObjectName("expert_data_max_index")
        self.expert_data_index_label = QtWidgets.QLabel(self.descriptorsTab)
        self.expert_data_index_label.setGeometry(QtCore.QRect(260, 310, 47, 21))
        self.expert_data_index_label.setObjectName("expert_data_index_label")
        self.label_6 = QtWidgets.QLabel(self.descriptorsTab)
        self.label_6.setGeometry(QtCore.QRect(360, 310, 16, 21))
        self.label_6.setObjectName("label_6")
        self.tabSecondaire.addTab(self.descriptorsTab, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.defauts_treeView = QtWidgets.QTreeWidget(self.tab)
        self.defauts_treeView.setGeometry(QtCore.QRect(0, 30, 441, 351))
        self.defauts_treeView.setObjectName("defauts_treeView")
        self.defauts_treeView_label = QtWidgets.QLabel(self.tab)
        self.defauts_treeView_label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.defauts_treeView_label.setObjectName("defauts_treeView_label")
        self.delete_defaut_button = QtWidgets.QPushButton(self.tab)
        self.delete_defaut_button.setGeometry(QtCore.QRect(10, 400, 131, 23))
        self.delete_defaut_button.setObjectName("delete_defaut_button")
        self.load_defauts_button = QtWidgets.QPushButton(self.tab)
        self.load_defauts_button.setGeometry(QtCore.QRect(160, 400, 121, 23))
        self.load_defauts_button.setObjectName("load_defauts_button")
        self.save_defaut_button = QtWidgets.QPushButton(self.tab)
        self.save_defaut_button.setGeometry(QtCore.QRect(310, 400, 131, 23))
        self.save_defaut_button.setObjectName("save_defaut_button")
        self.tabSecondaire.addTab(self.tab, "")
        self.tabPrincipal.addTab(self.advancedTab, "")
        self.btn_valider = QtWidgets.QPushButton(MainWindow)
        self.btn_valider.setGeometry(QtCore.QRect(370, 500, 75, 23))
        self.btn_valider.setObjectName("btn_valider")

        self.retranslateUi(MainWindow)
        self.tabPrincipal.setCurrentIndex(0)
        self.tabSecondaire.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.initialise()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MLA Settings"))
        self.pb_box.setTitle(_translate("MainWindow", "Défauts à observer"))
        self.pb_leaning_check.setText(_translate("MainWindow", "Leaning"))
        self.pb_elbow_check.setText(_translate("MainWindow", "Elbow Move"))
        self.pb_arm_check.setText(_translate("MainWindow", "Arm Alignment"))
        self.pb_javelin_check.setText(_translate("MainWindow", "Javelin"))
        self.data_box.setTitle(_translate("MainWindow", "Accès aux données"))
        self.btn_parcourir.setText(_translate("MainWindow", "Parcourir..."))
        self.msg_data_path.setText(_translate("MainWindow", "Chemin vers le dossier \"mixed\""))
        self.tabPrincipal.setTabText(self.tabPrincipal.indexOf(self.basicsTab), _translate("MainWindow", "Basique"))
        self.data_treatment_box.setTitle(_translate("MainWindow", "Traitement des données"))
        self.scale_check.setText(_translate("MainWindow", "Scale"))
        self.normalize_check.setText(_translate("MainWindow", "Normalize"))
        self.algorithms_box.setTitle(_translate("MainWindow", "Algorithmes"))
        self.algo_comboBox.setItemText(0, _translate("MainWindow", "k-means"))
        self.algo_comboBox.setItemText(1, _translate("MainWindow", "option 2"))
        self.algo_comboBox.setItemText(2, _translate("MainWindow", "option 3"))
        self.algo_comboBox.setItemText(3, _translate("MainWindow", "option 4"))
        self.algo_label.setText(_translate("MainWindow", "Algorithme à utiliser :"))
        self.centroids_label.setText(_translate("MainWindow", "Nombre de centroïdes :"))
        self.display_box.setTitle(_translate("MainWindow", "Affichage"))
        self.display_check.setText(_translate("MainWindow", "Afficher les graphes"))
        self.tabSecondaire.setTabText(self.tabSecondaire.indexOf(self.algorithmTab), _translate("MainWindow", "Algorithmes"))
        self.data_path_label.setText(_translate("MainWindow", "Chemin :"))
        self.path_explore_button.setText(_translate("MainWindow", "Parcourir..."))
        self.load_button.setText(_translate("MainWindow", "Charger les donnees"))
        self.descriptor_label.setText(_translate("MainWindow", "Descripteur :"))
        self.add_articulation_button.setText(_translate("MainWindow", "Ajouter"))
        self.lateralite_check.setText(_translate("MainWindow", "Latéralité"))
        self.add_button.setText(_translate("MainWindow", "Ajouter"))
        self.defaut_name_label.setText(_translate("MainWindow", "Nom du défaut :"))
        self.default_tree_recap.headerItem().setText(0, _translate("MainWindow", "Descripteurs"))
        self.default_tree_recap.headerItem().setText(1, _translate("MainWindow", "Latéralité"))
        self.expert_data_label.setText(_translate("MainWindow", "Données de l\'expert :"))
        self.expert_data_index_label.setText(_translate("MainWindow", "De l\'index "))
        self.label_6.setText(_translate("MainWindow", "à"))
        self.tabSecondaire.setTabText(self.tabSecondaire.indexOf(self.descriptorsTab), _translate("MainWindow", "Descripteurs"))
        self.defauts_treeView_label.setText(_translate("MainWindow", "Liste des défauts :"))
        self.defauts_treeView.headerItem().setText(0, _translate("MainWindow", "Défauts"))
        self.delete_defaut_button.setText(_translate("MainWindow", "Supprimer le défaut"))
        self.load_defauts_button.setText(_translate("MainWindow", "Charger des défauts"))
        self.save_defaut_button.setText(_translate("MainWindow", "Sauvegarder les défauts"))
        self.tabSecondaire.setTabText(self.tabSecondaire.indexOf(self.tab), _translate("MainWindow", "Résumé"))
        self.tabPrincipal.setTabText(self.tabPrincipal.indexOf(self.advancedTab), _translate("MainWindow", "Avancé"))
        self.btn_valider.setText(_translate("MainWindow", "Valider"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
