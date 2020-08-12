import os
import sys
from shutil import rmtree
from distutils.dir_util import copy_tree

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler, Normalizer

from data_import import json_import

from data_processing import (run_clustering,
                             plot_good_vs_student_all_data,
                             good_and_bad_vs_student_all_data)

from data_selection import data_gathering

from data_visualization import (plot_PCA,
                                multi_plot_PCA,
                                plot_all_defaults)

from feedback_tools import *

from tools import Person, merge_list_of_dictionnaries

import constants as cst

from constants import problemes_et_solutions as problems_and_advices

from PyQt5 import QtCore, QtGui, QtWidgets


#TODO: use a dict for removed values, instead of ad-hoc value like now
#      (see compute_distance_to_clusters for an example)

@dataclass
class Problem :
    name : str
    caracs : list
    laterality : bool

class Ui_MainWindow(object):

    def on_btn_valider_clicked(self):
       sys.exit(0)

    def data_path_explorer(self) :
        explorer = QtWidgets.QFileDialog()
        explorer.setFileMode(QtWidgets.QFileDialog.Directory)
        explorer.exec_()
        self.data_path_edit.setText(explorer.selectedFiles()[0])
        self.data_path_edit_2.setText(explorer.selectedFiles()[0])
        return

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
        self.comboBox = QtWidgets.QComboBox(self.algorithms_box)
        self.comboBox.setGeometry(QtCore.QRect(140, 30, 121, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_2 = QtWidgets.QLabel(self.algorithms_box)
        self.label_2.setGeometry(QtCore.QRect(30, 30, 111, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.algorithms_box)
        self.label_3.setGeometry(QtCore.QRect(30, 70, 111, 16))
        self.label_3.setObjectName("label_3")
        self.spinBox = QtWidgets.QSpinBox(self.algorithms_box)
        self.spinBox.setGeometry(QtCore.QRect(150, 70, 42, 22))
        self.spinBox.setObjectName("spinBox")
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
        self.articulation_display = QtWidgets.QTextBrowser(self.descriptorsTab)
        self.articulation_display.setGeometry(QtCore.QRect(190, 50, 241, 141))
        self.articulation_display.setObjectName("articulation_display")
        self.lateralite_check = QtWidgets.QCheckBox(self.descriptorsTab)
        self.lateralite_check.setGeometry(QtCore.QRect(250, 200, 70, 17))
        self.lateralite_check.setObjectName("lateralite_check")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.descriptorsTab)
        self.textBrowser_2.setGeometry(QtCore.QRect(0, 240, 241, 141))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.defaut_name_edit = QtWidgets.QLineEdit(self.descriptorsTab)
        self.defaut_name_edit.setGeometry(QtCore.QRect(250, 260, 181, 20))
        self.defaut_name_edit.setObjectName("defaut_name_edit")
        self.add_button = QtWidgets.QPushButton(self.descriptorsTab)
        self.add_button.setGeometry(QtCore.QRect(250, 360, 75, 23))
        self.add_button.setObjectName("add_button")
        self.defaut_name_label = QtWidgets.QLabel(self.descriptorsTab)
        self.defaut_name_label.setGeometry(QtCore.QRect(250, 240, 81, 16))
        self.defaut_name_label.setObjectName("defaut_name_label")
        self.tabSecondaire.addTab(self.descriptorsTab, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.defauts_treeView = QtWidgets.QTreeWidget(self.tab)
        self.defauts_treeView.setGeometry(QtCore.QRect(0, 30, 441, 411))
        self.defauts_treeView.setObjectName("defauts_treeView")
        self.defauts_treeView.setHeaderLabel("Defauts")

        leaning = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Leaning"])
        meanSpeed = QtWidgets.QTreeWidgetItem(leaning, ["Mean Speed"])
        QtWidgets.QTreeWidgetItem(meanSpeed, ["Left Shoulder"])
        QtWidgets.QTreeWidgetItem(meanSpeed, ["Right Shoulder"])

        elbowMove = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Elbow Move"])
        meanSpeed = QtWidgets.QTreeWidgetItem(elbowMove, ["Mean Speed"])
        QtWidgets.QTreeWidgetItem(meanSpeed, ["Left Arm"])
        QtWidgets.QTreeWidgetItem(meanSpeed, ["Right Arm"])

        javelin = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Javelin"])
        distanceX = QtWidgets.QTreeWidgetItem(javelin, ["Distance X"])
        QtWidgets.QTreeWidgetItem(distanceX, ["Distance Right Hand - Head"])
        distanceY = QtWidgets.QTreeWidgetItem(javelin, ["Distance Y"])
        QtWidgets.QTreeWidgetItem(distanceY, ["Distance Right Hand - Head"])
        distanceZ = QtWidgets.QTreeWidgetItem(javelin, ["Distance Z"])
        QtWidgets.QTreeWidgetItem(distanceZ, ["Distance Right Hand - Head"])

        alignArm = QtWidgets.QTreeWidgetItem(self.defauts_treeView, ["Align Arm"])
        boundingBowWithMean = QtWidgets.QTreeWidgetItem(alignArm, ["Bounding Box with Mean"])
        QtWidgets.QTreeWidgetItem(boundingBowWithMean, ["Head - Right Shoulder - Right Arm - Right Forearm - RightHand"])
        boundingBoxWidthStd = QtWidgets.QTreeWidgetItem(alignArm, ["Bounding Box with std"])
        QtWidgets.QTreeWidgetItem(boundingBoxWidthStd, ["Head - Right Shoulder - Right Arm - Right Forearm - RightHand"])

        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(10, 10, 91, 16))
        self.label.setObjectName("label")
        self.tabSecondaire.addTab(self.tab, "")
        self.tabPrincipal.addTab(self.advancedTab, "")
        self.btn_valider = QtWidgets.QPushButton(MainWindow)
        self.btn_valider.setGeometry(QtCore.QRect(370, 500, 75, 23))
        self.btn_valider.setObjectName("btn_valider")

        self.retranslateUi(MainWindow)
        self.tabPrincipal.setCurrentIndex(0)
        self.tabSecondaire.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.btn_valider.clicked.connect(self.on_btn_valider_clicked)
        self.btn_parcourir.clicked.connect(self.data_path_explorer)
        self.path_explore_button.clicked.connect(self.data_path_explorer)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MLA Settings"))
        self.pb_box.setTitle(_translate("MainWindow", "Problemes a observer"))
        self.pb_leaning_check.setText(_translate("MainWindow", "Leaning"))
        self.pb_elbow_check.setText(_translate("MainWindow", "Elbow Move"))
        self.pb_arm_check.setText(_translate("MainWindow", "Arm Alignment"))
        self.pb_javelin_check.setText(_translate("MainWindow", "Javelin"))
        self.data_box.setTitle(_translate("MainWindow", "Acces aux donnees"))
        self.btn_parcourir.setText(_translate("MainWindow", "Parcourir..."))
        self.msg_data_path.setText(_translate("MainWindow", "Chemin vers le dossier \"mixed\""))
        self.tabPrincipal.setTabText(self.tabPrincipal.indexOf(self.basicsTab), _translate("MainWindow", "Basique"))
        self.data_treatment_box.setTitle(_translate("MainWindow", "Traitement des donnees"))
        self.scale_check.setText(_translate("MainWindow", "Scale"))
        self.normalize_check.setText(_translate("MainWindow", "Normalize"))
        self.algorithms_box.setTitle(_translate("MainWindow", "Algorithmes"))
        self.comboBox.setItemText(0, _translate("MainWindow", "k-means"))
        self.comboBox.setItemText(1, _translate("MainWindow", "option 2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "option 3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "option 4"))
        self.label_2.setText(_translate("MainWindow", "Algorithme a utiliser :"))
        self.label_3.setText(_translate("MainWindow", "Nombre de centroides :"))
        self.display_box.setTitle(_translate("MainWindow", "Affichage"))
        self.display_check.setText(_translate("MainWindow", "Afficher les graphes"))
        self.tabSecondaire.setTabText(self.tabSecondaire.indexOf(self.algorithmTab), _translate("MainWindow", "Algorithmes"))
        self.data_path_label.setText(_translate("MainWindow", "Chemin :"))
        self.path_explore_button.setText(_translate("MainWindow", "Parcourir..."))
        self.load_button.setText(_translate("MainWindow", "Charger les donnees"))
        self.descriptor_label.setText(_translate("MainWindow", "Descripteur :"))
        self.add_articulation_button.setText(_translate("MainWindow", "Ajouter"))
        self.lateralite_check.setText(_translate("MainWindow", "Lateralite"))
        self.add_button.setText(_translate("MainWindow", "Ajouter"))
        self.defaut_name_label.setText(_translate("MainWindow", "Nom du defaut :"))
        self.tabSecondaire.setTabText(self.tabSecondaire.indexOf(self.descriptorsTab), _translate("MainWindow", "Descripteurs"))
        self.label.setText(_translate("MainWindow", "Liste des defauts :"))
        self.tabSecondaire.setTabText(self.tabSecondaire.indexOf(self.tab), _translate("MainWindow", "Resume"))
        self.tabPrincipal.setTabText(self.tabPrincipal.indexOf(self.advancedTab), _translate("MainWindow", "Avance"))
        self.btn_valider.setText(_translate("MainWindow", "Valider"))

def import_data(path, import_find):
    original_data = []

    # Gathering the data
    # If it's a list, then we have to import multiple people's data
    if isinstance(import_find[0], list):
        for to_import in import_find[0]:
            original_data.extend(json_import(path, to_import))
    # Else, it's just a name, so we import their data
    else:
        original_data = json_import(path, import_find)

    if not original_data:
        print('ERROR: no data found (check your names).')
        return

    return original_data

def feedback():
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students/mixed'
    # Expert Data
    name = 'aurel'
    expert_data = import_data(path, name)
    # Student data
    name = 'me'
    student_data = import_data(path, name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = 'Right'

    for motion in student_data:
        motion.laterality = 'Left'

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])

    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])

    datatype_joints_list.append(['javelin', {'PosX': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosY': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosZ': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}]
                                            }])

    datatype_joints_list.append(['align_arm', {'BoundingBoxMinusX': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusX':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusY': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusY':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusZ': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusZ':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])



    # Scaling and normalisaing (nor not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    # Setting the data repartition
    expert_data_repartion = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 20)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}

    # Algorithm to test
    algos = {'k-means': {'n_clusters': 2}}

    distances_and_clusters = []
    results = []
    std_features_all = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]
        expert_sub_data = expert_data[:10] + expert_data[min(expert_data_repartion[problem[0]])-1:max(expert_data_repartion[problem[0]])]

        for algo, param in algos.items():
            print(problem[0])

            model = run_clustering(path, expert_sub_data, name, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale, normalise_features=normalise,
                                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid(std_features)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:
            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

        # Compute the distance from the student's centroid to the expert's ones
        distances_to_centroid = compute_distance(model[0].cluster_centers_, std_centroid)

        # Get the distance from the student's centroid to the line between the two expert's centroids
        distance_from_line = get_distance_from_expert_centoids_line(model[0].cluster_centers_, std_centroid)
        # Used to check if the student's centroid is between the expert centroids (diamond shape)
        distance_from_line /= (dst_pts(model[0].cluster_centers_[0], model[0].cluster_centers_[1]) / 2)

        # Normalise the distances to the centroids
        summ = sum(distances_to_centroid)
        for i, distance in enumerate(distances_to_centroid):
            distances_to_centroid[i] = distance/summ

        # Get the most probable cluster label for expert data
        clusters_label = get_cluster_label(expert_sub_data, expert_data_repartion, model[0].labels_)
        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)
        distances_and_clusters.append(mix_res)

        res = (model[0], model[2], model[0].labels_, model[1], problem[0], std_features, get_trapezoid(model, std_centroid), get_circle(model, std_centroid))
        results.append(res)
        std_features_all.append(std_features)

    models = [x[0] for x in results]
    features = [x[1] for x in results]

    for i, feat in enumerate(features):
        features[i] = np.concatenate([feat, std_features_all[i]], axis=0)

    centroids = [x[0].cluster_centers_ for x in results]
    labels = [x[2] for x in results]

    for i, lab in enumerate(labels):
        max_val = max(lab)
        labels[i] = np.concatenate([lab, [max_val+1 for x in range(len(std_features))]], axis=0)

    trapezoids = [x[6] for x in results]
    circles = [x[7] for x in results]

    sss = [x[3]['ss'] for x in results]
    names = ['k-means ' + x[4] for x in results]
    title = 'None'

    clusters_names = [x[0] for x in distances_and_clusters]

    std_data = [[std_feat, [max(labels[i])+1 for x in range(len(std_features_all[i]))]] for i, std_feat in enumerate(std_features_all)]

    multi_plot_PCA(features, labels, clusters_names, names, models, sss, title, trapezoids, circles, only_centroids=True, centroids=centroids, std_data=std_data)

def only_feedback(expert, student, path):
    folders_path = path
    if folders_path.split('/')[-1] == 'mixed':
        folders_path = "/".join(path.split('/')[:-1])

    if not take_last_data(folders_path, student, expert, number=9):
        print(f'ERROR: {folders_path} does not exists')
        return

    # Expert Data
    expert_data = import_data(path, expert.name)
    # Student data
    student_data = import_data(path, student.name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = expert.laterality

    for motion in student_data:
        motion.laterality = student.laterality

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])

    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])

    datatype_joints_list.append(['javelin', {'PosX': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosY': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}],
                                             'PosZ': [{'joint': 'LeftHand', 'laterality': True},
                                                      {'joint': 'Head',     'laterality': False}]
                                            }])

    datatype_joints_list.append(['align_arm', {'BoundingBoxMinusX': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusX':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusY': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusY':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxMinusZ': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxPlusZ':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])


    # Scaling and normalisaing (nor not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    expert_data_repartion = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 19)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}
    
    # Algorithm(s) to use
    algos = {'k-means': {'n_clusters': 2}}

    clustering_problems = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]

        expert_sub_data = expert_data[:10] + expert_data[min(expert_data_repartion[problem[0]])-1:max(expert_data_repartion[problem[0]])]

        for algo, param in algos.items():
            print(problem[0])

            model = run_clustering(expert_sub_data, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale, normalise_features=normalise,
                                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid(std_features)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:
            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

        # Compute the distance from the student's centroid to the expert's ones
        distances_to_centroid = compute_distance(model[0].cluster_centers_, std_centroid)

        # Get the distance from the student's centroid to the line between the two expert's centroids
        distance_from_line = get_distance_from_expert_centoids_line(model[0].cluster_centers_, std_centroid)
        # Used to check if the student's centroid is between the expert centroids (diamond shape)
        distance_from_line /= (dst_pts(model[0].cluster_centers_[0], model[0].cluster_centers_[1]) / 2)

        # Normalise the distances to the centroids
        summ = sum(distances_to_centroid)
        for i, distance in enumerate(distances_to_centroid):
            distances_to_centroid[i] = distance/summ

        # Get the most probable cluster label for expert data
        clusters_label = get_cluster_label(expert_sub_data, expert_data_repartion, model[0].labels_)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #TODO: display numbers of motions in labelled clusters#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        get_closeness_to_clusters(clusters_label, std_features, model[0].cluster_centers_, problem[0])

        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)

        clusters_names = get_cluster_labels_from_data_repartition(model[0].labels_, model[0].cluster_centers_)

        clus_prob = ClusteringProblem(problem=problem[0],
                                      model=model[0], features=model[2],
                                      labels=model[0].labels_,
                                      centroids=model[0].cluster_centers_,
                                      sil_score=model[1]['ss'],
                                      clusters_names=clusters_names,
                                      algo_name=problem[0],
                                      std_data=std_features,
                                      std_centroid=get_centroid(std_features),
                                      distance_to_line=distance_from_line)

        clus_prob.trapezoid = get_trapezoid(model, std_centroid)
        clus_prob.circles = get_circle(model, std_centroid)
        clus_prob.std_data = std_features

        clustering_problems.append(clus_prob)

    give_two_advices(clustering_problems)
    plot_all_defaults(clustering_problems, only_centroids=False)

def only_feedback_new_descriptors(expert, student, path, display=True):
    folders_path = path
    if folders_path.split('/')[-1] == 'mixed':
        folders_path = "/".join(path.split('/')[:-1])

    # if not take_last_data(folders_path, student, expert, number=9):
    #     print(f'ERROR: {folders_path} does not exists')
    #     return

    # Expert Data
    expert_data = import_data(path, expert.name)
    # Student data
    student_data = import_data(path, student.name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = expert.laterality

    for motion in student_data:
        motion.laterality = student.laterality

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])
   
    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])
    
    datatype_joints_list.append(['javelin', {'DistanceX': [{'joint': 'distanceRightHandHead', 'laterality': True}],
                                             'DistanceY': [{'joint': 'distanceRightHandHead', 'laterality': True}],
                                             'DistanceZ': [{'joint': 'distanceRightHandHead', 'laterality': True}]
                                             }])
    
    datatype_joints_list.append(['align_arm', {'BoundingBoxWidthMean': [{'joint': 'HeadRightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxWidthStd': [{'joint': 'HeadRightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])
   
    # Scaling and normalisaing (or not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    expert_data_repartion = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 19)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}

    # Algorithm(s) to use
    algos = {'k-means': {'n_clusters': 2}}

    clustering_problems = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]

        expert_sub_data = expert_data[:10] + expert_data[min(expert_data_repartion[problem[0]])-1:max(expert_data_repartion[problem[0]])]

        # if problem[0] == 'javelin':
            # del expert_sub_data[16]
            # del expert_sub_data[15]
            # del expert_sub_data[12]
            # del expert_sub_data[3]
            # del expert_sub_data[2]
            # del expert_sub_data[1]
            # del expert_sub_data[0]

        if problem[0] == 'align_arm':
            del expert_sub_data[15]
            del expert_sub_data[9]
            del expert_sub_data[8]

        for algo, param in algos.items():
            print(problem[0])

            # true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            # if problem[0] == 'align_arm':
            #     true_labels = [0, 0, 0, 0, 0, 0, 0, 0,
            #                    1, 1, 1, 1, 1, 1, 1, 1, 1]
            # elif problem[0] == 'leaning':
            #     true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #                    1, 1, 1, 1, 1, 1, 1, 1, 1]

            model = run_clustering(expert_sub_data, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale,
                                   normalise_features=normalise, true_labels=None,
                                   verbose=False, to_file=True, to_json=True,
                                   return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid(std_features)

        # Compute the distance from the student's centroid to the expert's ones
        distances_to_centroid = compute_distance(model[0].cluster_centers_, std_centroid)

        # Get the distance from the student's centroid to the line between the two expert's centroids
        distance_from_line = get_distance_from_expert_centoids_line(model[0].cluster_centers_, std_centroid)
        # Used to check if the student's centroid is between the expert centroids (diamond shape)
        distance_from_line /= (dst_pts(model[0].cluster_centers_[0], model[0].cluster_centers_[1]) / 2)

        # Normalise the distances to the centroids
        summ = sum(distances_to_centroid)
        for i, distance in enumerate(distances_to_centroid):
            distances_to_centroid[i] = distance/summ

        # Get the most probable cluster label for expert data
        clusters_label = get_cluster_label(expert_sub_data, expert_data_repartion, model[0].labels_)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #TODO: display numbers of motions in labelled clusters#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        get_closeness_to_clusters(clusters_label, std_features, model[0].cluster_centers_, problem[0], to_csv=False)

        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)

        clusters_names = get_cluster_labels_from_data_repartition(model[0].labels_, model[0].cluster_centers_)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:

            # model[2] = normalize(model[2])
            # model[0].cluster_centers_ = normalize(model[0].cluster_centers_)
            # std_centroid = normalize(std_centroid.reshape(1, -1))[0]

            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

        # mixed = np.concatenate((model[2], std_features, model[0].cluster_centers_, std_centroid.reshape(1, -1)), axis=0)
        # min_max_scaler = MinMaxScaler()
        # mixed =  min_max_scaler.fit_transform(mixed)

        # model[2] = mixed[0:len(model[2]), :]
        # std_features = mixed[len(model[2]):len(model[2])+len(std_features), :]
        # model[0].cluster_centers_ = mixed[len(model[2])+len(std_features):-1, :]
        # std_centroid = mixed[-1, :]

        clus_prob = ClusteringProblem(problem=problem[0],
                                      model=model[0], features=model[2],
                                      labels=model[0].labels_,
                                      centroids=model[0].cluster_centers_,
                                      sil_score=model[1]['ss'],
                                      clusters_names=clusters_names,
                                      algo_name=problem[0],
                                      std_data=std_features,
                                      std_centroid=get_centroid(std_features),
                                      distance_to_line=distance_from_line)

        clus_prob.trapezoid = get_trapezoid(model, std_centroid)
        clus_prob.circles = get_circle(model, std_centroid)
        clus_prob.std_data = std_features

        clustering_problems.append(clus_prob)

    give_two_advices(clustering_problems)
    plot_all_defaults(clustering_problems, only_centroids=True)

    if display:
        all_features_to_extract = merge_list_of_dictionnaries([x[1] for x in datatype_joints_list])
        expert_good_data = expert_data[min(expert_data_repartion['good'])-1:max(expert_data_repartion['good'])]
        plot_good_vs_student_all_data(expert_good_data, student_data, algos, all_features_to_extract, only_centroids=True)

    if True:
        expert_bad_data = expert_data[max(expert_data_repartion['good']):]
        good_and_bad_vs_student_all_data(expert_good_data, expert_bad_data, student_data, algos, all_features_to_extract, only_centroids=True)

def compute_distance_to_clusters(expert, student, path, begin, end, fullname=True):
    folders_path = path
    if folders_path.split('/')[-1] == 'mixed':
        folders_path = "/".join(path.split('/')[:-1])

    if not take_specific_data(folders_path, student, expert, begin=begin, end=end, fullname=fullname):
        print(f'ERROR: {folders_path} does not exists')
        return

    # Expert Data
    expert_data = import_data(path + 'mixed', expert.name)
    # Student data
    student_data = import_data(path + 'mixed', student.name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = expert.laterality

    for motion in student_data:
        motion.laterality = student.laterality

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])

    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])

    datatype_joints_list.append(['javelin', {'DistanceX': [{'joint': 'distanceRightHandHead', 'laterality': True}],
                                             'DistanceY': [{'joint': 'distanceRightHandHead', 'laterality': True}],
                                             'DistanceZ': [{'joint': 'distanceRightHandHead', 'laterality': True}]
                                             }])

    datatype_joints_list.append(['align_arm', {'BoundingBoxWidthMean': [{'joint': 'HeadRightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxWidthStd': [{'joint': 'HeadRightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])



    # Scaling and normalisaing (nor not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    removed_values = {'leaning': [19],
                      'align_arm': [15, 9, 8]}

    expert_data_repartion = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 20)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}

    # Algorithm(s) to use
    algos = {'k-means': {'n_clusters': 2}}

    clustering_problems = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]

        expert_sub_data = expert_data[:10] + expert_data[min(expert_data_repartion[problem[0]])-1:max(expert_data_repartion[problem[0]])]
        full_expert_data = expert_sub_data.copy()

        full_features = data_gathering(full_expert_data, datatype_joints)

        if problem[0] == 'align_arm':
            del expert_sub_data[15]
            del expert_sub_data[9]
            del expert_sub_data[8]

        if problem[0] == 'leaning':
            del expert_sub_data[19]

        for algo, param in algos.items():
            #print(problem[0])

            model = run_clustering(expert_sub_data, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale, normalise_features=normalise,
                                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid(std_features)

        # Compute the distance from the student's centroid to the expert's ones
        distances_to_centroid = compute_distance(model[0].cluster_centers_, std_centroid)

        # Get the distance from the student's centroid to the line between the two expert's centroids
        distance_from_line = get_distance_from_expert_centoids_line(model[0].cluster_centers_, std_centroid)
        # Used to check if the student's centroid is between the expert centroids (diamond shape)
        distance_from_line /= (dst_pts(model[0].cluster_centers_[0], model[0].cluster_centers_[1]) / 2)

        # Normalise the distances to the centroids
        summ = sum(distances_to_centroid)
        for i, distance in enumerate(distances_to_centroid):
            distances_to_centroid[i] = distance/summ

        # Get the most probable cluster label for expert data
        clusters_label = get_cluster_label(expert_sub_data, expert_data_repartion, model[0].labels_)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #TODO: display numbers of motions in labelled clusters#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        get_closeness_to_clusters(clusters_label,
                                  std_features,
                                  model[0].cluster_centers_,
                                  problem[0],
                                  name=student.name,
                                  to_print=False,
                                  to_csv=False)

        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)

        clusters_names = get_cluster_labels_from_data_repartition(model[0].labels_, model[0].cluster_centers_)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        # if len(model[0].cluster_centers_[0]) > 2:

        #     # model[2] = normalize(model[2])
        #     # model[0].cluster_centers_ = normalize(model[0].cluster_centers_)
        #     # std_centroid = normalize(std_centroid.reshape(1, -1))[0]

        #     pca = PCA(n_components=2, copy=True)
        #     pca.fit(model[2])

        #     model[2] = pca.transform(model[2])
        #     full_features = pca.transform(full_features)
        #     model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
        #     std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
        #     std_features = pca.transform(std_features)

        # mixed = np.concatenate((model[2], std_features, model[0].cluster_centers_, std_centroid.reshape(1, -1)), axis=0)
        # min_max_scaler = MinMaxScaler()
        # mixed =  min_max_scaler.fit_transform(mixed)

        # model[2] = mixed[0:len(model[2]), :]
        # std_features = mixed[len(model[2]):len(model[2])+len(std_features), :]
        # model[0].cluster_centers_ = mixed[len(model[2])+len(std_features):-1, :]
        # std_centroid = mixed[-1, :]

        clus_prob = ClusteringProblem(problem=problem[0],
                                      model=model[0], features=full_features,
                                      labels=expert_data_repartion,
                                      centroids=model[0].cluster_centers_,
                                      sil_score=model[1]['ss'],
                                      clusters_names=clusters_names,
                                      algo_name=problem[0],
                                      std_data=std_features,
                                      std_centroid=get_centroid(std_features),
                                      distance_to_line=distance_from_line)

        clus_prob.trapezoid = get_trapezoid(model, std_centroid)
        clus_prob.circles = get_circle(model, std_centroid)
        clus_prob.std_data = std_features

        clustering_problems.append(clus_prob)

    get_global_closeness_to_good_cluster_2(clustering_problems,
                                           name=student.name,
                                           to_print=True,
                                           to_csv=False,
                                           to_xlsx=True)

def rotated_feedback_comparison(expert, student, path, begin, end):
    folders_path = path
    if folders_path.split('/')[-1] == 'mixed':
        folders_path = "/".join(path.split('/')[:-1])

    if not take_specific_data(folders_path, student, expert, begin=begin, end=end):
        print(f'ERROR: {folders_path} does not exists')
        return

    # Expert Data
    expert_data = import_data(path, expert.name)
    # Student data
    student_data = import_data(path, student.name)

    # Setting the laterality
    for motion in expert_data:
        motion.laterality = expert.laterality

    for motion in student_data:
        motion.laterality = student.laterality

    # List of datatypes and joint to process
    # default to check: {Descriptor: [{joint, laterality or not (check left for lefthanded and vice versa)},
    #                                  other joint, laterality or not}],
    #                    Other descriptor: [{joint, laterality or not}]
    #                   }
    datatype_joints_list = []

    datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                           {'joint': 'RightShoulder', 'laterality': False}]
                                }])

    datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                              {'joint': 'LeftShoulder', 'laterality': True}]
                                }])

    datatype_joints_list.append(['javelin', {'Distances': [{'joint': 'distanceRightHandHead', 'laterality': True}]
                                }])

    datatype_joints_list.append(['align_arm', {'BoundingBoxWidthMean': [{'joint': 'HeadRightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                               'BoundingBoxWidthStd': [{'joint': 'HeadRightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                              }])

    # datatype_joints_list.append(['javelin', {'PosX': [{'joint': 'LeftHand', 'laterality': True},
    #                                                   {'joint': 'Head',     'laterality': False}],
    #                                          'PosY': [{'joint': 'LeftHand', 'laterality': True},
    #                                                   {'joint': 'Head',     'laterality': False}],
    #                                          'PosZ': [{'joint': 'LeftHand', 'laterality': True},
    #                                                   {'joint': 'Head',     'laterality': False}]
    #                                         }])

    # datatype_joints_list.append(['align_arm', {'BoundingBoxMinusX': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
    #                                            'BoundingBoxPlusX':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
    #                                            'BoundingBoxMinusY': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
    #                                            'BoundingBoxPlusY':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
    #                                            'BoundingBoxMinusZ': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
    #                                            'BoundingBoxPlusZ':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
    #                                           }])



    # Scaling and normalisaing (nor not) the data
    # Useful for DBSCAN for example
    scale = False
    normalise = False

    expert_data_repartion = {'good': [x+1 for x in range(10)],
                     'leaning': [x+1 for x in range(10, 19)],
                     'javelin': [x+1 for x in range(20, 30)],
                     'align_arm': [x+1 for x in range(30, 40)],
                     'elbow_move': [x+1 for x in range(40, 50)]}

    # Algorithm(s) to use
    algos = {'k-means': {'n_clusters': 2}}

    clustering_problems = []

    for problem in datatype_joints_list:

        datatype_joints = problem[1]

        expert_sub_data = expert_data[:10] + expert_data[min(expert_data_repartion[problem[0]])-1:max(expert_data_repartion[problem[0]])]

        for algo, param in algos.items():
            print(problem[0])

            model = run_clustering(expert_sub_data, validate_data=False,
                                   datatype_joints=datatype_joints, algorithm=algo,
                                   parameters=param, scale_features=scale, normalise_features=normalise,
                                   true_labels=None, verbose=False, to_file=True, to_json=True, return_data=True)

        # Taking the student features
        std_features = data_gathering(student_data, datatype_joints)

        # If the expert data has been scaled, do the same for the student's ones
        if scale:
            scaler = RobustScaler()
            std_features = scaler.fit_transform(std_features)
        # Same for normalising
        if normalise:
            min_max_scaler = MinMaxScaler()
            std_features = min_max_scaler.fit_transform(std_features)

        # Compute the centroid of the student's features (Euclidean distance for now)
        std_centroid = get_centroid(std_features)

        # For the rest of the algorithm, if there are more than 2 features,
        # we run the data through a PCA for the next steps
        if len(model[0].cluster_centers_[0]) > 2:
            pca = PCA(n_components=2, copy=True)
            pca.fit(model[2])

            model[2] = pca.transform(model[2])
            model[0].cluster_centers_ = pca.transform(model[0].cluster_centers_)
            std_centroid = pca.transform(std_centroid.reshape(1, -1))[0]
            std_features = pca.transform(std_features)

        # Compute the distance from the student's centroid to the expert's ones
        distances_to_centroid = compute_distance(model[0].cluster_centers_, std_centroid)

        # Get the distance from the student's centroid to the line between the two expert's centroids
        distance_from_line = get_distance_from_expert_centoids_line(model[0].cluster_centers_, std_centroid)
        # Used to check if the student's centroid is between the expert centroids (diamond shape)
        distance_from_line /= (dst_pts(model[0].cluster_centers_[0], model[0].cluster_centers_[1]) / 2)

        # Normalise the distances to the centroids
        summ = sum(distances_to_centroid)
        for i, distance in enumerate(distances_to_centroid):
            distances_to_centroid[i] = distance/summ

        # Get the most probable cluster label for expert data
        clusters_label = get_cluster_label(expert_sub_data, expert_data_repartion, model[0].labels_)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #TODO: display numbers of motions in labelled clusters#
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        get_closeness_to_clusters(clusters_label, std_features, model[0].cluster_centers_, problem[0])

        # Display the closeness of the student's data to each expert cluster
        mix_res = mix(distances_to_centroid, clusters_label, distance_from_line)

        clusters_names = get_cluster_labels_from_data_repartition(model[0].labels_, model[0].cluster_centers_)

        clus_prob = ClusteringProblem(problem=problem[0],
                                      model=model[0], features=model[2],
                                      labels=model[0].labels_,
                                      centroids=model[0].cluster_centers_,
                                      sil_score=model[1]['ss'],
                                      clusters_names=clusters_names,
                                      algo_name=problem[0],
                                      std_data=std_features,
                                      std_centroid=get_centroid(std_features),
                                      distance_to_line=distance_from_line)

        clus_prob.trapezoid = get_trapezoid(model, std_centroid)
        clus_prob.circles = get_circle(model, std_centroid)
        clus_prob.std_data = std_features

        clustering_problems.append(clus_prob)

    return give_two_advices(clustering_problems)

def take_last_data(path, student, expert, number=9):

    if not student.full_name in os.listdir(path):
        print(f'ERROR: {student.full_name} not found in {path}.')
        return False

    mixed_path = os.path.normpath(os.path.join(path, 'mixed'))

    # Remove all non-expert data )
    for file in os.listdir(mixed_path):
        if expert.name not in file:
            print(f'Removed {file}')
            rmtree(os.path.normpath(os.path.join(mixed_path, file)))

    std_path = os.path.normpath(os.path.join(path, student.full_name))

    file_list = os.listdir(std_path)
    file_list = sorted(file_list, key=lambda sin: int(sin.replace(student.name + '_', '').replace('Char00', '')))
    folders_to_copy = file_list[-number:]

    for folder_to_copy in folders_to_copy:
        std_folder = os.path.join(std_path, folder_to_copy)
        print(f'Copying {folder_to_copy}')
        copy_tree(std_folder, os.path.join(mixed_path, folder_to_copy))

    return True

def take_specific_data(path, student, expert, begin=0, end=9, verbose=False, fullname=True):

    student_name_to_use = student.full_name
    if not fullname:
        student_name_to_use = student.name

    if not student_name_to_use in os.listdir(path):
        breakpoint()
        print(f'ERROR: {student_name_to_use} not found in {path}.')
        return False

    mixed_path = os.path.normpath(os.path.join(path, 'mixed'))

    # Remove all non-expert data
    for file in os.listdir(mixed_path):
        if expert.name not in file:
            if verbose:
                print(f'Removed {file}')
            rmtree(os.path.normpath(os.path.join(mixed_path, file)))

    std_path = os.path.normpath(os.path.join(path, student_name_to_use))

    file_list = os.listdir(std_path)
    file_list = sorted(file_list, key=lambda sin: int(sin.replace(student.name + '_', '').replace('Char00', '')))

    folders_to_copy = file_list[begin:end]

    for folder_to_copy in folders_to_copy:
        std_folder = os.path.join(std_path, folder_to_copy)
        if verbose:
            print(f'Copying {folder_to_copy}')
        copytree(std_folder, os.path.join(mixed_path, folder_to_copy))

    return True

def test_student_list():

    expert = Person(r'', 'aurel', 'Right')
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students/mixed'

    for student in cst.students_list:
        print(f'Testing {student.name}')

        folders_path = path
        if folders_path.split('/')[-1] == 'mixed':
            folders_path = "/".join(path.split('/')[:-1])

        if not take_specific_data(folders_path, student, expert, begin=0, end=1):
            print(f'ERROR AT {student.name}')
            return

        # Expert Data
        expert_data = import_data(path, expert.name)
        # Student data
        student_data = import_data(path, student.name)

        # Setting the laterality
        for motion in expert_data:
            motion.laterality = expert.laterality

        for motion in student_data:
            motion.laterality = student.laterality

        datatype_joints_list = []

        datatype_joints_list.append(['leaning', {'MeanSpeed': [{'joint': 'LeftShoulder', 'laterality': False},
                                                               {'joint': 'RightShoulder', 'laterality': False}]
                                    }])

        datatype_joints_list.append(['elbow_move', {'MeanSpeed': [{'joint': 'LeftArm', 'laterality': True},
                                                                  {'joint': 'LeftShoulder', 'laterality': True}]
                                    }])

        datatype_joints_list.append(['javelin', {'PosX': [{'joint': 'LeftHand', 'laterality': True},
                                                          {'joint': 'Head',     'laterality': False}],
                                                 'PosY': [{'joint': 'LeftHand', 'laterality': True},
                                                          {'joint': 'Head',     'laterality': False}],
                                                 'PosZ': [{'joint': 'LeftHand', 'laterality': True},
                                                          {'joint': 'Head',     'laterality': False}]
                                                }])

        datatype_joints_list.append(['align_arm', {'BoundingBoxMinusX': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                                   'BoundingBoxPlusX':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                                   'BoundingBoxMinusY': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                                   'BoundingBoxPlusY':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                                   'BoundingBoxMinusZ': [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}],
                                                   'BoundingBoxPlusZ':  [{'joint': 'RightShoulderRightArmRightForeArmRightHand', 'laterality': True}]
                                                  }])



        # Scaling and normalisaing (nor not) the data
        # Useful for DBSCAN for example
        scale = False
        normalise = False

        expert_data_repartion = {'good': [x+1 for x in range(10)],
                         'leaning': [x+1 for x in range(10, 19)],
                         'javelin': [x+1 for x in range(20, 30)],
                         'align_arm': [x+1 for x in range(30, 40)],
                         'elbow_move': [x+1 for x in range(40, 50)]}

        # Algorithm(s) to use
        algos = {'k-means': {'n_clusters': 2}}

        clustering_problems = []

        for problem in datatype_joints_list:
            std_features = data_gathering(student_data, problem[1])

def do_all():
    expert = Person(r'', 'aurel', 'Right')
    path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/all_good_students/mixed'
    for student in cst.students_list:
        only_feedback_new_descriptors(expert, student, path)

def redo_gr1():
    expert = Person(r'', 'aurel', 'Right')
    std_list = [ Person(r'', 'DoneauR', 'Right', 'Doneau_Rafael'),
                 Person(r'', 'CorgniardA', 'Right', 'Corgniard_Antoine'),
                 Person(r'', 'AubertJ', 'Right', 'Aubert_Julian'),
                 Person(r'', 'BrouardS', 'Right', 'Brouard_Samuel'),
                 Person(r'', 'BlanchardA', 'Right', 'Blanchard_Axel'),
                 Person(r'', 'KherratiY', 'Right', 'Kherrati_Yazid'),
                 Person(r'', 'BrunetL', 'Right', 'Brunet_Leo'),
                 Person(r'', 'EuvrardL', 'Right', 'Euvrard_Louis'),
                 Person(r'', 'RouxelV', 'Right', 'Rouxel_Valentin'),
                 Person(r'', 'DelhommaisT', 'Right', 'Delhommais_Tony'),
                 Person(r'', 'DizelC', 'Right', 'Dizel_Corentin'),
                 Person(r'', 'HuetL', 'Right', 'Huet_Loic'),
                 Person(r'', 'BouligandC', 'Right', 'Bouligand_Colin'),
                 Person(r'', 'JonardA', 'Right', 'Jonard_Antoine'),
                 Person(r'', 'LaghouaoutaY', 'Right', 'Laghouaouta_Youness')]
    for student in std_list:
        print(student.full_name)
        for i in range(4):
            print(f'Jeu {i}')
            print(f'{i*9} - {(i*9)+9}')
            compute_distance_to_clusters(expert,
                                         student,
                                         r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/redo_students_gr1/descriptors/',
                                         0,
                                         36,
                                         fullname=False)

def redo_all():
    expert = Person(r'', 'aurel', 'Right')
    for student in cst.new_students_list:
        print(student.full_name)
        compute_distance_to_clusters(expert,
                                        student,
                                        r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/redo_all_students/',
                                        0,
                                        36,
                                        fullname=False)

if __name__ == '__main__':
    expert = Person(r'', 'aurel', 'Right')
    student = Person(r'', 'RannouP', 'Right', 'Pierre_Rannou')
    #path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/visitelabo/mixed'
    #path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/students_2/mixed'
    path = r'C:/Nathan/Travail IUT/STAGE/mixed'
    #path = r'C:/Users/quentin/Documents/Programmation/C++/MLA/Data/alldartsdescriptors/test/noneed_rotated/mixed'
    #only_feedback(expert, student, path)
    redo_all()
    #merge_same_names_xlsx()

    only_feedback_new_descriptors(expert, student, path)

    # export_advices_to_xlsx(path)
    # one_sheet_xlsx()
