# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/random_toolbox_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_RandomToolboxMainWindow(object):
    def setupUi(self, RandomToolboxMainWindow):
        RandomToolboxMainWindow.setObjectName("RandomToolboxMainWindow")
        RandomToolboxMainWindow.resize(1398, 949)
        self.centralwidget = QtWidgets.QWidget(RandomToolboxMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(-1, 9, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_OpenFile = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_OpenFile.setObjectName("pushButton_OpenFile")
        self.verticalLayout_2.addWidget(self.pushButton_OpenFile)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.spinBox_BitWidth = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_BitWidth.setMinimum(1)
        self.spinBox_BitWidth.setMaximum(64)
        self.spinBox_BitWidth.setProperty("value", 8)
        self.spinBox_BitWidth.setObjectName("spinBox_BitWidth")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox_BitWidth)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_PlotPsd = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_PlotPsd.setObjectName("pushButton_PlotPsd")
        self.gridLayout.addWidget(self.pushButton_PlotPsd, 1, 1, 1, 1)
        self.pushButton_PlotAcf = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_PlotAcf.setObjectName("pushButton_PlotAcf")
        self.gridLayout.addWidget(self.pushButton_PlotAcf, 1, 0, 1, 1)
        self.pushButton_PlotHistogram = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_PlotHistogram.setObjectName("pushButton_PlotHistogram")
        self.gridLayout.addWidget(self.pushButton_PlotHistogram, 0, 1, 1, 1)
        self.pushButton_PlotTimeSeries = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_PlotTimeSeries.setObjectName("pushButton_PlotTimeSeries")
        self.gridLayout.addWidget(self.pushButton_PlotTimeSeries, 0, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.horizontalLayout.addWidget(self.groupBox)
        RandomToolboxMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RandomToolboxMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1398, 20))
        self.menubar.setObjectName("menubar")
        RandomToolboxMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RandomToolboxMainWindow)
        self.statusbar.setObjectName("statusbar")
        RandomToolboxMainWindow.setStatusBar(self.statusbar)
        self.label.setBuddy(self.spinBox_BitWidth)

        self.retranslateUi(RandomToolboxMainWindow)
        QtCore.QMetaObject.connectSlotsByName(RandomToolboxMainWindow)

    def retranslateUi(self, RandomToolboxMainWindow):
        _translate = QtCore.QCoreApplication.translate
        RandomToolboxMainWindow.setWindowTitle(_translate("RandomToolboxMainWindow", "Random Toolbox"))
        self.groupBox.setTitle(_translate("RandomToolboxMainWindow", "Plot"))
        self.pushButton_OpenFile.setText(_translate("RandomToolboxMainWindow", "Open File"))
        self.label.setText(_translate("RandomToolboxMainWindow", "Bit Width"))
        self.pushButton_PlotPsd.setText(_translate("RandomToolboxMainWindow", "Plot PSD"))
        self.pushButton_PlotAcf.setText(_translate("RandomToolboxMainWindow", "Plot ACF"))
        self.pushButton_PlotHistogram.setText(_translate("RandomToolboxMainWindow", "Plot Histogram"))
        self.pushButton_PlotTimeSeries.setText(_translate("RandomToolboxMainWindow", "Plot Time Series"))
