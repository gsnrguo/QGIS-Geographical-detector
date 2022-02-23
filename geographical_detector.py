# -*- coding: utf-8 -*-

"""
/***************************************************************************
 Geo_detector
                                 A QGIS plugin
 This plugin adds an algorithm to measure the spatial stratified heter
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-12-21
        copyright            : (C) 2021 by Guojg
        email                : guojg@lreis.ac.cn
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Guojg'
__date__ = '2021-12-21'
__copyright__ = '(C) 2021 by Guojg'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

import os
import sys
import inspect

from qgis import processing
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon

from qgis.core import QgsApplication
from .geographical_detector_provider import Geo_detectorProvider

cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)


class Geo_detectorPlugin(object):

    def __init__(self, iface):
        self.provider = None
        self.iface = iface

    def initProcessing(self):
        """Init Processing provider for QGIS >= 3.8."""
        self.provider = Geo_detectorProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        icon = os.path.join(os.path.join(cmd_folder, 'icon.png'))
        self.action = QAction(
            QIcon(icon),
            u"Geographic detector", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(u"&Geographic detector", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        # We will also need to add code to the unload method, to remove these elements when plugin is removed.
        QgsApplication.processingRegistry().removeProvider(self.provider)
        self.iface.removePluginMenu(u"&Geographic detector", self.action)
        self.iface.removeToolBarIcon(self.action)

    def run(self):
        processing.execAlgorithmDialog("Geographical detector")
