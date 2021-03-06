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
import inspect
from qgis.PyQt.QtGui import QIcon

from qgis.core import QgsProcessingProvider

from .geographical_detector_algorithm import Geo_detectorAlgorithm


class Geo_detectorProvider(QgsProcessingProvider):

    def __init__(self):
        """
        Default constructor.
        """
        QgsProcessingProvider.__init__(self)

    def unload(self):
        """
        Unloads the provider. Any tear-down steps required by the provider
        should be implemented here.
        """
        pass

    def loadAlgorithms(self):
        """
        Loads all algorithms belonging to this provider.
        """
        self.addAlgorithm(Geo_detectorAlgorithm())
        # self.addAlgorithm(Stratification_gd())
        # self.addAlgorithm(RandomPointsPolygons())
        # Stratification_gd

    def id(self):
        """
        Returns the unique provider id, used for identifying the provider. This
        string should be a unique, short, character only string, eg "qgis" or
        "gdal". This string should not be localised.
        """
        return 'Geographical detector'

    def name(self):
        """
        Returns the provider name, which is used to describe the provider
        within the GUI.

        This string should be short (e.g. "Lastools") and localised.
        """
        return self.tr('Geographical detector')

    def icon(self):
        """
        Should return a QIcon which is used for your provider inside
        the Processing toolbox.
        """
        # return QgsProcessingProvider.icon(self) default icon
        cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
        icon = QIcon(os.path.join(os.path.join(cmd_folder, 'icon.png')))
        return icon

    def longName(self):
        """
        Returns the a longer version of the provider name, which can include
        extra details such as version numbers. E.g. "Lastools LIDAR tools
        (version 2.2.1)". This string should be localised. The default
        implementation returns the same string as name().
        """
        return self.name()
