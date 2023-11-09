# QGIS-Geographical-detector

QGIS geographical detector plugin: Tool for measuring spatial stratified heterogeneity and spatial associations of geographical attributes.


## Geographical detector
Geographical detector, or GeoDetector, is a statistical tool to measure Spatial Stratified Heterogeneity (SSH) and to make attribution for/by SSH; 
(1) measure and find SSH among data;
(2) test the coupling between two variables Y and X and 
(3) investigate interaction between two explanatory variables X1 and X2 to a response variable Y.
## Installation

![Menus and procedure for one-time activation of the Geographical detector plugin within QGIS](image/Snipaste_2022-04-01_12-18-20.png)

## Parameters
![Q_GD GUI](https://github.com/gsnrguo/QGIS-Geographical-detector/blob/main/image/Q_GD%20GUI.png)
### Basic parameters
1. Input layer: vector layer 
2. Study variable : field name of study variable 
3. Field(s) with categories [optional]: field(s) of categories explanatory variables
4. Field(s) with numeric [optional]: field(s) of  numeric explanatory variables 
  
  *Parameters 3 and 4 cannot both be empty, if parameter 4 is not empty, then a stratification procedure is required.*

### Advanced parameters (stratification parameters)

1. Maximum number of groups [optional]: Maximum number of strata, if the Maximum number is equal to 
2. Minimum number of groups [optional]: 
3. Field for equality constraint [optional]: equality means the populations/geographical areas in the new strata are of sufficient size and as similar as possible
4. Minimum ratio for equality measures [optional]: restrict the minimum population/geographical area in each strata
5. The number of samples for stratification [optional]: if the number of vector is too large, sampling is 
6. Minimum threshold for q-value increase [optional]: will be deprecated in the new version
7. Cross-validation number: default is 10
8. Cross-validation random stata [optional]: random seed, if a random seed is given so that the results of the running are reproducible
9. Times of repeating cross-validation [optional] : Improving the stability of stratification results.

### Author

  Jiangang Guo (<guojg@lreis.ac.cn>); Jinfeng Wang (<wangjf@lreis.ac.cn>)

## References
1. Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X & Zheng XY. 2010. Geographical detectors-based health risk assessment and its application in the neural tube defects study of the Heshun region, China. International Journal of Geographical Information Science 24(1): 107-127.
2. Wang JF, Zhang TL, Fu BJ. 2016. A measure of spatial stratified heterogeneity. Ecological Indicators 67: 250-256.
3. GeoDetector Website (http://www.geodetector.cn/)


