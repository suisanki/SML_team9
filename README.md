# SML_team9
Project Title: Detection of Counterfeit Banknote with ML

Brief description: Train various ML algorithms with image feature data of real/counterfeit money to observe efficiency of each algoithm in this issue and build a classfier. If then, make another classifier with real data using same method.

Data set: Banknote Authentication Dataset from UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/267/banknote+authentication)
(citation:Lohweg,Volker. (2013). Banknote Authentication. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57)

- 1,372 data points
- 5 variables (variance of Wavelet Transformed image, skewness of Wavelet Transformed image, curtosis of Wavelet Transformed image, entropy of image, real/counterfeit flag)
- No missing value
- Does not specify the kind of banknote

# Codes
1. Wavelet.py: Used to apply wavelet transform on pictures
2. visualization_final.ipynb: Used to visualize raw data
3. modelSelection_final.ipynb: Used to train and evaluate models
