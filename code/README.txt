1) RawImages: contains images of the 50Food (Chen et al.) and ImageNet115 (Deng et al.) data sets

2) BoFFeatureExtraction: Code to extract BoF features used in report
	Note: requires vl_feat toolbox from vlfeat.org
	
3) Data: contains pre-processed images in .mat files.

4) ingredientsScraper.py: Python script to scrape AllRecipes.com for recipe data

5) crossValidate.py: Python script to train baseline classifier, attribute classifiers
	and enhanced baseline classifier
	Note: requires sci-kit learn library and scraped recipe data using
	ingredientsScraper.py


6) labels.py: contains class labels required by crossValidate.py and ingredientsScraper.py



Note: Code to extract PiCoDes and Meta-Class descriptors available from:
	vlg.cs.dartmouth.edu/picodes


References:
[1] Alessandro Bergamo, Lorenzo Torresani, Andrew Fitzgibbon
      	PiCoDes: Learning a Compact Code for Novel-Category Recognition, NIPS 2011
[2] Alessandro Bergamo, Lorenzo Torresani
      	Meta-Class Features for Large-Scale Object Categorization on a Budget
[3] Mei-Yun Chen, Yung-Hsiang Yang, Chia-Ju Ho, Shih-Han Wang, Shane-Ming Liu, Eugene Chang, Che-Hua Yeh, Ming Ouhyoung. 
	Automatic Chinese food identification and quantity estimation. In SIGGRAPH Asia 2011 Technical Briefs (SA 
[4] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, 
	ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.
[5] A. Vedaldi and B. Fulkerson
 	VLFeat: An Open and Portable Library of Computer Vision Algorithms, 2008
[6] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
