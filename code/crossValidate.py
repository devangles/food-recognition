import numpy as np
from scipy import interp
import scipy.io
import pylab as pl

from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.cross_validation import StratifiedKFold
from labels import *
import pickle
import multiprocessing
from functools import partial
from sklearn.grid_search import GridSearchCV
import time

###############################################################################
# Attribute Classifier
def trainAttributeClassifiers(ingred,trainlabels,Recipes,fooddish,traindata):
    #find positive examples of attribute i 
    attr_labels = np.zeros((len(trainlabels),),dtype=np.uint8) 
    for j in xrange(len(fooddish)):
	if ingred in Recipes[fooddish[j]]:
	    pos_idx = np.where(trainlabels==j)
	    attr_labels[pos_idx[0]] = 1
    
    #train classifier
    #add some fake negative labels if only one class exists
    #print ingred, sum(attr_labels)
    if sum(attr_labels) == traindata.shape[0]:
	attr_labels[range(0,traindata.shape[0],100)] = 0
    elif sum(attr_labels) == 0:
	attr_labels[range(0,traindata.shape[0],100)] = 1

    #split training data into k-folds
    kfold = StratifiedKFold(attr_labels,2)
    param_grid = [
      {'C': [0.001, 0.01, 1, 10, 100]}#, 'gamma': [0.0], 'kernel': ['linear'], 'degree':[3]},  
      #{'C': [0.001, 0.01, 1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf'], 'degree':[3]},
     ]
    
    cv = GridSearchCV(estimator=svm.LinearSVC(class_weight='auto'), param_grid=param_grid, cv=kfold)
    #cv=svm.SVC(class_weight='auto')
    cv.fit(traindata,attr_labels)

    #attributeclassifier = svm.SVC(C=cv.best_params_['C'], kernel=cv.best_params_['kernel'], gamma = cv.best_params_['gamma'], degree=cv.best_params_['degree'], class_weight='auto') 
    attributeclassifier = svm.LinearSVC(C=cv.best_params_['C'], class_weight='auto')
    #attributeclassifier = svm.SVC(kernel='rbf',C=1,gamma=0.001, class_weight='auto')
    attributeclassifier.fit(traindata,attr_labels)
    return attributeclassifier



##############################################################################
if __name__=="__main__":
# Data IO 

    # import data
    dataset = "vlg_extractor/ImageNetSurveyMC/ImageNetSurveyMC"
    var=scipy.io.loadmat(dataset)
    traindata = var['X'].astype(np.float32)
    trainlabels = var['trainlabels'].flatten().astype(int)
    testdata = var['Xtest'].astype(np.float32)
    testlabels = var['testlabels'].flatten()
    X = np.concatenate((traindata,testdata),0)
    y = np.concatenate((trainlabels,testlabels),0)
    n_samples, n_features = X.shape
    del traindata, testdata, var
    
    recipedict = recipeDict[0]
    fooddish = fooddish[0]	
    I,R = pickle.load(file(recipedict,'r'))
    ingsorted = sorted(I.keys())[1:]
	    
    Recipes = {}
    for dish in fooddish:
	recipes = R[dish].values()
	ing = [r.keys() for r in recipes]
	Recipes[dish] = set().union(*ing)
	    
    
    
    ###############################################################################
    # Classification and ROC analysis
    
    # Run classifier with cross-validation and plot ROC curves
    scores = np.zeros((2,5))
    scores2 = np.zeros((2,5))
    cv = StratifiedKFold(y, n_folds=5)
    #classifier = svm.SVC(kernel='rbf', C=10, gamma=0.001, probability=True, random_state=0)
    
    
    mean_tpr = np.zeros((len(fooddish),100))
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    mean_tpr2 = np.zeros((len(fooddish),100))
    mean_fpr2 = np.linspace(0, 1, 100)
    all_tpr2 = []
    
    
    for k, (train, test) in enumerate(cv):
	print k, time.asctime(time.localtime(time.time()))
	classifier = svm.LinearSVC(C=1)
	classifier2 = svm.LinearSVC(C=0.001)	
	
	classifier.fit(X[train], y[train])
	scores[0,k]=accuracy_score(y[test],classifier.predict(X[test]))
	scores[1,k]=f1_score(y[test],classifier.predict(X[test]))
	#probas_ = classifier.predict_proba(X[test])
	del classifier
	print "BL fitted", time.asctime(time.localtime(time.time()))
	
	#predict attributes
	pool = multiprocessing.Pool()
	AttributeClassifier = partial(trainAttributeClassifiers,trainlabels=y[train],Recipes=Recipes,fooddish=fooddish, traindata=X[train])
	attributeclassifiers= pool.map(AttributeClassifier,ingsorted)
	pool.close()
	pool.join()    
	
	print "Attribute classifiers trained", time.asctime(time.localtime(time.time()))
	
	IngredientsTest = np.zeros((len(y[test]),len(ingsorted)),dtype=np.float32)
	for i in xrange(len(attributeclassifiers)):
	    #print 'ingred:', i
	    IngredientsTest[:,i] = attributeclassifiers[i].predict(X[test])    
	XExtraTest = np.concatenate((X[test],IngredientsTest),1)
	pickle.dump(IngredientsTest,file("/".join(dataset.split('/')[0:2])+'/IngredientAttributes'+str(k)+'.npy','w'))   
	del attributeclassifiers
	
	IngredientsTrain = np.zeros((len(y[train]),len(ingsorted)),dtype=np.float32)
	thresh=0.25
	for i in xrange(len(y[train])):
	    dish = fooddish[y[i]]
	    IngredientsTrain[i,:] = [1 if ing in Recipes[dish] and prob > thresh else 0 for ing,prob in zip(ingsorted,np.random.random((len(ingsorted),)))] 
	
	XExtraTrain = np.concatenate((X[train],IngredientsTrain),1)    
	
	classifier2.fit(XExtraTrain,y[train])
	scores2[0,k] = accuracy_score(y[test],classifier2.predict(XExtraTest))
	scores2[1,k] = f1_score(y[test],classifier2.predict(XExtraTest))
	#probas2_ = classifier2.predict_proba(XExtraTest)
	print "Enhanced classifier trained", time.asctime(time.localtime(time.time()))
	del classifier2, 
	
	# Compute ROC curve and area the curve for each class
	#for j in xrange(len(fooddish)):
	    #fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1],j)
	    #mean_tpr[j,:] += interp(mean_fpr, fpr, tpr)
	    #mean_tpr[j,0] = 0.0
	    ##roc_auc = auc(fpr, tpr)
	    ##pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
	    
	    ##do the same for extra attributes
	    #fpr2, tpr2, thresholds = roc_curve(y[test], probas2_[:,1],j)
	    #mean_tpr2[j,:] += interp(mean_fpr2,fpr2,tpr2)
	    #mean_tpr2[j,0] = 0.0
	    #roc_auc2 = auc(fpr2,tpr2)
    
    f = file(dataset+"-CVScores.txt",'w')
    f.write("Method\tAccuracy\tF1\n")
    f.write("BL\t %0.3f (+/-%0.03f)\t" % (scores[0,:].mean(), scores[0,:].std()/2))
    f.write("%0.3f (+/-%0.03f)\n" % (scores[1,:].mean(), scores[1,:].std()/2))
    f.write("Attribute\t %0.3f (+/-%0.03f)\t" % (scores2[0,:].mean(), scores[0,:].std()/2))
    f.write("%0.3f (+/-%0.03f)\n" % (scores2[1,:].mean(), scores[1,:].std()/2))    
    f.close()
    
    #plot ROC curves for each class
    #mean_tpr /= len(cv)
    #mean_tpr[:,-1] = 1.0   
    #mean_tpr2 /= len(cv)
    #mean_tpr2[:,-1] = 1.0    
    #for i in xrange(len(fooddish)):
    
	#pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

	#mean_auc = auc(mean_fpr, mean_tpr[i,:])
	#pl.plot(mean_fpr, mean_tpr[i,:], 'b--',
	        #label='BL (AUC = %0.2f)' % mean_auc, lw=2)
	
	
	#mean_auc2 = auc(mean_fpr2, mean_tpr2[i,:])
	#pl.plot(mean_fpr2, mean_tpr2[i,:], 'r-',
	        #label='Attributes (AUC = %0.2f)' % mean_auc2, lw=2)
	
	#pl.xlim([-0.05, 1.05])
	#pl.ylim([-0.05, 1.05])
	#pl.xlabel('False Positive Rate')
	#pl.ylabel('True Positive Rate')
	#pl.title('%s ROC' % fooddish[i])
	#pl.legend(loc="lower right")
	##pl.show()
	#pl.savefig("/".join(dataset.split('/')[0:2])+'/ROC_%s.png' % fooddish[i], format='png',dpi=250) 
	#pl.close("all")