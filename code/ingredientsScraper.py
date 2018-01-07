from bs4 import BeautifulSoup as bs
import urllib2
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random as rnd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from labels import *
import sys

def IngredientScraper2(fooddish):
    #dictionary for ingredients
    I = {}
    #dictionary for food recipes
    R = {}
      
    website = 'http://allrecipes.com'
    for food in fooddish:
        R[food] = {}
        #search for food
        print food
	
	#for page in xrange(2):
	resultspage = urllib2.urlopen("http://allrecipes.com/search/default.aspx?qt=k&wt="+food)
	results = bs(resultspage)
	for recipelinks in results.find_all('a',class_='title'):
	    recipelink = recipelinks.get('href')
	    #go to recipe page
	    recipepage = urllib2.urlopen(website+recipelink)
	    recipe = bs(recipepage)
	    recipename = recipe.find('h1',id='itemTitle').text
	    
	    if recipename not in R[food]:
		#print "Recipe: ", recipename
		#ingredients for this recipe
		ingredients = recipe.find_all('li', id='liIngredient')
		R[food][recipename] = {}
		for ing in ingredients:
		    ingid = ing.attrs['data-ingredientid']
		    ingname = ing.find(id='lblIngName').text 
		    if ingid not in I:
			I[ingid] = ingname
		    amt=float(ing.attrs['data-grams'])
		    R[food][recipename][ingid] = amt

		#normalize values
		m = sum(R[food][recipename].values())
		R[food][recipename]={ingid: R[food][recipename][ingid]/m for ingid in R[food][recipename].keys()}

        
    
    #Recipes = {}
    #ingsorted = sorted(I.keys())
    #for food in R.keys():
        ##m = sum(R[food].values())
        ##normalize values
        ##R[food] = {ingid: R[food][ingid]/m for ingid in R[food].keys()}
        #Recipes[food] = [0]*len(ingsorted)
        #for i in range(len(ingsorted)):
            ###if ingredient is in dish R[food]
            #if ingsorted[i] in R[food]:
                #Recipes[food][i] = R[food][ingsorted[i]]
	#m = sum(Recipes[food])
	#Recipes[food] = [x/m for x in Recipes[food]]
    pickle.dump((I,R),file('AllRecipesIngImageNet.npy','w'))        
    #return I,R
#=================================================================================
# Ingredient Scraper with cooking terms and nutritional info

def IngredientScraper(fooddish):
    #dictionary for ingredients
    I = {}
    #dictionary for food recipes
    R = {}
      
    website = 'http://allrecipes.com'
    for food in fooddish:
        R[food] = {}
        #search for food
        print food
	
	#for page in xrange(2):
	resultspage = urllib2.urlopen("http://allrecipes.com/search/default.aspx?qt=k&wt="+food)
	results = bs(resultspage)
	for recipelinks in results.find_all('a',class_='title'):
	    recipelink = recipelinks.get('href')
	    #go to recipe page
	    recipepage = urllib2.urlopen(website+recipelink)
	    recipe = bs(recipepage)
	    recipename = recipe.find('h1',id='itemTitle').text
	    
	    if recipename not in R[food]:
		#print "Recipe: ", recipename
		#ingredients for this recipe
		ingredients = recipe.find_all('li', id='liIngredient')
		#list containing ingredients, cookingterms, nutritionrating
		R[food][recipename] = [{},[],[0]*7]
		for ing in ingredients:
		    ingid = ing.attrs['data-ingredientid']
		    ingname = ing.find(id='lblIngName').text 
		    if ingid not in I:
			I[ingid] = ingname
		    amt=float(ing.attrs['data-grams'])
		    R[food][recipename][0][ingid] = amt

		#normalize values
		m = sum(R[food][recipename][0].values())
		R[food][recipename][0]={ingid: R[food][recipename][0][ingid]/m for ingid in R[food][recipename][0].keys()}
		
		#get cooking terms
		directions = [step.text.lower() for step in recipe.find_all('span', class_='plaincharacterwrap break')]
		R[food][recipename][1] = directions
		#get nutrition
		nutritionrating = recipe.find_all('ul', id='ulNutrient')
		n = 0
		for nutrient in nutritionrating:
		    #category = nutrient.find('li',class_='categories').text
		    R[food][recipename][2][n]=float(nutrient.find('li',id='divNutrientGradient').attrs['style'][6:-1])/100
		    n += 1

        
    
    pickle.dump((I,R),file('AllRecipesIng50FoodExtra.npy','w'))        






#================================================================================

#X = np.zeros((len(trainlabels),len(I.keys())-1),dtype=np.float32)
#ingsorted = sorted(I.keys())[1:]
#for i in xrange(len(trainlabels)):
    ##thresh = np.random.uniform(0,RecipeMax[trainlabels[i]],n)
    #dish = fooddish[trainlabels[i]]
    #X[i,:] = [1 if x != 0 else 0 for x in Recipes[dish][1:]]
    ##if len(R[dish].keys()) != 0:
	###randomly pick recipe
	##recipe = rnd.choice(R[dish].keys())
	##print recipe
	##for j in xrange(len(ingsorted)):
	    ##if ingsorted[j] in R[dish][recipe]:
		###X[i,j] = R[dish][recipe][ingsorted[j]]
		##X[i,j] = 1
	
    ###Recipes[food] = [0]*len(ingsorted)
            ###for i in range(len(ingsorted)):
                ####if ingredient is in dish R[food]
                ###if ingsorted[i] in R[food]:
                    ###Recipes[food][i] = R[food][ingsorted[i]]     
    ###X[i,:] = [1 if x>t else 0 for x,t in zip(Recipes[dish],thresh)]
    ###X[i,:] = Recipes[dish]

##train classifier for each ingredient attribute
#attributeclassifiers = [None]*len(ingsorted)
#for i in xrange(len(ingsorted)):
    ##find positive examples of attribute i
    #pos_idx = np.where(X[:,i]==1)
    #print i, len(pos_idx[0])
    #attr_labels = np.zeros((len(trainlabels),),dtype=np.uint8)
    #attr_labels[pos_idx[0]] = 1
    
    ##train classifier
    #if len(pos_idx[0]) == traindata.shape[0]:
	#attr_labels[range(0,800,100)] = 0
    #attributeclassifiers[i] = svm.SVC(kernel='linear',C=0.001)
    #attributeclassifiers[i].fit(traindata,attr_labels)    

#Xtest = np.zeros((len(testlabels),len(I.keys())-1),dtype=np.float32)
#for i in xrange(len(testlabels)):
    #print 'test case:', i
    #Xtest[i,:] = [x.predict(testdata[i,:])[0] for x in attributeclassifiers]
 
#pickle.dump((X,Xtest),file('vlg_extractor_1.1.2/ImageNetSurveyMC/IngredientAttributes.npy','w'))
###fill out correlation matrix
#m = traindata.shape[1] #number of visual word
#n = len(I.keys()) #number of ingredients
#corr_mat = np.zeros((m,n))
#for i in xrange(len(trainlabels)):
    #for visualword in xrange(m):
        #if traindata[i,visualword] != 0:
            ##count co-occurrence of ingredient and visual word
            ##binaryIng = [1 if x!=0 else 0 for x in Recipes[fooddish[trainlabels[i]]]]
            #corr_mat[visualword,:] = corr_mat[visualword,:] + X[i,:]

#pickle.dump(corr_mat,file('corr_mat50Food.npy','w'))
###traindata = np.concatenate((traindata,X),1)
##corr_mat = pickle.load(file('corr_mat.npy','r'))
                
###normalize corr_mat
#row_sums = corr_mat.sum(axis=1)
#row_sums = np.array([1 if x==0 else x for x in row_sums])
#corr_mat = corr_mat/row_sums[:,np.newaxis]
##avg = corr_mat.mean(axis=0)

#logcormat = np.log(corr_mat+1)
#Xtest = np.zeros((len(testlabels),len(I.keys())),dtype=np.float32)
#for i in xrange(len(testlabels)):
    #x = np.dot(testdata[i,:],logcormat)
    #Xtest[i,:] = x/sum(x)
    ##dish = fooddish[testlabels[i]]
    ###randomly pick recipe
    ##recipe = rnd.choice(R[dish].keys())
    ##print recipe
    #for j in xrange(len(ingsorted)):
	#if attributeclassifiers[j] is not None:
	    #Xtest[i,j]=attributeclassifiers[j].predict(testdata[i,:])
	##if ingsorted[j] in R[dish][recipe]:
	    ##Xtest[i,j] = 1    
    ##Xtest[i,:] = [1 if xt>t else 0 for xt,t in zip(x,avg)]

#fig = plt.figure()
#ax = fig.add_subplot(5,2,10)
#count = [0]*len(ingsorted)
#for i in xrange(len(ingsorted)):
    ##find negative examples of attribute i
    #pos_idx = np.where(X[np.where(trainlabels==9)[0],i]==1)
    #count[i] = len(pos_idx[0])
#r = plt.bar(range(589),count)
#ax.set_xticks([])
#plt.xlabel(fooddish[9])
##ax = fig.add_subplot(522)
##r = plt.bar(range(440),Recipes['casserole'])
##ax.set_xticks([])
##plt.xlabel('casserole')
##ax = fig.add_subplot(523)
##r = plt.bar(range(440),Recipes['deviled%eggs'])
##ax.set_xticks([])
##plt.xlabel('deviledegg')
##ax = fig.add_subplot(524)
##r = plt.bar(range(440),Recipes['fried%rice'])
##ax.set_xticks([])
##plt.xlabel('friedrice')
##ax = fig.add_subplot(525)
##r = plt.bar(range(440),Recipes['kebab'])
##ax.set_xticks([])
##plt.xlabel('kebab')
##ax = fig.add_subplot(526)
##r = plt.bar(range(440),Recipes['samosa'])
##ax.set_xticks([])
##plt.xlabel('samosa')
##ax = fig.add_subplot(527)
##r = plt.bar(range(440),Recipes['pasta%salad'])
##ax.set_xticks([])
##plt.xlabel('pastasalad')
##ax = fig.add_subplot(528)
##r = plt.bar(range(440),Recipes['paella'])
##ax.set_xticks([])
##plt.xlabel('Paella')
##ax = fig.add_subplot(529)
##r = plt.bar(range(440),Recipes['spaghetti'])
##ax.set_xticks([])
##plt.xlabel('spaghetti')
##ax = fig.add_subplot(5,2,10)
##r = plt.bar(range(440),Recipes['roulade'])
##ax.set_xticks([])
##plt.xlabel('roulade')

#============== script to get top features ============================
#from sklearn.multiclass import OneVsRestClassifier
#import random as rnd
#recipedict='AllRecipesIng.npy'
#fooddish = fooddish[0]
#dataset = 'vlg_extractor/ImageNetSurveyMC/ImageNetSurveyMC'
#var=scipy.io.loadmat(dataset)
#traindata = np.ndarray.astype(var['X'],dtype=np.float32)
#trainlabels = np.ndarray.astype(var['trainlabels'].flatten(),dtype=np.int)
#testdata = np.ndarray.astype(var['Xtest'],dtype=np.float32)
#testlabels = var['testlabels'].flatten()


#Xtest = pickle.load(file("/".join(dataset.split('/')[0:2])+'/IngredientAttributes.npy','r'))
#I,R = pickle.load(file(recipedict,'r'))
#ingsorted = sorted(I.keys())[1:]
#X = np.zeros((len(trainlabels),len(ingsorted)),dtype=np.uint8)    
#for i in xrange(len(trainlabels)):
    #dish = fooddish[trainlabels[i]]
    #if len(R[dish].keys()) != 0:
	####randomly pick recipe
	#recipe = rnd.choice(R[dish].keys())
	##print recipe
	#X[i,:] = [1 if ing in R[dish][recipe] else 0 for ing in ingsorted]

#k=5
##split training data into k-folds
#kfold = cross_validation.StratifiedKFold(trainlabels,k)
#param_grid = [
          #{'estimator__C': [0.001, 0.01, 1, 10, 100], 'estimator__kernel': ['linear']},  
          ##{'estimator__C': [1, 10, 100, 1000], 'estimator__gamma': [0.01, 0.001, 0.0001], 'estimator__kernel': ['rbf']},
         #]

#svc = OneVsRestClassifier(svm.SVC(kernel='linear',C=1))
#svc.fit(X,trainlabels)
##clf = GridSearchCV(estimator=svc, param_grid=param_grid, cv=kfold, n_jobs=-1)
##clf.fit(np.concatenate((traindata,X),1),trainlabels)


#svm_weights = svc.coef_
#topfeatures = [None]*svm_weights.shape[0] #topfeatures for each class
#for i in xrange(svm_weights.shape[0]):
    #featureIdx=np.argsort(abs(svm_weights[i,:]))
    #topfeatures[i] = featureIdx[::-1][0:30] #get top 30

##allfeatures = sorted(list(set().union(*topfeatures)))

###print top features for each class
#for f in xrange(len(fooddish)):
    #xlabels = [None]*30
    #for ingIdx in xrange(30):
	#print fooddish[f], I[ingsorted[topfeatures[f][ingIdx]]], svm_weights[f,topfeatures[f][ingIdx]]
	#xlabels[ingIdx] = I[ingsorted[topfeatures[f][ingIdx]]]
    #fig=plt.figure()
    #ax = fig.add_subplot(111)
    #r = plt.bar(range(30),svm_weights[f,topfeatures[f]],color='b')
    #ax.set_xticks(np.arange(30)+0.5)
    #ax.set_xticklabels(xlabels,rotation=90,fontsize=8)
    #ax.set_title(fooddish[f])
    #ax.set_ylabel('Feature Weights')
    #plt.show()

#=============================END ==================================
##train classifier for each ingredient attribute
#attributeclassifiers = [None]*len(allfeatures)
#param_grid = [
          #{'C': [0.001, 0.01, 1, 10, 100], 'kernel': ['linear']},  
          ##{'estimator__C': [1, 10, 100, 1000], 'estimator__gamma': [0.01, 0.001, 0.0001], 'estimator__kernel': ['rbf']},
         #]
#for i in xrange(len(allfeatures)):
    ##find positive examples of attribute i
    #pos_idx = np.where(X[:,allfeatures[i]]==1)
    #print I[ingsorted[allfeatures[i]]], len(pos_idx[0])
    #attr_labels = np.zeros((len(trainlabels),),dtype=np.uint8)
    #attr_labels[pos_idx[0]] = 1
    
    ##train classifier
    #if len(pos_idx[0]) != 0:
	#attributeclassifiers[i] = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, cv=kfold, n_jobs=-1)
	#attributeclassifiers[i].fit(traindata,attr_labels)

#Xtest = np.zeros((len(testlabels),len(I.keys())),dtype=np.float32)
#for i in xrange(len(testlabels)):
    #for j in xrange(len(allfeatures)):
	#Xtest[i,allfeatures[j]]=attributeclassifiers[j].predict(testdata[i,:])[0]
	
#fig = plt.figure()
#ax = fig.add_subplot(111)
#res = ax.imshow(X,cmap=plt.cm.bone,interpolation='nearest',aspect='auto')
#cb = fig.colorbar(res)
#plt.show()
##testdata = np.concatenate((testdata,Xtest),1)


#==============script to output data for use with cygwin MKL ============
#dataset = "vlg_extractor/ImageNetSurveyPicodes2048/ImageNetSurveyPicodes2048"
#dataset = "BoW2/ImageNet/ImageNetBoW2"
#recipedict = recipeDict[0] #change this
#fooddish = fooddish[0] #change this
#var=scipy.io.loadmat(dataset)
#traindata = np.ndarray.astype(var['X'],dtype=np.float)
#trainlabels = np.ndarray.astype(var['trainlabels'].flatten(),dtype=np.int)
#testdata = np.ndarray.astype(var['Xtest'],dtype=np.float)
#testlabels = var['testlabels'].flatten()
#images = var['testimages'][0]

#Xtest = pickle.load(file("/".join(dataset.split('/')[0:2])+'/IngredientAttributes.npy','r'))
#I,R = pickle.load(file(recipedict,'r'))
#ingsorted = sorted(I.keys())[1:]

#X = np.zeros((len(trainlabels),len(ingsorted)),dtype=np.int)

#for i in xrange(len(trainlabels)):
    #dish = fooddish[trainlabels[i]]
    #if len(R[dish].keys()) != 0:
	####randomly pick recipe
	#recipe = rnd.choice(R[dish].keys())
	##print recipe
	#X[i,:] = [1 if ing in R[dish][recipe] else 0 for ing in ingsorted]

#np.savez_compressed(dataset+"-MKL",traindata=traindata, testdata=testdata, X=X, 
        # Xtest=Xtest, trainlabels=trainlabels,testlabels=testlabels)

#pred=np.load(dataset+"-MKL_predictions.npz")
#y_true = pred['y_true']
#y_pred = pred['y_pred']
#from sklearn.metrics import classification_report
#print classification_report(y_true,y_pred)


#================= SCRIPT TO FIND POPULAR INGREDIENTS ====================
#ingredient histogram
#IngHist = {}
#for food in fooddish:
    #IngHist[food] = {}
    #for recipe in R[food].keys():
	#for ingredient in R[food][recipe].keys():
	    #if ingredient not in IngHist[food]:
		#IngHist[food][ingredient] = 1
	    #else:
		#IngHist[food][ingredient] += 1

#commonIngredients = [None]*len(fooddish)
#commonIngredientsIdx = []
#for f in xrange(len(fooddish)):
    #commonIngredientsIdx.extend([ingsorted.index(x) for x in IngHist[fooddish[f]].keys() if IngHist[fooddish[f]][x] >= 2 and x != '0'])

#commonIngredientsIdx = sorted(set(commonIngredientsIdx))
#pickle.dump(commonIngredientsIdx,file('CommonIngredientsImageNet.npy','w'))

#fig = plt.figure()
#i=9   
#ax = fig.add_subplot(1,1,1)
#r = plt.bar(np.arange(len(IngHist[fooddish[i]].keys())),IngHist[fooddish[i]].values())
#ax.set_xticks(np.arange(len(IngHist[fooddish[i]].keys()))+0.5)
#ax.set_xticklabels([I[x] for x in IngHist[fooddish[i]]],rotation=90,fontsize=8)
#ax.set_title(fooddish[i])
#ax.set_ylabel('Ingredient Count')
#plt.show()



if __name__=="__main__":
    IngredientScraper(fooddish[int(sys.argv[1])])

