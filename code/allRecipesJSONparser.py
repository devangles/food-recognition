import json
from labels import *
import sys
import pickle

json_file = file("../AllRecipes/tutorial/recipe.jl","r")

#dictionary for ingredients
I = {}
#dictionary for food recipes
R = {}

recipes = foodlabels[3]
i = 1
for line in json_file:
    recipe = json.loads(line)
    recipename = recipe['name'][0].replace(" ","_")
    if recipename in recipes:
	print recipename
	R[recipename] = [{},[]]
	ingredients = recipe['ingredients']
	for ingid, ingname in ingredients.items():
	    if ingid not in I:
		I[ingid] = ingname[0]
	    
	    amt = ingname[1]
	    R[recipename][0][ingid] = amt
	R[recipename][1] = recipe['directions']

   
## save Recipe info to disk
pickle.dump((I,R),file('AllRecipes50.npy','w'))        


