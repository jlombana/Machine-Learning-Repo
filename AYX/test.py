import numpy as np
import copy 

# listProducts contains the simulations (permutations) for all possible 
# combinations of items
# listShelves contains the shelves (horizontal or vertical) that are fixed
# shelvesMaster contains all listShelves once loaded with all products
# eg.
### Step 1: Run a simulation that returns 7 combinations of products
### Step 2: load each simulation into the canvas (shelves)
### Step 3: load the resulting shelf (listShelves) to the  shelfMaster
### Result:    
########## 7 combinations x 1 shelf (1...n) = 7 
########## 7 listShelves [] 

listShelves =\
    [['1','2','3','4','5','6','7','8','9','10'],\
    ['11','12','13','14','15','16','17','18','19','20','21','22','23'],\
    ['24','25','26','27','28','29','30'],\
    ['31','32','33','34','35','36','37','38','39','40']]
        
listProducts =\
   [['G1','G1','G1','G1'],\
    ['G2','G2','G2','G2'],\
    ['G3','G3','G3','G3'],\
    ['G4','G4','G4','G4'],\
    ['G5','G5','G5','G5'],\
    ['G6','G6'],\
    ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12']]

#############################
########VARIABLES############
#############################
       
listShelves2 =\
    [['1','2','3'],\
    ['4','5','6','7','8','9','10','11','12','13','14']]
        
listProducts2 =\
   [['G1','G2','G3','G4','G5','G6','G7','G8'],\
    ['G9','G10','G11','G12']]

shelfMasterFlat = []
shelfMasterComposed = []
listShelvesAux = []

# resultado:
#shelfMaster =\
#    [['G1','G2','G3'],\
#    ['G4','5','6','7','8']]
#    [['G5','G6','G7'],\
#    ['G8','5','6','7','8']]


#############################
##########METHODS############
#############################

def getFlatList(list):
    listShelves2 = [item for sublist in list for item in sublist]
    return listShelves2

def getComposedList(list):
    listShelves2 = [item for sublist in list for item in sublist]
    return listShelves2

def initializeList(list):
    return copy.deepcopy(list)

###########################
###########################
###########################
start = 0
flatListShelf = getFlatList(listShelves)
listShelvesAux = []

# Fit products to shelves (flat mode)
end = start + len(flatListShelf)
start = 0
for item in listProducts:
    listShelvesAux = initializeList(flatListShelf)
    i = 0 
    for a, b in zip(flatListShelf , item):
        listShelvesAux[i] = b
        i = i + 1
    
    shelfMasterFlat.append(listShelvesAux)
    
# tranform resulting list to original structure (list of lists)
for list in shelfMasterFlat:
    start = 0 
    end = 0 
    i = 0
    biglist = []
    for a, b in zip(listShelves , list):
        biglist.append(list[start:len(a)+start])
        start = len(a)+start
        end = len(a)+start
    
    shelfMasterComposed.append(biglist)



