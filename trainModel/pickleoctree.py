import pickle
def saveOctree(octree,inputName):
  with open(inputName,"wb") as file:
    pickle.dump(octree,file)

def readOctree(inputName):
  with open(inputName,"rb") as file:
    data=pickle.load(file)
  return data

def check():
    print("yes")
