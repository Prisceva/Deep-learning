import os
import csv  
import pandas
import shutil

n=pandas.read_csv("/Users/priscille/Desktop/whale-categorization-playground/train.csv",sep=";")
print(n.head())

#https://www.geeksforgeeks.org/how-to-count-distinct-values-of-a-pandas-dataframe-column/
cnt = 0
visited = [] 
   
for i in range(0, len(n['Id'])): 
    if n['Id'][i] not in visited:  
        visited.append(n['Id'][i]) 
        cnt += 1
#print("No.of.unique values :", cnt) 
#print("unique values :", visited)

#https://stackabuse.com/creating-and-deleting-directories-with-python/
for i in range(len(visited)) :
  path = "/pathto.../whale-categorization-playground/data/" + visited[i]

  try:
      os.makedirs(path)
  except OSError:
      print ("Creation of the directory %s failed" % path)
  else:
      print ("Successfully created the directory %s" % path)


for image,label in n.itertuples(index=False): 
    shutil.move("/Users/priscille/Desktop/whale-categorization-playground/train/"+image, "/Users/priscille/Desktop/whale-categorization-playground/data/"+label+"/"+image)