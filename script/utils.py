import sys
import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K

#get regions of interest of an image (return all possible bounding boxes when splitting the image into a grid)
def getROIS(resolution=33,gridSize=3, minSize=1):
	
	coordsList = []
	step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
	for column1 in range(0, gridSize + 1):
		for column2 in range(0, gridSize + 1):
			for row1 in range(0, gridSize + 1):
				for row2 in range(0, gridSize + 1):
					
					#get coordinates using grid layout
					x0 = int(column1 * step)
					x1 = int(column2 * step)
					y0 = int(row1 * step)
					y1 = int(row2 * step)
					
					if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
						
						if not (x0==y0==0 and x1==y1==resolution): #ignore full image
							
							#calculate height and width of bounding box
							w = x1 - x0
							h = y1 - y0
							
							coordsList.append([x0, y0, w, h]) #add bounding box to list

	coordsArray = np.array(coordsList)	 #format coordinates as numpy array						

	return coordsArray
	
def getIntegralROIS(resolution=42,step=8, winSize=14):
    coordsList = []
    #step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
    for column1 in range(0, resolution, step):
        for column2 in range(0, resolution, step):
            for row1 in range(column1+winSize, resolution+winSize, winSize):
                for row2 in range(column2+winSize, resolution+winSize, winSize):
                    #get coordinates using grid layout
                    if row1 > resolution or row2 > resolution:
                        continue
                    x0 = int(column1)
                    y0 = int(column2)
                    x1 = int(row1)	
                    y1 = int(row2)	
                    
                    #if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
                    #    if not (x0==y0==0 and x1==y1==resolution): #ignore full image
                            #calculate height and width of bounding box
                    if not (x0==y0==0 and x1==y1==resolution):  #ignore full image
                        w = x1 - x0
                        h = y1 - y0
                        coordsList.append([x0, y0, w, h]) #add bounding box to list
    #coordsList.append([0, 0, resolution, resolution])#whole image
    coordsArray = np.array(coordsList)	 #format coordinates as numpy array
    return coordsArray	

def crop(dimension, start, end): #https://github.com/keras-team/keras/issues/890
    #Use this layer for a model that has individual roi bounding box
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return layers.Lambda(func)

def squeezefunc(x):
    return K.squeeze(x, axis=1)

'''This is to convert stacked tensor to sequence for LSTM'''
def stackfunc(x):
    return K.stack(x, axis=1) 