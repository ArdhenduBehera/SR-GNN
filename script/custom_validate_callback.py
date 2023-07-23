# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:44:09 2018

@author: beheraa

from custom_validate_callback import TestCallback

callbacks = [TestCallback(test_datagen, 10)] # the model evaluate every 10 epochs
"""
import keras
from sklearn.metrics import accuracy_score
import numpy as np
import csv
#from os.path import dirname, realpath

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name):
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs={}):        
        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
            try:
                loss, acc = self.model.evaluate_generator(self.test_generator)
                #prediction = self.model.predict_generator(self.test_generator)
            except:
                #model is regression, so must approximate accuracy
                loss = self.model.evaluate_generator(self.test_generator)
                acc = validateRegression(self.test_generator, self.model)
                
            print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))
            
            # writeValToCSV(self, epoch, loss, acc)
                
                
                
def validateRegression(val_dg, model):
    predsAcc=[]
    trues=[]
    
    for b in range(len(val_dg)):
        x, y_true = val_dg.__getitem__(b)
        pred = model.predict(x)
        
        batch_size = pred.size
        
        
        #put trues in column
        y_true = np.reshape(y_true, (batch_size,1))
        
        for i in range(0, batch_size):
            
            y_true[i] = int(y_true[i] * 90)
            pred[i] = int(pred[i] * 90)
            
            
            if y_true[i] - 22.5 <= pred[i] <= y_true[i] + 22.5:
                pred[i] = y_true[i]
            else:
                pred[i] = int(-10) #rogue value
        
        predsAcc.append(pred[i])
        trues.append(y_true[i])
    
    #print(predsAcc)
    #print(trues)
    return accuracy_score(trues,predsAcc)

#writes validation metrics to csv file
def writeValToCSV(self, epoch, loss, acc):
    
    #get root directory
    #filepath = realpath(__file__)
    #metrics_dir = dirname(dirname(filepath)) + '/Metrics/'
    metrics_dir = 'Metrics/'
    
    
    with open(self.model_name + '(Validation).csv', 'a', newline='') as csvFile:
        metricWriter = csv.writer(csvFile)
        metricWriter.writerow([epoch, loss, acc])