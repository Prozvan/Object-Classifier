import numpy as np
import cv2 as cv

from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier

import warnings


from keras.models import load_model, save_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import Sequential, Input, utils, regularizers


from sklearn.svm import LinearSVC

from joblib import dump, load
    
from os import listdir, path


class Model():

    def __init__(self):
        self.St_Model = 0
        self.model = None  #če nč ne izbere pa klikne train nemore
        self.DEFAULT_NUM = 50850  # len(data)
        self.EPOCHS = 15

        

        
    def Train(self, samples):
        self.model = None
        # print(self.St_Model)
        data = []
        
        

        for imagePath in ["1", "2"]:
            for image in listdir(imagePath):



                #Procces image
                img = cv.imread(path.join(imagePath, image))  #113, 150 size
                img = np.array(img)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 

                img = img / 255  #Normalization of data
                    

                #KNN & LSVC - Flattening image
                CLASS = int(imagePath) - 1
                if self.St_Model == 1 or self.St_Model == 3:  
                    img = img.reshape(self.DEFAULT_NUM) 
                    data.append([img, CLASS])

                #NN
                elif self.St_Model == 2: 
                    data.append([img, CLASS])

                # print(imagePath,"/",image)
        


        if self.St_Model == 1 or self.St_Model == 3: #KNN
            x_train, y_train = self.Prepare_Data(data, samples)


            if self.St_Model == 1: self.KNN(x_train, y_train)
            elif self.St_Model == 3: self.LSVC(x_train, y_train)

        
        elif self.St_Model == 2: #NN

            x_train, y_train, valX, valY = self.Prepare_Data(data, samples)

            self.NN(x_train, y_train, valX, valY)
            
        
        return True
    

    def Prepare_Data(self, data, samples):

        if self.St_Model == 2:  #NN


            #How many samples for VAL in each object
            countVAL1 = (samples[0] * 20) // 100 
            countVAL2 = (samples[1] * 20) // 100 

            # MINIMUM(OBJ1_VAL, OBJ2_VAL)
            countVAL = 0
            if countVAL1 > countVAL2: countVAL = countVAL2
            elif countVAL2 > countVAL1: countVAL = countVAL1
            elif countVAL1 == countVAL2: countVAL = countVAL1
            
            print(r"These are 20% of both classes", countVAL1,  countVAL2, " - Went with MIN->", countVAL)
            

            # data is seperated between VAL and TRAIN data
            VAL = data[0:countVAL]+ data[samples[0]:samples[0] + countVAL]
            TRAIN = data[countVAL:samples[0]]+data[samples[0] + countVAL:]


            print("TRAIN: ", len(TRAIN), ", VAL: ", len(VAL))
          
            # print(VAL)

            #Sort data -> 0, 1, 0, 1
            NEW_TRAIN = []
            half = len(TRAIN) // 2
            for i in range(half):
                NEW_TRAIN.append(TRAIN[i])
                NEW_TRAIN.append(TRAIN[i+half])


            NEW_VAL = []
            half = len(VAL)//2
            for i in range(half):
                NEW_VAL.append(VAL[i])
                NEW_VAL.append(VAL[i+half])



            #Separate VAL and TRAIN to labels and features
            trainX, trainY = [],[]
            for pic, label in NEW_TRAIN: 
                trainX.append(pic)
                trainY.append(label)


            valX, valY = [],[]
            for pic, label in NEW_VAL: 
                valX.append(pic)
                valY.append(label)


            #To Numpy array - FASTER Array
            trainX = np.array(trainX)
            trainY = np.array(trainY)

            valX = np.array(valX)
            valY = np.array(valY)


            
            return trainX, trainY, valX ,valY
        
        
        
        elif self.St_Model == 1 or self.St_Model == 3:  #KNN LSVC
            


            #To labels, features
            trainX, trainY = [], []
            for feature, label in data:

                trainX.append(feature)
                trainY.append(label)


            print("Valid Samples:", len(trainY))


            #To numpy array
            trainX = np.array(trainX)
            trainY = np.array(trainY)

            print("Train Y SHAPE:",trainY.shape,"- Train X SHAPE:", trainX.shape)

            return trainX, trainY






    #ALGORITHMSSSSSS
    

    
    def KNN(self, x_train, y_train):
        #Found that 3 n is the best in most casses
        self.model = KNeighborsClassifier(n_neighbors=3)
            
        #Fit labels and features into a KNN
        self.model = self.model.fit(x_train, y_train)  
    

    def LSVC(self, x_train, y_train):
        with warnings.catch_warnings():  #To catch warnings
            warnings.filterwarnings("ignore", category=ConvergenceWarning) 

            #Linear Support Vector Classifier
            self.model = LinearSVC()

            #Fit labels and features into a LSVC
            self.model.fit(x_train, y_train)



    def NN(self, x_train, y_train, valX, valY):
        
        #Monitoring
        c1, c2 = 0,0
        for i in range(len(y_train)):
            if y_train[i] == 0: c1+=1
            elif y_train[i] == 1: c2+=1

        print(f"Za train -> 1. class: {c1}, 2. class: {c2}")
        print(valY)

        

        self.model = Sequential([
            Input(shape=(113, 150, 3)),  #rows, cols, R G B


            #RELU -> if x <= 0: x = 0
            #        elif x > 0: x = x


            #SIGMOID od 0 do 1:
            # bližje 0 kot 1 -> object 1
            # bližje 1 kot 0 -> object 2  


            Conv2D(32, (3,3), 1, activation="relu"),
            MaxPooling2D(2,2),

            Conv2D(64, (3,3), 1,  activation="relu"),
            MaxPooling2D(2,2),

            Conv2D(32, (3,3), 1, activation="relu"),
            MaxPooling2D(2,2),

            Conv2D(16, (3,3), 1, activation="relu"),
            MaxPooling2D(2,2),

            Dropout(0.1),
            Flatten(),  #iz 2D ali več v 1D array

            Dense(256, activation="relu"),
            Dense(1, activation="sigmoid")
        
        ])


        #Adam - how it adjusts weights and bias
        #binary crossentropy - how well does the A preform on train 
        self.model.compile(optimizer="adam", loss='binary_crossentropy', 
                            metrics=['accuracy'])


        #Fit labels and features of VAL and TRAIN data
        #Epochs - how many times its cycles through data
        #Batch size - how much data to feed it into at the time
        self.model.fit(x_train, y_train, epochs=self.EPOCHS, batch_size=32, 
                       validation_data = (valX, valY))
    

    def Predict_Class(self):
   
        img = cv.imread("Predict/frame.jpg")  #113, 150 size
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)     #To RGB 
        
        img = img / 255  #Normalization

        
        if self.St_Model == 1:  #K-Nearest Neighbors
            img = img.reshape(self.DEFAULT_NUM)

            prediction = self.model.predict([img])

            print("Pred ->", prediction)

            return prediction[0]
        

        elif self.St_Model == 2:  #Neural Network
            
            img = img.reshape(-1, 113, 150, 3)

            pred = self.model.predict(img, verbose=0)

            if pred < 0.5: prediction = 0
            else: prediction = 1
            
            print("Pred ->", pred ,"------- After argmax ->", prediction)
            return prediction 
        
        
        elif self.St_Model == 3:  #Linear Support Vector Classifier
            img = img.reshape(self.DEFAULT_NUM)  

            prediction = self.model.predict([img])  #
            print("Pred ->", prediction)
            
            return prediction[0]         

    


# SAVEEEEEEEEEEEEEEEEEEEEEEEE  LOADDDDDDDDDDDDD
            

    def Save_Model(self, DIR, TrainingStatus):
        if self.St_Model == 1: kratica = "KNC" #KNN
        elif self.St_Model == 2: kratica = "NN"  #NN
        elif self.St_Model == 3: kratica = "LSVC" #LSVC 
        else: kratica = "Empty"    #No model selected

        if TrainingStatus:  #If its trained
                 
            if self.St_Model == 2:  
                save_model(self.model, DIR)  #Save for NN

            elif self.St_Model == 1 or self.St_Model == 3: 
                dump(self.model, DIR) #Save for LSVC, KNN
           
        return kratica  #Vrne Kratico modela
    
    
    def Load_Model(self, DIR, ime_modela, trainingStatus):
        #Ker model
        print()
        if ime_modela[0] == "K": self.St_Model = 1
        elif ime_modela[0] == "N": self.St_Model = 2
        elif ime_modela[0] == "L": self.St_Model = 3  
        
        elif ime_modela[0] == "E": 
            
            self.St_Model = 0
            self.model = None
        
        print(trainingStatus)
        if trainingStatus:
            try:
                if self.St_Model == 2: 
                    self.model = load_model(DIR)
                
                elif self.St_Model == 1 or self.St_Model == 3:
                    self.model = load(DIR)
            except OSError as e: pass
        

           

        return self.St_Model
        


  

  


    
        
    
        
            
        

