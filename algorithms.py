import numpy as np
import cv2 as cv
import warnings
import random as r


from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight


from keras.models import load_model, save_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import Sequential, Input

from joblib import dump, load
from os import listdir, path


class Model():

    def __init__(self):
        self.modelIndex = 0
        self.model = None  #če nč ne izbere pa klikne train nemore
        self.DEFAULT_NUM = 72900  # len(data)
        self.EPOCHS = 10

        

        
    def Train(self, samples):

        def applyFilters(image, object, flt):

            #Parameters
            _factor = r.uniform(0.25, 0.75)
            # FACTORS = ["Darken", "Lighten"]
    

            
            if flt: #Light
                _factor /= 2
                _factor += 1
                filtered_image = np.clip(image * _factor, 0, 1)

            else:  #Dark
                _factor += 0.05
                filtered_image = np.clip(image * _factor, 0, 1)
            
            #Appending to data
            samples[object] += 1
            data.append([filtered_image, object])

         

            # print(f"Applying Factor: {_factor} {FACTORS[flt]}, {object} ----- {flt}")

            return flt



        self.model = None
        data = []
        
        filterType = 0
        global filterCount
        filterCount = 0
        for imagePath in ["1", "2"]:
            for image in listdir(imagePath):

                
                filterCount += 1

                
                #Procces image
                img = cv.imread(path.join(imagePath, image))  #113, 150 size
                img = np.array(img)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 

                img = img / 255  #Normalization of data
                    

                #If its class 2 or 1 > 1 or 0
                CLASS = int(imagePath) - 1

                #KNN - Flattening image
                if self.modelIndex == 1:  
                    img = img.reshape(self.DEFAULT_NUM) 
                    data.append([img, CLASS])

                #NN - no need to reshape it
                elif self.modelIndex == 2: 
                    data.append([img, CLASS])

                #LSVC - Flattening image
                elif self.modelIndex == 3:
                    img = img.reshape(self.DEFAULT_NUM) 
                    data.append([img, CLASS])



                #Darken or Lighten
                if filterCount % 5 == 0:  #ZA vsako 5 sliko se naredi ena filtered slika

                    if filterType == 0: filterType = applyFilters(img, CLASS, 1)
                    elif filterType == 1: filterType = applyFilters(img, CLASS, 0)




                # print(imagePath,"/",image)
        


        if self.modelIndex in [1, 3]: #KNN
            x_train, y_train = self.Prepare_Data(data, samples)


            if self.modelIndex == 1: self.KNN(x_train, y_train)

            elif self.modelIndex == 3: self.LSVC(x_train, y_train)

        
        elif self.modelIndex == 2: #NN

            x_train, y_train, valX, valY = self.Prepare_Data(data, samples)

            self.NN(x_train, y_train, valX, valY)

        return True
    

    def Prepare_Data(self, data, samples):

        if self.modelIndex == 2:  #NN


            #How many samples for VAL in each object
            countVAL1 = (samples[0] * 20) // 100 
            countVAL2 = (samples[1] * 20) // 100 

            # MINIMUM(OBJ1_VAL, OBJ2_VAL)
            countVAL = 0
            if countVAL1 > countVAL2: countVAL = countVAL2
            elif countVAL2 > countVAL1: countVAL = countVAL1
            elif countVAL1 == countVAL2: countVAL = countVAL1
            print(f"Filtered Images: {(filterCount//5)}")
            print(r"These are 20% of both classes", countVAL1,  countVAL2, " - Went with MIN ->", countVAL, "(includes filtered images)")
            

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
        
        
        
        elif self.modelIndex in [1, 3]:  #KNN LSVC
            


            #To labels, features
            trainX, trainY = [], []
            for feature, label in data:

                trainX.append(feature)
                trainY.append(label)


            # print("Valid Samples:", len(trainY))


            #To numpy array
            trainX = np.array(trainX)
            trainY = np.array(trainY)

            print("Train Y SHAPE:",trainY.shape,"- Train X SHAPE:", trainX.shape)
            print(f"Filtered Images: {(filterCount//5)}")

            return trainX, trainY






    #ALGORITHMSSSSSS
    

    
    def KNN(self, x_train, y_train):
        #Found that 3 n is the best in most casses
        self.model = KNeighborsClassifier(n_neighbors=3)
            
        #Fit labels and features into a KNN
        self.model = self.model.fit(x_train, y_train)  
    

    def LSVC(self, x_train, y_train):
        # print("LSVCCCCC")
        with warnings.catch_warnings():  #To catch warnings
            warnings.filterwarnings("ignore", category=ConvergenceWarning) 
            warnings.filterwarnings("ignore", category=FutureWarning) 

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
        print("Validation y:",valY[:10])

        

        self.model = Sequential([
            Input(shape=(135, 180, 3)),  #rows, cols, R G B


            #RELU -> if x <= 0: x = 0
            #        elif x > 0: x = x

            #SIGMOID od 0 do 1:
            # bližje 0 kot 1 -> object 1
            # bližje 1 kot 0 -> object 2  

            #Extracts features (multiply - to produce 1 value) 
            Conv2D(32, (3,3), 1, activation="relu"), 
            MaxPooling2D(2,2), #Zmanjša

            Conv2D(64, (3,3), 1,  activation="relu"),
            MaxPooling2D(2,2),

            Conv2D(64, (3,3), 1, activation="relu"),
            MaxPooling2D(2,2),

            Conv2D(16, (3,3), 1, activation="relu"),
            MaxPooling2D(2,2),

            Dropout(0.35),
            Flatten(),  #iz 2D ali več v 1D array

            Dense(512, activation= "relu"),  
            Dense(1, activation="sigmoid")
        
        ])

        #Adam - how it adjusts weights and bias
        #binary crossentropy - how well does the A preform on train 
        self.model.compile(optimizer="adam", loss='binary_crossentropy', 
                            metrics=['accuracy'])

        #Fit labels and features of VAL and TRAIN data
        #Epochs - how many times its cycles through data
        #Batch size - how much data to feed it into at the time
        self.model.fit(x_train, y_train, epochs=self.EPOCHS, batch_size=64, 
                       validation_data = (valX, valY))
    

    def Predict_Class(self):
   
        imgData = cv.imread("Predict/frame.jpg")  #113, 150 size
        imgData = np.array(imgData)
        img = cv.cvtColor(imgData, cv.COLOR_BGR2RGB)     #To RGB 
        
        img = img / 255  #Normalization

        
        #Linear Support Vector Classifier and  #K-Nearest Neighbors
        if self.modelIndex in [1, 3]: 
            img = img.reshape(self.DEFAULT_NUM)

            prediction = self.model.predict([img])

            print("Prediction ->", prediction)

            return prediction[0]
        

        elif self.modelIndex == 2:  #Neural Network
            
            img = img.reshape(-1, 135, 180, 3)

            pred = self.model.predict([img], verbose=0)

            if pred < 0.5: prediction = 0
            else: prediction = 1
            
            print("Prediction ->", pred ,"------- After argmax ->", prediction)
            return prediction 
        
        
    


# SAVEEEEEEEEEEEEEEEEEEEEEEEE  LOADDDDDDDDDDDDD
            

    def Save_Model(self, DIR, TrainingStatus):
        if self.modelIndex == 1: algorithm = "KNC" #KNN
        elif self.modelIndex == 2: algorithm = "NN"  #NN
        elif self.modelIndex == 3: algorithm = "LSVC" #LSVC 
        else: algorithm = "EMPTY"    #No model selected

        if TrainingStatus:  #If its trained
            
            try:
                if self.modelIndex == 2:  
                    
                    save_model(self.model, DIR)  #Save for NN

                elif self.modelIndex in [1, 3]: 
                    dump(self.model, DIR) #Save for LSVC, KNN
            except Exception as e:
                return algorithm, True
           
        return algorithm, None  #Vrne Kratico modela
    
    
    def Load_Model(self, DIR, algorithm, trainingStatus):
        #Which model
        print()
        if algorithm == "KNC": self.modelIndex = 1
        elif algorithm == "NN": self.modelIndex = 2
        elif algorithm == "LSVC": self.modelIndex = 3  
        elif algorithm == "EMPTY": 
            self.modelIndex = 0
            self.model = None
        
       
        if trainingStatus:  #If trained
            try:
                if self.modelIndex == 2: 
                    self.model = load_model(DIR) #Load NN
                
                elif self.modelIndex in [1, 3]:
                    self.model = load(DIR)   #Load KNC, LSVC

            except OSError as e: pass  #Weird Error
        

        return self.modelIndex
        


  

  


    
        
    
        
            
        

