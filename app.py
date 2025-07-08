from tkinter import messagebox
import PIL.Image, PIL.ImageTk
import os
from threading import Thread
from shutil import copytree

import algorithms



class App:

    def __init__(self):
        # print("APP")
        self.listAlgorithms = ["K-Nearest Neighbors", "Neural Network",
                               "Linear Support Vector Classifier"]

        self.MODEL = algorithms.Model()
        

        self.object_1 = "1"
        self.object_2 = "2"

        self.samples = [0,0]
    

        self.SavedWithoutAlogorithm = False
        self.NoFile = False
        self.TrainingStatus = False
        self.saveError = False
         
        self.object = 0
        self.before = 0
        self.active = 0
        self.predictedClass = -1

    
        self.DEF_DIR = r"Projects\\"
        self.JSONFILE = "data.json"
        self.MODEL_NAME = "Model.keras"

        #Important folders                 #current directory
        if "Projects" not in os.listdir(os.getcwd()): os.makedirs("Projects")
        if "1" not in os.listdir(os.getcwd()): os.makedirs("1")
        if "2" not in os.listdir(os.getcwd()): os.makedirs("2")
        if "Predict" not in os.listdir(os.getcwd()): os.makedirs("Predict")
        
        
        
        



    def Object_Name(self, name): #Pusti not
        
        try: name[1]  #If there isn't 
        except IndexError: return 0, 0  

        else:
            inputName = name[1]

            if len(inputName) <= 10: 
                if all(c != '' and c != ' ' for c in inputName):
                    
                    if self.object_1 == name[0] and inputName != self.object_2:
                        print(f"Name change -> {self.object_1} - {name}")
                        self.object_1 = inputName
                        return inputName, 1
                        

                    elif self.object_2 == name[0] and inputName != self.object_1:
                        print(f"Name change -> {self.object_2} - {name}")
                        self.object_2 = inputName
                        return inputName, 2

            else: messagebox.showerror("Length Error", "To long name! -> Max length: 10 characters")
    
        return 0, 0
    
       

    def Select_Algorithm(self, st):
        self.TrainingStatus = False
        self.MODEL.EPOCHS = 10  #Its set to Default
        self.MODEL.modelIndex = st

        print("New algorithm selected ->", self.listAlgorithms[st - 1])




    #Change Epochs for Neural network
    def Change_Epochs(self, epochs): self.MODEL.EPOCHS = epochs
    

           



#KeyStrokes or Button
    def Key_Pressed(self, key, image):
        
        if key == 1 and key != self.before and self.active != 2:
            
            self.active = 1
            self.before = key
            self.Save_Image(1, image)
           
        elif key == 2 and key != self.before and self.active != 1:
            
            self.active = 2
            self.before = key
            self.Save_Image(2, image)

        elif key == 11 and self.active == 1:
            self.active = 0
            self.before = key
            
        elif key == 22 and self.active == 2:
            self.active = 0
            self.before = key
        
        return self.samples
    


    def Button_Click(self, obj, image):
        
        Thread(target=self.Save_Image, args=[obj, image]).start()
        # self.Save_Image(obj, image)

        # return self.samples

    def Save_Image(self, object, image):
 
        self.object = object
        # slika = self.image
         
        self.samples[self.object-1]+=1
        print("Samples:", self.samples)
       

        if not os.path.exists("1"): os.mkdir("1")
        if not os.path.exists("2"): os.mkdir("2")
        
        
        img = PIL.Image.fromarray(image)
        img.thumbnail((180,180), PIL.Image.LANCZOS)
        
        img.save(f"{self.object}/slika{self.samples[self.object-1]}.jpg")
        




    #TRAINNNNNNNNNNNNNNNNNN
    def Parallel_Train_Algorithm(self): 
        Thread(target=self.Train_Algorithm).start()
       
    
    def Train_Algorithm(self):
        self.samples = self.Number_Of_Samples()
        self.TrainingStatus = False

        self.TrainingStatus = self.MODEL.Train(self.samples)
                
        print(f"TRAIN: {self.samples} -- including filtered images")


        
    #SAVEEEEEEEEEEEEEEEEEEEEEE
    def Parallel_Save_Algorithm(self, obj1, obj2,TStatus, DIR):

        Thread(target=self.Save_Algorithm, args=[obj1, obj2,TStatus, DIR]).start()
        

    def Save_Algorithm(self, obj1, obj2, TrainingStatus, PROJECT):  
        
        
        if PROJECT not in os.listdir(self.DEF_DIR):  #If the name isn't used
            try:
                #Creates a project folder in a "Projects" folder
                directory = os.path.join(self.DEF_DIR, PROJECT)
                os.mkdir(directory)
                    
                
                modelPath = os.path.join(directory, self.MODEL_NAME)
                
                modelName, fill = self.MODEL.Save_Model(modelPath, TrainingStatus)
                
                if fill != None: TrainingStatus = False

                self.samples = self.Number_Of_Samples()
                #For JSON - important data
                data = { "Model": modelName,
                        "TrainingStatus": TrainingStatus,
                        "Samples1": self.samples[0], 
                        "Samples2": self.samples[1], 
                        "Object1": self.object_1,
                        "Object2": self.object_2,}
                        
                #Writes to data.json file
                PATH = os.path.join(directory, self.JSONFILE)
                with open(PATH, "w") as file:
                    for key, value in data.items(): #All key values
                        file.write(f"{key}: {value}\n")

                                    
                #Pictures are copied to the Project folder
                copytree("1", f"{directory}/TrainingData/{obj1}", dirs_exist_ok=True)
                copytree("2", f"{directory}/TrainingData/{obj2}", dirs_exist_ok=True)     
                
                print(f"Saved to: {directory}")

            except PermissionError as e:
                messagebox.showerror("Error", "Permission denied while copying files!")

            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")       

        else: 
            messagebox.showerror("Error", "Project name is already used!!")
            


    #LOADDDDDDDDDDDDDDDD

    def Load_Algorithm(self, DIR):
    
        print(f"Path of a Project: {DIR}")


        #If in the folder there are data.json and model.pkl
        if self.JSONFILE in os.listdir(DIR) and self.MODEL_NAME in os.listdir(DIR):   #za txt
                
            self.TrainingStatus = self.Get_Data(DIR, self.JSONFILE)
            self.SavedWithoutAlogorithm = False
            print("Trained!!",self.TrainingStatus)
            return self.TrainingStatus

        elif self.JSONFILE in os.listdir(DIR):  #No trained Model
            
            self.TrainingStatus = self.Get_Data(DIR, self.JSONFILE)
            self.SavedWithoutAlogorithm = True
            print("Not Trained",self.TrainingStatus)
            return self.TrainingStatus

        else:  #Errors
            messagebox.showerror("Error","Couldn't find a file OR Selcted DIR wasn't a project!!")
            self.NoFile = True
            return False
        

        
    def Get_Data(self, DIR, JSONFILE):
        data = {}   #For data
        
        #Geting settings from data.json file
        with open(os.path.join(DIR, JSONFILE), "r") as file:
            for line in file:
                key, value = line.strip().split(": ")

                #Appending to dictionary
                data[key] = value  

        #Geting data from that dictionary
        ime_modela = data["Model"]
        self.object_1 = data["Object1"]
        self.object_2 = data["Object2"]
        
        self.TrainingStatus = eval(data["TrainingStatus"])  #True / False

        self.samples[0] = int(data["Samples1"])
        self.samples[1] = int(data["Samples2"])

        print()
        print("Data.json ->",data)
        print()

        #Loads algorithm
        self.modelIndex = self.MODEL.Load_Model(DIR + "/"+ self.MODEL_NAME, ime_modela, self.TrainingStatus)

    
        return self.TrainingStatus
        
        
        

    def Return_Load_Parameters(self):
        if self.SavedWithoutAlogorithm: #There is an algorithm saved
            if self.NoFile:  #Če ni najdenga projekta
                return None, self.object_1, self.object_2, self.modelIndex
            
            elif self.NoFile == False:
                return self.samples, self.object_1, self.object_2, self.modelIndex


        elif self.SavedWithoutAlogorithm == False:
            if self.NoFile:
                return None, self.object_1, self.object_2, self.modelIndex
            
            elif self.NoFile == False:
                return self.samples, self.object_1, self.object_2, self.modelIndex


    #Loading images
    def Load_Images(self, DIR):
        
        copytree(f"{DIR}/TrainingData/{self.object_1}", "1",dirs_exist_ok=True)
        copytree(f"{DIR}/TrainingData/{self.object_2}", "2", dirs_exist_ok=True)
        print("AfterLoad - Finished")


        

        
    def Number_Of_Samples(self):
        self.samples = [0,0]
        for folder in ["1","2"]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file) #Skp
    
                if os.path.isfile(file_path) and folder == "1": 
                    self.samples[0]+=1
                if os.path.isfile(file_path) and folder == "2": 
                    self.samples[1]+=1

        return self.samples

    


    def Reset_Project(self):
        print("Reset")

        self.samples = [0,0]
        self.MODEL = algorithms.Model()  

        self.SavedWithoutAlogorithm = False
        self.NoFile = False 
        self.saveError = False
        self.TrainingStatus = False

        self.object_1 = "1"
        self.object_2 = "2"

        self.object = 0
        self.before = 0
        self.active = 0
        self.predictedClass = -1

        self.Change_Epochs(10)

        for i in ["1","2"]:
            for file in os.listdir(i):
                pot = os.path.join(i,file)
                if os.path.isfile(pot): os.unlink(pot)
        
        
    
    def Parallel_Predict(self, img):
        Thread(target=self.Predict, args=[img]).start()

    def Predict(self, img):
        img.thumbnail((180,180), PIL.Image.LANCZOS)
        
        img.save("Predict/frame.jpg")
        try:
            self.predictedClass = self.MODEL.Predict_Class()
        except Exception as e:
            pass



        
        

    