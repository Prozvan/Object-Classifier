from tkinter import simpledialog, filedialog, messagebox
import PIL.Image, PIL.ImageTk
import os
import threading
import shutil
from json import dump, dumps, load 

import kamera
import model



class App:

    def __init__(self):
        self.listAlgorithms = ["K-Nearest Neighbors", "Neural Network",
                               "Linear Support Vector Classifier"]

        self.KAMERA = kamera.Kamera()
        self.MODEL = model.Model()
        

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
        self.MODEL_NAME = "Model.pkl"
        if "Projects" not in os.listdir(os.getcwd()): os.makedirs("Projects")
        
        
        
        



    def Object_Name(self, name): #Pusti not
        
        try: name[1]  #If there isn't 
        except IndexError: return 0, 0  

        else:
            inputName = name[1]

            if len(inputName) <= 10: 
                if all(c != '' and c != ' ' for c in inputName):

                    if self.object_1 == name[0] and inputName != self.object_2:
                        self.object_1 = inputName
                        return inputName, 1
                        

                    elif self.object_2 == name[0] and inputName != self.object_1:
                        self.object_2 = inputName
                        return inputName, 2

            else: messagebox.showerror("Error", "To long name! -> Max length: 10 characters")
    
        return 0, 0
    
       

    def Select_Algorithm(self, st):
        self.TrainingStatus = False
        self.MODEL.EPOCHS = 15  #Its set to Default
        self.MODEL.St_Model = st

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
        
        self.Save_Image(obj, image)

        return self.samples

    def Save_Image(self, object, image):
 
        self.object = object
        # slika = self.image
         
        self.samples[self.object-1]+=1
        print("Samples:", self.samples)
       

        if not os.path.exists("1"): os.mkdir("1")
        if not os.path.exists("2"): os.mkdir("2")
        
        
        # img = PIL.Image.fromarray(cv.cvtColor(slika, cv.COLOR_BGR2GRAY))
        img = PIL.Image.fromarray(image)
        img.thumbnail((150,150), PIL.Image.LANCZOS)
        
        img.save(f"{self.object}/slika{self.samples[self.object-1]}.jpg")
        




    #TRAINNNNNNNNNNNNNNNNNN
    def Parallel_Train_Algorithm(self): 
        threading.Thread(target=self.Train_Algorithm).start()
       
    
    def Train_Algorithm(self):
        
        self.TrainingStatus = False

        self.TrainingStatus = self.MODEL.Train(self.samples)
                
        print(f"TRAIN: {self.samples}")


        
    #SAVEEEEEEEEEEEEEEEEEEEEEE
    def Parallel_Save_Algorithm(self, obj1, obj2,TStatus, DIR):

        threading.Thread(target=self.Save_Algorithm, args=[obj1, obj2,TStatus, DIR]).start()
        

    def Save_Algorithm(self, obj1, obj2, TrainingStatus, PROJECT):  
        
        
        if PROJECT not in os.listdir(self.DEF_DIR):  #If the name isn't used
            try:
                #Creates a project folder in a "Projects" folder
                directory = os.path.join(self.DEF_DIR, PROJECT)
                os.mkdir(directory)
                    
                
                modelPath = os.path.join(directory, self.MODEL_NAME)
                Model_Name = self.MODEL.Save_Model(modelPath, TrainingStatus)


                #For JSON - important data
                data = { "Model": Model_Name,
                        "TrainingStatus": TrainingStatus, 
                        "Object1": self.object_1,
                        "Object2": self.object_2,}
                        
                #Writes to data.json file
                PATH = os.path.join(directory, self.JSONFILE)
                with open(PATH, "w") as file:
                    for key, value in data.items(): #All key values
                        file.write(f"{key}: {value}\n")

                                    
                #Pictures are copied to the Project folder
                shutil.copytree("1", f"{directory}/TrainingData/{obj1}", dirs_exist_ok=True)
                shutil.copytree("2", f"{directory}/TrainingData/{obj2}", dirs_exist_ok=True)     
                    

            except PermissionError as e:
                messagebox.showerror("Error", "Permission denied while copying files!")

            except Exception as e:
                messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")       

        else: 
            messagebox.showerror("Error", "Project name is already used!!")
            


    #LOADDDDDDDDDDDDDDDD
    def Get_Data(self, DIR, JSONFILE):
        data = {}
        
        with open(os.path.join(DIR, JSONFILE), "r") as file:
            for line in file:
                key, value = line.strip().split(": ")
                data[key] = value

        ime_modela = data["Model"]
        self.object_1 = data["Object1"]
        self.object_2 = data["Object2"]
        TrainingStatus = eval(data["TrainingStatus"])

        # print(self.predmet_ena, self.predmet_dva)
        print()
        print(data)
        print()

        self.St_Model = self.MODEL.Load_Model(DIR + "/"+ self.MODEL_NAME, ime_modela, TrainingStatus)
            
        Maps = [f"{DIR}/TrainingData/{self.object_1}",
                f"{DIR}/TrainingData/{self.object_2}"]
                
        self.samples = [sum(1 for _ in os.scandir(dir)) for dir in Maps]
        
        return TrainingStatus
    


    def Load_Algorithm(self, DIR):
    
        print(f"Path of a Project: {DIR}")


        
        if self.JSONFILE in os.listdir(DIR) and self.MODEL_NAME in os.listdir(DIR):   #za txt
                
            TrainingStatus = self.Get_Data(DIR, self.JSONFILE)
            self.SavedWithoutAlogorithm = False
            return TrainingStatus

        elif self.JSONFILE in os.listdir(DIR):  #No trained Model
            
            TrainingStatus = self.Get_Data(DIR, self.JSONFILE)
            self.SavedWithoutAlogorithm = True
            return TrainingStatus

        else:
            messagebox.showerror("Error","Couldn't find a file OR Selcted DIR wasn't a project!!")
            self.NoFile = True
            return False
        
        
        

    def Return_Load_Parameters(self):
        if self.SavedWithoutAlogorithm:
            if self.NoFile:  #Če ni najdenga projekta
                return None, self.object_1, self.object_2, self.St_Model
            
            elif self.NoFile == False:
                return self.samples, self.object_1, self.object_2, self.St_Model

        elif self.SavedWithoutAlogorithm == False:
            if self.NoFile:
                return None, self.object_1, self.object_2, self.St_Model
            
            elif self.NoFile == False:
                return self.samples, self.object_1, self.object_2, self.St_Model


#Slike naloži po GUI updatu
    def Load_Images(self, DIR):
        
        shutil.copytree(f"{DIR}/TrainingData/{self.object_1}", "1",dirs_exist_ok=True)
        shutil.copytree(f"{DIR}/TrainingData/{self.object_2}", "2", dirs_exist_ok=True)
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
        self.MODEL = model.Model()  

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

        for i in ["1","2"]:
            for file in os.listdir(i):
                pot = os.path.join(i,file)
                if os.path.isfile(pot): os.unlink(pot)
        
        
    
    def Parallel_Predict(self, img):
        threading.Thread(target=self.Predict, args=[img]).start()

    def Predict(self, img):
        img.thumbnail((150,150), PIL.Image.LANCZOS)
        
        img.save("Predict/frame.jpg")

        self.predictedClass = self.MODEL.Predict_Class()



        
        

    