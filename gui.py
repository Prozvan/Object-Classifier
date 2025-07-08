from customtkinter import CTkRadioButton, END
from tkinter import Tk, Canvas,Label, Entry, StringVar, Button, filedialog, PhotoImage, simpledialog, messagebox

import time
import camera
import app
import PIL.Image, PIL.ImageTk
import threading

class GUI:
    def __init__(self):
        print("Object Recognition\n")
        self.CAMERA = camera.Kamera()
        self.APP = app.App()
        
        #model.py kličem samo iz app.py!!, nemorem iz dveh hkrati!!


        self.DELAY = 10
        self.samples = self.APP.Number_Of_Samples() 
        
        self.isAlgorithmSelected = False
        self.TrainingStatus = False
        self.autoON = False
        self.ToSave = False #Za savat

        self.beforeKey = 0
        self.beforeChoice = 0

        self.object_1 = "1"
        self.object_2 = "2"


        #Settings for GUI in tkinter
        self.WINDOW = Tk()
        self.WINDOW.title("Object Recognition")
        self.WINDOW.geometry("1100x665")
        self.WINDOW.configure(bg = "#C0D2D6")

        self.Components()


        #Hot keys
        self.WINDOW.bind("<F1>", lambda e: self.Pressed_Key(1))
        self.WINDOW.bind("<F2>", lambda e: self.Pressed_Key(2))

        self.WINDOW.bind("<F3>", lambda e: self.Prediction(False))  #Predict
        self.WINDOW.bind("<F4>", lambda e: self.Prediction(True))  #AutoPredict

        self.WINDOW.bind("<F5>", lambda e: self.Auto_Parallel(1))  #AutoPredict
        self.WINDOW.bind("<F6>", lambda e: self.Auto_Parallel(2))  #AutoPredict

        self.WINDOW.bind("<KeyRelease-F1>",lambda e: self.Pressed_Key(11))
        self.WINDOW.bind("<KeyRelease-F2>",lambda e: self.Pressed_Key(22))
        self.WINDOW.bind("<Return>", self.Replace_Object_Name)


        self.WINDOW.resizable(False, False)
        # self.WINDOW.attributes("-topmost", True)

        self.Update()
        self.WINDOW.protocol("WM_DELETE_WINDOW", self.Save_Project)
        self.WINDOW.mainloop()   #Loop




    def Components(self):

        def Get_Asset(item): return "Visual/" + item


        self.canvas = Canvas(
            self.WINDOW,
            bg = "#C0D2D6",
            height = 665,
            width = 1100,
            bd = 0,
            highlightthickness = 0,
        )
        self.canvas.place(x = 0, y = 0)




        #BUTTONS
        self.BUTTON = PhotoImage(file=Get_Asset("button.png"))
        BUTTON_SIZE_W = 135.0
        BUTTON_SIZE_H = 65.0 
        SIZE = 14

        #Dictionary
        BUT_params = {
            "font": ("Inter", SIZE),  # Set font with size and style
            "fg":"white",
            "activeforeground": "white",
            "compound": "center",   #Slika, text skp
            "image":self.BUTTON,
            "borderwidth" : 0,    
            "highlightthickness" : 0,   
            "activebackground" : "#C0D2D6",
            "background" : "#C0D2D6",
            "cursor" : "hand2", 
            "relief" : "flat"  #removes type border
        }

        BUT_SIZE_params = {
            "width":BUTTON_SIZE_W, "height":BUTTON_SIZE_H
        }



        btn_load = Button(
            text="Load",
            command= self.Parallel_Load,        #brez (), k je ne zažene na začetku
            **BUT_params)

        btn_load.place(
            x=771.0, y=83.0,
            **BUT_SIZE_params)


        btn_save = Button(
            text="Save",
            command= self.Save,
            **BUT_params)

        btn_save.place(
            x=917.0, y=83.0,
            **BUT_SIZE_params
        )


        btn_predict = Button(
            text="Predict",
            command= lambda: self.Prediction(False),
            **BUT_params)
        
        btn_predict.place(
            x=771.0, y=156.0,
            **BUT_SIZE_params)


        self.btn_autoPredict = Button(
            text="Auto Predict",
            command=lambda: self.Prediction(True),
            **BUT_params)

        self.btn_autoPredict.place(
            x=917.0, y=156.0,
            **BUT_SIZE_params)


        btn_reset = Button(
            text="Reset",
            command= self.Reset,
            **BUT_params)
        
        btn_reset.place(
            x=771.0,
            y=229.0,
            **BUT_SIZE_params
        )


        btn_train = Button(
            text="Train",
            command=lambda: self.Train_Algorithm(),
            **BUT_params)
        
        btn_train.place(
            x=917.0, y=229.0,
            **BUT_SIZE_params)


        self.btn_Object1 = Button(
            text="1",
            command=lambda: self.Button_Click(1),
            **BUT_params)
        
        self.btn_Object1.place(
            x=771.0, y=302.0,
            **BUT_SIZE_params)


        self.btn_Object2 = Button(
            text="2",
            command=lambda: self.Button_Click(2),
            **BUT_params)

        self.btn_Object2.place(
            x=917.0, y=302.0,
            **BUT_SIZE_params)




        #LABELS
        self.numSamples = Label(
            master=self.canvas, text=self.samples, font=("Inter", -50),
            background="#C0D2D6")
        self.numSamples.place(
            x=385.0, y=13.0, anchor="n"
        )


        self.labelClass = Label(
            master=self.canvas, text="Object", font=("Inter", -50),
            background="#C0D2D6", anchor="center", justify="center")
        self.labelClass.place(
            x=385.0, y=647.0, anchor="s"
        )
        

        #RADIO BUTTONS
        selectedModel = StringVar()

        RAD_params = {
            "master" : self.canvas,
            "width":100.0, 
            "height":32.0, 
            "radiobutton_height":26,
            "radiobutton_width": 26, 
            "text_color": ("#000000"),
            "border_color":"#0C5663",
            "font":("Inter", 17),
            "variable":selectedModel, 
        }

        self.btn_LSVC = CTkRadioButton( **RAD_params,
            text="Linear Support Vector Classifier", 
            command=lambda: self.Select_Algorithm(3))
        
        self.btn_LSVC.place(x=771.0, y=450.0)


        self.btn_NN = CTkRadioButton( **RAD_params,
            text="Neural Network", 
            command=lambda: self.Select_Algorithm(2))
        
        self.btn_NN.place(x=771.0, y=491.0)


        self.btn_KNc = CTkRadioButton( **RAD_params,
            text="K-Nearest Neighbors Classifier",
            command=lambda: self.Select_Algorithm(1))
        
        self.btn_KNc.place(x=771.0, y=531.0)


        #ENTRY
        self.canvas.create_rectangle(
            771.0, 384.0,
            1050.0, 433.0,
            fill="#C0D2D6",  
            outline="")


        self.entry_image = PhotoImage(file=Get_Asset("entry.png"))
        self.canvas.create_image(
            910.5, 409.5,
            image=self.entry_image)
        
        self.entry = Entry(
            bd=0, bg="#7CAFB7",
            font=("Inter", SIZE), highlightthickness=0)
        
        self.entry.place(
            x=785.0, y=385.0,
            width=251.0, height=49.0
        )


        #STATUS LABEL
        self.status_Train = Label(
            master=self.canvas, text="Status: UnTrained", font=("Inter", -28),
            background="#C0D2D6")
        
        self.status_Train.place(x=771.0, y=598.0)



    def Replace_Object_Name(self, bin):
        self.autoON = False

        #Gets value from Entry
        value = str(self.entry.get())

        #Deletes it from GUI
        self.entry.delete(0, END)

        #Spits it by a space " "
        value = value.split(" ", 1)

        if "Epochs" == value[0]:
            try: 
                epochs = int(value[1])
                if epochs > 30:
                    epochs = 30
                    
                self.APP.Change_Epochs(epochs)
                print(f"Changed Epochs to {epochs}")

            except ValueError as e:
                print("TypeError: 'Epochs x' -> x is a number!!")
                
            except IndexError as e:
                # print("IndexError: Epochs x -> x is a number!!")
                pass            


        else:
            #Gets an input name and if its in the right order
            name, objec = self.APP.Object_Name(value)

            if objec == 1: 
                self.btn_Object1.configure(text=name)
                self.object_1 = name
                self.ToSave = True

            elif objec == 2: 
                self.btn_Object2.configure(text=name)
                self.object_2 = name
                self.ToSave = True
            
            




    def Button_Click(self, key):  #Buttons
        self.canvas.focus_set()
        self.autoON = False
        self.ToSave = True

        
        self.APP.Button_Click(key, self.picture)
        self.samples = self.APP.samples
        self.numSamples.configure(text = self.samples)

  
    def Pressed_Key(self, key):  #KEYS F1 F2
        self.canvas.focus_set()
        self.ToSave = True
        

        if key != self.beforeKey:  #Nič tiščanja
            self.autoON = False
            
            self.APP.Key_Pressed(key, self.picture)
            self.samples = self.APP.samples
            self.numSamples.configure(text = self.samples)
        
        self.beforeKey = key  


    def Auto_Parallel(self, object):
        threading.Thread(target=self.Auto, args=[object]).start()

    def Auto(self, object):
        time.sleep(1.5)
        for i in range(100):
            time.sleep(0.14)
            # print(f"Slika {i+1}")
            self.APP.Button_Click(object, self.picture)
            self.samples = self.APP.samples
            self.numSamples.configure(text = self.samples)

        

    def Select_Algorithm(self, choice):
        self.canvas.focus_set()
        self.autoON = False
        print()
        
        #Doesn't work when the same Button is pressed multiple times
        if choice != self.beforeChoice:  
            self.isAlgorithmSelected = True
            self.TrainingStatus = False
            self.ToSave = True

            self.status_Train.configure(text="Status: UnTrained")
            self.labelClass.configure(text="Object")

            self.APP.Select_Algorithm(choice)
            self.beforeChoice = choice
            

    
    def Train_Algorithm(self):
        self.canvas.focus_set()
        self.autoON = False

        if self.isAlgorithmSelected: #izbran algoritem
            
            #For safety Reason
            self.samples = self.APP.Number_Of_Samples()
            self.numSamples.configure(text=self.samples)

            #For algorithms's bottom preformance
            if self.samples[0] >= 10 and self.samples[1] >= 10:
                
                self.status_Train.configure(text="Status: UnTrained")
                self.labelClass.configure(text="Object")
                
                #When training is finished, it's true
                self.TrainingStatus = False 
                self.ToSave = True 
                

                print("Started Training")

                self.APP.Parallel_Train_Algorithm()
                
            else:
                messagebox.showerror("Error", "At least 10 pictures in every object")
            
            
    def Prediction(self, option):
        self.canvas.focus_set()

        # print(self.TrainingStatus) #Vse vredu
        if self.TrainingStatus:  #Has to be trained
            
            if option: #Auto Predict
        
                if self.autoON: self.autoON = False  #Stop Auto Predict
                else: 
                    self.autoON = True   #Start Auto Predict
                     
                print(f"Auto Predict: {self.autoON}")

            else:  #Predict
                self.autoON = False
                self.Predict()
                print("Predict")


    def Predict(self): self.APP.Parallel_Predict(self.pictureARR)


    def Save(self):
        self.canvas.focus_set()
        self.autoON = False
        
        print("Save")

        #Name of a project folder
        PROJECT = simpledialog.askstring("Project name",
                            "Enter project name", parent=self.WINDOW)
        

        if PROJECT != None:  #User ni nič napisal / cancel
                
            self.APP.Parallel_Save_Algorithm(self.object_1, self.object_2, self.TrainingStatus, PROJECT)
            self.ToSave = False


    def Parallel_Load(self):
        threading.Thread(target=self.Load).start()
        # self.Load()

    def Load(self):
        self.canvas.focus_set()
        self.autoON = False
        print("Load")

        #From where to load?
        DIR = filedialog.askdirectory(parent=self.WINDOW, 
                                    initialdir=r"Projects\\", 
                                    title="Select a project directory")  
        
        if DIR != "":
            #Start up
            self.TrainingStatus = self.APP.Load_Algorithm(DIR)
            
            #Parameters
            samples, object_1, object_2, modelIndex = self.APP.Return_Load_Parameters()
            
            #If None - Ni projekta
            if samples is not None: 
                self.ToSave = True
                
                #Setting all the parameters
                self.samples = samples
                self.object_1 = object_1
                self.object_2 = object_2 
    
                self.ToSave = False

                if modelIndex != 0:  #If model is chossen
                    self.isAlgorithmSelected = True
                    print("Training Status",self.TrainingStatus)
                    # if self.TrainingStatus:
                    #     self.status_Train.configure(text="Status: Trained")
                    

                
                #All the visuals
                self.numSamples.configure(text=self.samples)
                
                self.labelClass.configure(text="Object")

                self.btn_Object1.configure(text=self.object_1)
                self.btn_Object2.configure(text=self.object_2)


                #Radio buttons
                if modelIndex == 1: self.btn_KNc.select()
                elif modelIndex == 2: self.btn_NN.select()
                elif modelIndex == 3: self.btn_LSVC.select()
                elif modelIndex == 0:
                    self.btn_KNc.deselect()
                    self.btn_NN.deselect()
                    self.btn_LSVC.deselect()
            


                #
                self.beforeChoice = modelIndex

                #After GUI updates
                self.APP.Load_Images(DIR)
                print()
                


    def Reset(self):
        self.canvas.focus_set()
        self.ToSave = False
        self.autoON = False
        self.isAlgorithmSelected = False
        self.TrainingStatus = False
        self.beforeChoice = 0

        self.isAlgorithmSelected = False
        self.TrainingStatus = False
        self.autoON = False
        self.ToSave = False #Za savat

        self.beforeKey = 0
        self.beforeChoice = 0

        self.samples = [0,0]
        self.object_1 = "1"
        self.object_2 = "2"

        self.btn_Object1.configure(text="1")
        self.btn_Object2.configure(text="2")

        self.labelClass.configure(text="Object")
        self.status_Train.configure(text="Status: UnTrained")
        self.numSamples.configure(text="0 0")

        self.btn_NN.deselect()
        self.btn_LSVC.deselect()
        self.btn_KNc.deselect()

        self.APP.Reset_Project()


    def Save_Project(self):
        self.autoON = False
        if self.ToSave:   #if the project wasn't saved yet

            choice = messagebox.askyesnocancel("Save", "Do you want to Save?")
            
            
            if choice:   #User wants to save
                
                self.Save()
                self.Reset()
                self.WINDOW.destroy()
                print(f"The end, {choice}")


            elif choice == False:  #
               
                self.Reset()
                self.WINDOW.destroy()
                print(f"The end, {choice}")


        else:   #If it was saved
            print("The end")
            self.Reset()
            self.WINDOW.destroy()



    def Update(self):

        #Geting image
        status, self.picture = self.CAMERA.Get_Image()
        
        #If an algorithm is trained
        if self.APP.TrainingStatus:
            self.TrainingStatus = True
            self.status_Train.configure(text="Status: Trained")
        else:
            self.TrainingStatus = False
            self.status_Train.configure(text="Status: UnTrained")
        


        #Prepares the image
        if status:   
            self.pictureARR = PIL.Image.fromarray(self.picture)
            self.slika = PIL.ImageTk.PhotoImage(image=self.pictureARR)
        

        

        #Auto predict
        if self.autoON: self.Predict()   
 
        #Which object is on the camera (Prediction)
        if self.APP.predictedClass == 0: 
            self.labelClass.configure(text=self.object_1)
            self.APP.predictedClass = -1

        elif self.APP.predictedClass == 1:
            self.labelClass.configure(text=self.object_2)
            self.APP.predictedClass = -1
        
        

        #Shows an image
        self.canvas.create_image(385.0, 323.0, image=self.slika)  #video


        #Updates its self after 10ms
        self.WINDOW.after(self.DELAY, self.Update)


