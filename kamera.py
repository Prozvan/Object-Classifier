import cv2 as cv

class Kamera:
    def __init__(self):
        # CAMERA = cv.VideoCapture(1) #Prva kamera
        self.CAMERA = cv.VideoCapture(1, cv.CAP_DSHOW)



        if not self.CAMERA.isOpened(): 
            raise ValueError("Camera isn't on!")
        else:
            self.width = self.CAMERA.get(cv.CAP_PROP_FRAME_WIDTH)
            self.height = self.CAMERA.get(cv.CAP_PROP_FRAME_HEIGHT)

          
    def __del__(self):
        if self.CAMERA.isOpened(): self.CAMERA.release()


    def Get_Image(self):
        if self.CAMERA.isOpened():
            stanje, slika_zaslona = self.CAMERA.read()

            if stanje: 
                return(stanje, cv.cvtColor(slika_zaslona, cv.COLOR_BGR2RGB))
            
            else: 
                return(stanje, None)
            
        else: 
            self.CAMERA.release()
            return (False, False)





