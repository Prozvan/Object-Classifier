import cv2 as cv

class Kamera:
    def __init__(self):
        self.CAMERA = None
        external_found = False

        # Try external camera (commonly index 1)
        external_cam = cv.VideoCapture(1, cv.CAP_DSHOW)
        if external_cam.isOpened():
            self.CAMERA = external_cam
            external_found = True
        else:
            external_cam.release()

        # Fall back to internal/default camera (index 0)
        if not external_found:
            self.CAMERA = cv.VideoCapture(0, cv.CAP_DSHOW)

        # Final check
        if not self.CAMERA.isOpened():
            raise ValueError("No working camera found!")

        # Store width and height
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





