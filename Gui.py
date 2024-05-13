import tkinter as tk
from tkinter import *
from tkinter import filedialog,messagebox
import os
import shutil
import cv2
from deepface import DeepFace
from retinaface import RetinaFace
import matplotlib.pyplot as pl
import pandas as ps


class app:
    def __init__(self, main):
        self.main = main
        self.main.geometry("800x600")
        self.unknownFaces=[]
        self.recognizedNumbers=[]
        self.TrainPhotosFolder = "./Train"
        self.TestPhotosFolder='./Test'
        self.uploadFaces()
    # sara
    def uploadVideos(self):
        for i in self.main.winfo_children():
            i.destroy()
            
        self.frame1 = Frame(self.main,  width=600, height=400, bg="lightblue")
        self.frame1.pack(padx=30, pady=25)
        
        self.label = tk.Label(self.frame1, text='Student Videos', font=("Arial", 16, "bold"), fg="black",width=50)
        self.label.pack(pady=25)
        
        self.uploadFacesLinks = tk.Button(self.frame1, text="Go to Upload Faces Imgs", command=self.uploadFaces, font=("Arial", 12, "bold"), fg="white", bg="purple", width=25)
        self.uploadFacesLinks.pack(pady=25)
        
        self.uploadVideosBtn=tk.Button(self.frame1, text="Upload Videos", command=self.upload,font=("Arial", 12, "bold"), fg="white", bg="royalblue", width=20)
        self.uploadVideosBtn.pack(pady=25)
        
        self.extractFramesBtn=tk.Button(self.frame1, text="Extract Frames", command=self.getFrames, font=("Arial", 12, "bold"), fg="white", bg="forestgreen", width=20)
        self.extractFramesBtn.pack(pady=25)
    
    # sara
    def uploadFaces(self):
        for i in self.main.winfo_children():
            i.destroy()
        
        self.frame2 = Frame(self.main,  width=600, height=400, bg="lightblue")
        self.frame2.pack(padx=30, pady=25)
        
        self.label = tk.Label(self.frame2, text='Student Faces Imgs', font=("Arial", 16, "bold"), fg="black",width=50)
        self.label.pack(pady=25)
        
        self.uploadVideos_btn = tk.Button(self.frame2, text="Go to Upload Student Videos", command=self.uploadVideos,font=("Arial", 12, "bold"), fg="white", bg="purple", width=25)
        self.uploadVideos_btn.pack(pady=25)
        
        self.uploadImgBtn = tk.Button(self.frame2, text="Upload Image", command=self.upload,font=("Arial", 12, "bold"), fg="white", bg="royalblue", width=20)
        self.uploadImgBtn.pack(pady=25)
        
        self.testBtn = tk.Button(self.frame2, text="Test Model", command=self.TestImgs,font=("Arial", 12, "bold"), fg="white", bg="blue", width=20)
        self.testBtn.pack(pady=25)
        
        self.trainBtn = tk.Button(self.frame2, text="Train Model (Optional)", command=self.TrainModel,font=("Arial", 12, "bold"), fg="white", bg="orange", width=20)
        self.trainBtn.pack(pady=25)
        
    # mostafa
    def upload(self):
        
        filePaths = filedialog.askopenfilenames(title="Select Files", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.wmv"),("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        
        for filePath in filePaths:
            if filePath.lower().endswith((".mp4", ".avi", ".mkv", ".wmv")): 
                save_dir = "./videos"
            
            elif  filePath.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                save_dir = "./Test"
                
            else:
                messagebox.showerror('File Error', 'Error: Try Again')
                continue
            
            if filePath:
                if not os.path.exists(save_dir):
                    
                    os.makedirs(save_dir)
                    
                filename = os.path.basename(filePath)
                
                shutil.copy(filePath, os.path.join(save_dir, filename))
                
                messagebox.showinfo("Success", "File uploaded successfully!")
            
    # sandy
    def getVideoPaths(self,folderPath):

        videos = []

        for filename in os.listdir(folderPath):
            
            if filename.endswith((".mp4", ".avi", ".mkv", ".wmv")): 

                videoPath = os.path.join(folderPath, filename)

                videos.append(videoPath)

        return videos

    def extractFrames(self,videoPath, outputFolder,check_existing=True, fbs=None):

        # Check or Create output folder
        
        if not os.path.exists(outputFolder):
            
            os.makedirs(outputFolder)
            
        if check_existing:

            videoName =os.path.splitext(os.path.basename(videoPath))[0]

            identifier = f"{videoName}_{3}fps" # 3004_3fbs

            isFileExist = os.path.join(outputFolder, f"{identifier}.txt")

            if os.path.exists(isFileExist):
                return

        cap = cv2.VideoCapture(videoPath)  # Open the video

        if fbs is None:

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:

                fps = 100 # default Value


        frameInterval = 3 / fps # 3 Frames per Second

        count = 0
        
        elapsedTime = 0
        
        while True:

            ret, frame = cap.read()

            if not ret:

                break  # Reached end of video

            if elapsedTime  >= count:

                filename = f"{os.path.splitext(os.path.basename(videoPath))[0]}_{count:0d}.jpg" # 3004_01.jpg

                cv2.imwrite(os.path.join(outputFolder, filename), frame)

                count += 1

            elapsedTime += frameInterval

        cap.release()  # Release video capture object

        if check_existing:
            with open(isFileExist, 'w') as file:
                file.write("Frames extracted")
                

    def getFrames(self):
        videosFolder = "./videos"

        videoPaths = self.getVideoPaths(videosFolder)

        if videoPaths:
            for videoPath in videoPaths:

                outputFolder = os.path.splitext(f"./dataset/{os.path.basename(videoPath)}")[0]

                self.extractFrames(videoPath, outputFolder)
                
            messagebox.showinfo("Success", "Frames Extracted successfully!")

        else:
            print("No video files found in the folder.")
    
    # sandy
    def getPhotos(self,folderPath):

        photos = []

        for filename in os.listdir(folderPath):

            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):

                photoPath = os.path.join(folderPath, filename)

                photos.append(photoPath)

        return photos
    #Ali
    def TrainModel(self):
        
        photoPaths = self.getPhotos(self.TrainPhotosFolder)
        
        if photoPaths:
            
            for photoPath in photoPaths:

                inputFaces = RetinaFace.extract_faces(photoPath)

                for face in inputFaces:

                    pl.imshow(face) 

                    pl.savefig('detectedFace.jpg')

                    pl.axis('off')
                    
                    obj = DeepFace.find(img_path="detectedFace.jpg", db_path = "./dataset",enforce_detection=False)

                    try:

                        faceNumber=int((obj[0]['identity'][0]).split('\\')[1])

                        self.recognizedNumbers.append(faceNumber)

                    except:

                        self.unknownFaces.append(face)

                    
            messagebox.showinfo("Success", "Train Have Successfully Completed")
            self.display()

        else:
            messagebox.showerror('File Error', 'No photo files found in the folder.')
    #Ali
    def TestImgs(self):
        
        photoPaths = self.getPhotos(self.TestPhotosFolder)
        
        if photoPaths:
            
            for photoPath in photoPaths:

                inputFaces = RetinaFace.extract_faces(photoPath)

                for face in inputFaces:

                    pl.imshow(face) 

                    pl.savefig('detectedFace.jpg')

                    pl.axis('off')
                    
                    obj = DeepFace.find(img_path="detectedFace.jpg", db_path = "./dataset",enforce_detection=False)

                    try:

                        faceNumber=int((obj[0]['identity'][0]).split('\\')[1])

                        print(faceNumber)
                        
                        self.recognizedNumbers.append(faceNumber)

                        pl.title(faceNumber)

                    except:

                        pl.title("Did not Recognize The Face")

                        self.unknownFaces.append(face)

                    pl.show(block=False)

                    pl.pause(5)

                    pl.close()

            messagebox.showinfo("Success", "Test Have Successfully Completed")
            self.display()

        else:
            messagebox.showerror('File Error', 'No photo files found in the folder.')
    
    
    def cleanAndSaveSheet(self):
        
        os.remove('./detectedFace.jpg')

        AttendanceData = ps.DataFrame({"ارقام الجلوس":self.recognizedNumbers})

        AttendanceData.to_excel('Attendance.xlsx',index=False)
                    
        messagebox.showinfo("Success", "Attendance Data Have Successfully Saved")

    def display(self):
        for i in self.main.winfo_children():
            i.destroy()
        
        self.frame3 = Frame(self.main,  width=600, height=400, bg="lightblue")
        self.frame3.pack(padx=50, pady=30)
        
        self.label = tk.Label(self.frame3, text='Student Faces Imgs', font=("Arial", 16, "bold"), fg="black",width=50)
        self.label.pack(pady=25)
        
        idData = "Numbers: " + ", ".join(str(id) for id in self.recognizedNumbers)
        self.ids = tk.Label(self.frame3, text=idData,font=("Arial", 12, "bold"), fg="white", bg="green", width=50)
        self.ids.pack(pady=25)
        
        self.saveData = tk.Button(self.frame3, text="Clean And Save Sheet", command=self.cleanAndSaveSheet,font=("Arial", 12, "bold"), fg="white", bg="green", width=20)
        self.saveData.pack(pady=25)
        
        self.goBackBtn = tk.Button(self.frame3, text="Go Back", command=self.uploadFaces,font=("Arial", 12, "bold"), fg="white", bg="orange", width=20)
        self.goBackBtn.pack(pady=30)
        
        
root = Tk()
root.title('Students Attendance System')    
root.configure(background='#338BA8')
app(root)
root.mainloop()