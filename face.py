from deepface import DeepFace
from retinaface import RetinaFace
import matplotlib.pyplot as pl
import os
import pandas as ps
import json


unknownFaces=[]

recognizedNumbers=[] 


def get_photos(folderPath):

    photos = []

    for filename in os.listdir(folderPath):

        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):

            photoPath = os.path.join(folderPath, filename)

            photos.append(photoPath)

    return photos


# Get Photos and Check Them

TrainPhotosFolder = "./Train"

TestPhotosFolder='./Test'

photoPaths = get_photos(TestPhotosFolder)

if photoPaths:
    for photoPath in photoPaths:

        inputFaces = RetinaFace.extract_faces(photoPath)

        # print(len(inputFaces) , ' faces')

        for face in inputFaces:

            pl.imshow(face) 

            pl.savefig('detectedFace.jpg')

            pl.axis('off')
            
            obj = DeepFace.find(img_path="detectedFace.jpg", db_path = "./dataset",enforce_detection=False)

            try:

                faceNumber=int((obj[0]['identity'][0]).split('\\')[1])

                print(faceNumber)
                
                recognizedNumbers.append(faceNumber)

                # pl.title(faceNumber)

            except:

                pl.title("Did not Recognize The Face")

                unknownFaces.append(face)

            # pl.show(block=False)

            # pl.pause(5)

            # pl.close()

else:
    print("No photo files found in the folder.")


# Clean + Save Data to Excel

os.remove('./detectedFace.jpg')

AttendanceData = ps.DataFrame({"ارقام الجلوس":recognizedNumbers})

AttendanceData.to_excel('Attendance.xlsx',index=False)

data = { "id": recognizedNumbers}

jsonData = json.dumps(data)

with open("data.json", "w") as outfile:
    outfile.write(jsonData)
    
    