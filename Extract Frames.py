import cv2
import os


def getVideoPaths(folderPath):

    videos = []

    for filename in os.listdir(folderPath):
        
        if filename.endswith((".mp4", ".avi", ".mkv", ".wmv")): 

            videoPath = os.path.join(folderPath, filename)

            videos.append(videoPath)

    return videos


def extractFrames(videoPath, outputFolder,check_existing=True, fbs=None):

    # Check or Create output folder
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    if check_existing:

        videoName =os.path.splitext(os.path.basename(videoPath))[0]

        identifier = f"{videoName}_{3}fps"

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

            filename = f"{os.path.splitext(os.path.basename(videoPath))[0]}_{count:0d}.jpg"

            cv2.imwrite(os.path.join(outputFolder, filename), frame)

            count += 1

        elapsedTime += frameInterval

    cap.release()  # Release video capture object

    if check_existing:
        with open(isFileExist, 'w') as file:
            file.write("Frames extracted")


# Getting Frames OF Videos 

videosFolder = "./videos"

videoPaths = getVideoPaths(videosFolder)

if videoPaths:
    for videoPath in videoPaths:

        outputFolder = os.path.splitext(f"./dataset/{os.path.basename(videoPath)}")[0]

        extractFrames(videoPath, outputFolder)

else:
    print("No video files found in the folder.")

