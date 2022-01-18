import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.spatial import distance as dis
import datetime

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mpPose = mp.solutions.pose
Pose = mpPose.Pose(min_detection_confidence=0.7,
               min_tracking_confidence=0.7)
mpHand = mp.solutions.hands
Hand = mpHand.Hands()

def point_xy(img1, point1):
    h, w = img1.shape[:2]
    p1_x = int(point1.x * w)
    p1_y = int(point1.y * h)
    return p1_x, p1_y


def dist2p(img1, point1, point2):
    h, w = img1.shape[:2]
    p1_x = int(point1.x * w)
    p1_y = int(point1.y * h)
    p1_pix = p1_x, p1_y
    p2_x = int(point2.x * w)
    p2_y = int(point2.y * h)
    p2_pix = p2_x, p2_y
    if 0 > p1_x or p1_x > 640 or 0 > p2_x or p2_x > 640 or 0 > p1_y or p1_y > 480 or 0 > p2_y or p2_y > 480:
        dist = 0
    else:
        dist = dis.euclidean(p1_pix, p2_pix)
    return dist


def ratio2dist(img1, point1, point2, point3, point4):
    h, w = img1.shape[:2]
    ratioP2P = 0
    p1_pix = int(point1.x * w), int(point1.y * h)
    p2_pix = int(point2.x * w), int(point2.y * h)
    dist_p1p2 = dis.euclidean(p1_pix, p2_pix)
    p3_pix = int(point3.x * w), int(point3.y * h)
    p4_pix = int(point4.x * w), int(point4.y * h)
    dist_p3p4 = dis.euclidean(p3_pix, p4_pix)
    if dist_p3p4 > 0:
        ratioP2P = round(dist_p1p2 / dist_p3p4, 2)
    return ratioP2P


def nothing(xa):
    alpha = xa


drowsyIcon = cv2.imread('ddaw_icons/sleepy-driver.png')
size = 100
drowsyIcon = cv2.resize(drowsyIcon, (size, size))
drowsyIcon2gray = cv2.cvtColor(drowsyIcon, cv2.COLOR_BGR2GRAY)
ret, drowsyIconMask = cv2.threshold(drowsyIcon2gray, 1, 255, cv2.THRESH_BINARY) # Create a mask for drowsy icon

phoneIcon = cv2.imread('ddaw_icons/no-phone.png')
size = 100
phoneIcon = cv2.resize(phoneIcon, (size, size))
phoneIcon2gray = cv2.cvtColor(phoneIcon, cv2.COLOR_BGR2GRAY)
ret2, phoneIconMask = cv2.threshold(phoneIcon2gray, 1, 255, cv2.THRESH_BINARY) # Create a mask for phone icon

smokeIcon = cv2.imread('ddaw_icons/no-smoke.png')
size = 100
smokeIcon = cv2.resize(smokeIcon, (size, size))
smokeIcon2gray = cv2.cvtColor(smokeIcon, cv2.COLOR_BGR2GRAY)
ret3, smokeIconMask = cv2.threshold(smokeIcon2gray, 1, 255, cv2.THRESH_BINARY) # Create a mask for smoke icon

attentionIcon = cv2.imread('ddaw_icons/attention.png')
size = 100
attentionIcon = cv2.resize(attentionIcon, (size, size))
attentionIcon2gray = cv2.cvtColor(attentionIcon, cv2.COLOR_BGR2GRAY)
ret3, attentionIconMask = cv2.threshold(attentionIcon2gray, 1, 255, cv2.THRESH_BINARY) # Create a mask for smoke icon
# RightLip: 62 LeftLip: 307
cap = cv2.VideoCapture(0)
# cap.set(3, 320)
# cap.set(4, 240)

cv2.namedWindow('DDAW')
cv2.createTrackbar('Min', 'DDAW', 0, 468, nothing)
cv2.createTrackbar('Max', 'DDAW', 0, 468, nothing)

startTime = 0
writeOnceSmoke = 0
writeOncePhone = 0
writeOnceDrowsy = 0
writeOnceAttention = 0

drowsyFrame = 0
drowsyCount = 0
awakeFrame = 0

phoneFrame = 0
phoneCount = 0
woPhoneFrame = 0

smokeFrame = 0
smokeCount = 0
woSmokeFrame = 0

distPx_IFT2L = 0
distPx_IFT2RE = 0
distPx_IFT2LE = 0
distPx_LIL2LOL = 0

while cap.isOpened():
    globalTime = time.time()
    timeStamp = datetime.datetime.fromtimestamp(globalTime).strftime('%d-%m-%Y_%H:%M:%S')
    logTimeStamp = datetime.datetime.fromtimestamp(globalTime).strftime('%d-%m-%Y %H:%M:%S')

    success, img = cap.read()
    imgRGB = img.copy()
    height, width = img.shape[:2]
    imgRGB.flags.writeable = False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceMeshResults = faceMesh.process(imgRGB)
    poseResults = Pose.process(imgRGB)
    handResults = Hand.process(imgRGB)
    imgRGB.flags.writeable = True
    #Yuzdeki noktaları gormek icin
    minVal = cv2.getTrackbarPos('Min', 'DDAW')
    maxVal = cv2.getTrackbarPos('Max', 'DDAW')

    if faceMeshResults.multi_face_landmarks:
        # 468-473 Right Irıs
        # 473-478 Left Iris
        for i in range(minVal, maxVal):
            fLms = faceMeshResults.multi_face_landmarks[0].landmark[i]
            x = int(fLms.x * width)
            y = int(fLms.y * height)
            cv2.circle(img, (x, y), 1, (100, 100, 0), -10)
            # cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
            #print(i, x, y)
        upperLip = faceMeshResults.multi_face_landmarks[0].landmark[13]
        bottomLip = faceMeshResults.multi_face_landmarks[0].landmark[14]
        rightLip = faceMeshResults.multi_face_landmarks[0].landmark[61]
        leftLip = faceMeshResults.multi_face_landmarks[0].landmark[306]
        lipRatio = ratio2dist(img, leftLip, rightLip, upperLip, bottomLip)
        # Right Lid: Bottom: 145 Upper: 159 Outer: 130 Inner: 133
        # Left Lid: Bottom: 373 Upper: 387 Inner: 382 Outer: 359
        leftUpperLid = faceMeshResults.multi_face_landmarks[0].landmark[387]
        leftBottomLid = faceMeshResults.multi_face_landmarks[0].landmark[373]
        leftInnerLid = faceMeshResults.multi_face_landmarks[0].landmark[382]
        leftOuterLid = faceMeshResults.multi_face_landmarks[0].landmark[359]
        leftLidRatio = ratio2dist(img, leftInnerLid, leftOuterLid, leftUpperLid, leftBottomLid)
        rightUpperLid = faceMeshResults.multi_face_landmarks[0].landmark[387]
        rightBottomLid = faceMeshResults.multi_face_landmarks[0].landmark[373]
        rightInnerLid = faceMeshResults.multi_face_landmarks[0].landmark[382]
        rightOuterLid = faceMeshResults.multi_face_landmarks[0].landmark[359]
        rightLidRatio = ratio2dist(img, rightInnerLid, rightOuterLid, rightUpperLid, rightBottomLid)

        distPx_LIL2LOL = dist2p(img, leftInnerLid, leftOuterLid) #Distance from left inner lid to left outerlid

        # cv2.putText(img, str((leftLidRatio+rightLidRatio)/2), (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        # cv2.putText(img, 'Uyuyorsun', (40, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        if 0 < lipRatio < 10 or (leftLidRatio + rightLidRatio) / 2 > 4:
            drowsyFrame += 1
            awakeFrame = 0
            if drowsyFrame > 30:
                drowsyCount += 1
                drowsyFrame = 0
        else:
            awakeFrame += 1

        # print(drowsyFrame, awakeFrame, lipRatio, (leftLidRatio+rightLidRatio)/2)

        #AWAKE FRAME IS BIGGER THAN LIMIT DRIVER IS COMPLETE AWAKE
        if awakeFrame > 300:
            drowsyFrame = 0
            drowsyCount = 0

        if drowsyCount > 5:
            roiDrowsy = img[-size - 370:-370, -size - 10:-10]
            roiDrowsy[np.where(drowsyIconMask)] = 0
            roiDrowsy += drowsyIcon

        # cv2.putText(img, f'Drowsiness Count: {int(drowsyCount)}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        # ATTENTION ICON ON SCREEN
        # roiAttention = img[-size - 40:-40, -size - 10:-10]
        # roiAttention[np.where(attentionIconMask)] = 0
        # roiAttention += attentionIcon

    # Ekranda POSE Gösterme
    # mpDraw.draw_landmarks(img, poseResults.pose_landmarks, mpPose.POSE_CONNECTIONS,
    #                       mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
    #                       mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


    # if handResults.multi_hand_landmarks and faceMeshResults.multi_face_landmarks:
    #     for i in range(0, 20):
    #         pLms = handResults.multi_hand_landmarks[0].landmark[i]
    #         x = int(pLms.x * width)
    #         y = int(pLms.y * height)
    #         cv2.circle(img, (x, y), 1, (100, 100, 0), -10)
    #         cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)



    ##########################
    if poseResults.pose_landmarks and faceMeshResults.multi_face_landmarks and handResults.multi_hand_landmarks:
        # for i in range(0, 20):
        #     pLms = poseResults.pose_landmarks.landmark[i]
        #     x = int(pLms.x * width)
        #     y = int(pLms.y * height)
        #     cv2.circle(img, (x, y), 1, (100, 100, 0), -10)
        #     cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        #     print(i, x, y)


        # PHONE DETECTION
        rightEar = faceMeshResults.multi_face_landmarks[0].landmark[93]
        leftEar = faceMeshResults.multi_face_landmarks[0].landmark[323]
        leftWrist = poseResults.pose_landmarks.landmark[15]
        rightWrist = poseResults.pose_landmarks.landmark[16]
        leftIndex = poseResults.pose_landmarks.landmark[19]
        rightIndex = poseResults.pose_landmarks.landmark[20]
        indexFingerTip = handResults.multi_hand_landmarks[0].landmark[8]
        bottomLip = faceMeshResults.multi_face_landmarks[0].landmark[14]
        leftIndex = poseResults.pose_landmarks.landmark[19]
        rightIndex = poseResults.pose_landmarks.landmark[20]

        distPx_RI2RE = dist2p(img, rightIndex, rightEar)
        distPx_LI2LE = dist2p(img, leftIndex, leftEar)
        distPx_LE2RE = dist2p(img, leftEar, rightEar)
        distPx_IFT2RE = dist2p(img, indexFingerTip, rightEar)
        distPx_IFT2LE = dist2p(img, indexFingerTip, leftEar)
        distPx_IFT2L = dist2p(img, indexFingerTip, bottomLip)
        distPx_RI2L = dist2p(img, bottomLip, rightIndex)
        distPx_LI2L = dist2p(img, bottomLip, leftIndex)

    else:
        distPx_RI2RE = 0
        distPx_LI2LE = 0
        distPx_LE2RE = 0
        distPx_IFT2RE = 0
        distPx_IFT2LE = 0
        distPx_IFT2L = 0
        distPx_RI2L = 0
        distPx_LI2L = 0

    #SMOKE DETECT
    if (distPx_IFT2L < distPx_IFT2RE or distPx_IFT2L < distPx_IFT2LE) and (distPx_IFT2L < distPx_LIL2LOL):
        smokeFrame += 1
        woSmokeFrame = 0
        if smokeFrame > 20:
            smokeCount += 1
            smokeFrame = 0
    else:
        woSmokeFrame += 1

    # WITHOUT SMOKE FRAME IS BIGGER THAN LIMIT DRIVER IS NOT TALKING
    if woSmokeFrame > 80:
        smokeFrame = 0
        smokeCount = 0
        writeOnceSmoke = 0

    if smokeCount > 2:
        roiSmoke = img[-size - 150:-150, -size - 10:-10]
        roiSmoke[np.where(smokeIconMask)] = 0
        roiSmoke += smokeIcon
        if writeOnceSmoke == 0:
            writeOnceSmoke = 1
            fileName = "smokeBreach_" + str(timeStamp)
            filePath = "breachs/smoke/" + fileName + ".jpg"
            cv2.imwrite(filePath, img)
            breachtext = "Smoke Breach!"
            log = 'breachs/breachs.txt'
            logfile = open(log, "a")
            with open(log, 'a') as logfile:
                logfile.write(logTimeStamp+ " " + breachtext + '\n')

    #PHONE DETECT
    if ((0 < distPx_IFT2RE < distPx_IFT2L) and distPx_IFT2RE < distPx_LIL2LOL) or (
            (0 < distPx_IFT2LE < distPx_LIL2LOL) and distPx_IFT2LE < distPx_IFT2L):
        phoneFrame += 1
        woPhoneFrame = 0
        if phoneFrame > 20:
            phoneCount += 1
            phoneFrame = 0
    else:
        woPhoneFrame += 1
    # print(distPx_RI2E, distPx_LI2E, distPx_LE2RE, "   " ,phoneFrame, woPhoneFrame)

    # WITHOUT PHONE FRAME IS BIGGER THAN LIMIT DRIVER IS NOT TALKING
    if woPhoneFrame > 80:
        phoneFrame = 0
        phoneCount = 0
        writeOncePhone = 0
    if phoneCount > 2:
        roiPhone = img[-size - 260:-260, -size - 10:-10]
        roiPhone[np.where(phoneIconMask)] = 0
        roiPhone += phoneIcon
        if writeOncePhone == 0:
            writeOncePhone = 1
            fileName = "phoneBreach_" + str(timeStamp)
            filePath = "breachs/phone/" + fileName + ".jpg"
            cv2.imwrite(filePath, img)
            breachtext = "Phone Breach!"
            log = 'breachs/breachs.txt'
            logfile = open(log, "a")
            with open(log, 'a') as logfile:
                logfile.write(logTimeStamp + " " + breachtext + '\n')


    # print(round(distPx_IFT2L,2), round(distPx_IFT2RE,2), round(distPx_IFT2LE,2), round(distPx_LIL2LOL,2), "SMOKE: ", smokeFrame, woSmokeFrame, "PHONE: ",
    #       phoneFrame, woPhoneFrame)
    print("SMOKE: ", smokeFrame, woSmokeFrame, " PHONE:", phoneFrame, woPhoneFrame, " DROWSY:", drowsyFrame, " AWAKE:", awakeFrame)
    cTime = time.time()
    fps = 1 / (cTime - startTime)
    startTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(img, f'HxW: {int(height), int(width)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("DDAW", cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
