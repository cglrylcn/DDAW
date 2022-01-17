import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dis

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True)

cap = cv2.VideoCapture(0)
pTime = 0
while cap.isOpened():
    success, img = cap.read()
    height, width = img.shape[:2]
    imgRGB = img.copy()
    imgRGB.flags.writeable = False
    imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
    # faceMeshResults = faceMesh.process(imgRGB)

    holisticResults = holistic.process(imgRGB)
    imgRGB.flags.writeable = True

    # print(holisticResults.face_landmarks)
    # print(holisticResults.face_landmarks.landmark[0])
    # mpDraw.draw_landmarks(img, holisticResults.face_landmarks, mpHolistic.FACEMESH_CONTOURS,
    #                       mpDraw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #                       mpDraw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    if holisticResults.face_landmarks:
        for i in range(0, 468):
            fLms = holisticResults.face_landmarks.landmark[i]
            x = int(fLms.x * width)
            y = int(fLms.y * height)
            #cv2.circle(imgRGB, (x, y), 1, (100, 100, 0), -1)
            #cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            # print(i, x, y)
        print(holisticResults.face_landmarks)
        upperLip = holisticResults.face_landmarks.landmark[13]
        bottomLip = holisticResults.face_landmarks.landmark[14]
        p1 = int(upperLip.x * width), int(upperLip.y * height)
        p2 = int(bottomLip.x * width), int(bottomLip.y * height)
        distance = dis.euclidean(p1, p2)
        # print(int(distance))
        if distance > 30:
            cv2.putText(img, 'Uyuyorsun', (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # 159sağ üst göz 144 sağ alt göz
        rightUpperLid = holisticResults.face_landmarks.landmark[159]
        rightBottomLid = holisticResults.face_landmarks.landmark[144]
        p1 = int(rightUpperLid.x * width), int(rightUpperLid.y * height)
        p2 = int(rightBottomLid.x * width), int(rightBottomLid.y * height)
        distance = dis.euclidean(p1, p2)
        # print(int(distance))
        if distance < 11:
            cv2.putText(img, 'Gozunu Ac', (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # 386sol üst göz 374 sol alt göz

    # Sag El Tanima
    mpDraw.draw_landmarks(img, holisticResults.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
                          mpDraw.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=2),
                          mpDraw.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))
    if holisticResults.right_hand_landmarks:
        for i in range(0, 10):
            fLms = holisticResults.right_hand_landmarks.landmark[i]
            x = int(fLms.x * width)
            y = int(fLms.y * height)
            # print(i, x, y)
        rightEar = holisticResults.face_landmarks.landmark[93]  # Sağ Kulak
        rightWrist = holisticResults.right_hand_landmarks.landmark[0]  # Sağ Bilek
        p1 = int(rightEar.x * width), int(rightEar.y * height)
        p2 = int(rightWrist.x * width), int(rightWrist.y * height)
        distance = dis.euclidean(p1, p2)
        # print(int(distance))
        if distance < 100:
            cv2.putText(img, 'Elini Indir', (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    # Poz Tanima
    # 16sag bilek 15sol bile
    mpDraw.draw_landmarks(img, holisticResults.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
                          mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                          mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    if holisticResults.pose_landmarks:
        for i in range(0, 20):
            fLms = holisticResults.pose_landmarks.landmark[i]
            x = int(fLms.x * width)
            y = int(fLms.y * height)
            # print(i, x, y)
        rightEar = holisticResults.face_landmarks.landmark[93]  # Sağ Kulak
        rightWrist = holisticResults.pose_landmarks.landmark[16]  # Sağ Bilek
        p1 = int(rightEar.x * width), int(rightEar.y * height)
        p2 = int(rightWrist.x * width), int(rightWrist.y * height)
        distance = dis.euclidean(p1, p2)
        #print(p1, p2, int(distance))
        if distance < 100:
            cv2.putText(img, 'O Telefonu Birak', (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    # Holistic çalışan başlangıç
    # # Draw face landmarks
    # mpDraw.draw_landmarks(img, holisticResults.face_landmarks, mpHolistic.FACEMESH_CONTOURS,
    #                       mpDraw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #                       mpDraw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # # Right hand
    #
    # mpDraw.draw_landmarks(img, holisticResults.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
    #                       mpDraw.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=2),
    #                       mpDraw.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))
    #
    # # Left Hand
    # mpDraw.draw_landmarks(img, holisticResults.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
    #                       mpDraw.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),
    #                       mpDraw.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1))
    #
    # # Pose Detections
    # mpDraw.draw_landmarks(img, holisticResults.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
    #                       mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
    #                       mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    # #holistic bitiş

    # çalışan facemesh
    # if faceMeshResults.multi_face_landmarks:
    #     for faceLms in faceMeshResults.multi_face_landmarks:
    #         mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)
    #         for id, lm in enumerate(faceLms.landmark):
    #             #print(lm)
    #             ih, iw, ic = img.shape
    #             x, y = int(lm.x*iw), int(lm.y*ih)
    #             cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
    #             print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(img, f'HxW: {int(height), int(width)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("Image", cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA))
    # cv2.imshow("IMG2", imgRGB)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
