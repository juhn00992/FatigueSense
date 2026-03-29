import cv2
import mediapipe as mp
import numpy as np
import time

from fatigue_model import normalized_fatigue

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

yawn_start_time = None
yawn_count = 0


def mouth_aspect_ratio(landmarks):

    top_lip = np.array([landmarks[13].x, landmarks[13].y])
    bottom_lip = np.array([landmarks[14].x, landmarks[14].y])

    left_corner = np.array([landmarks[78].x, landmarks[78].y])
    right_corner = np.array([landmarks[308].x, landmarks[308].y])

    vertical = np.linalg.norm(top_lip - bottom_lip)
    horizontal = np.linalg.norm(left_corner - right_corner)

    return vertical / horizontal


def detect_yawn(frame):

    global yawn_start_time, yawn_count

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            mar = mouth_aspect_ratio(face_landmarks.landmark)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if mar > 0.6:

                if yawn_start_time is None:
                    yawn_start_time = time.time()

                duration = time.time() - yawn_start_time

                cv2.putText(frame, f"Open Time: {duration:.1f}s", (30,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                if duration >= 2.5:
                    cv2.putText(frame, "YAWN DETECTED", (30,110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            else:

                if yawn_start_time is not None:
                    duration = time.time() - yawn_start_time

                    if duration >= 2.5:
                        yawn_count += 1
                        print("Yawn detected! Total:", yawn_count)

                yawn_start_time = None

            # Calculate fatigue score
            fatigue = normalized_fatigue(yawn_count)

            cv2.putText(frame, f"Yawns: {yawn_count}", (30,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.putText(frame, f"Fatigue percentage: {fatigue:.2f}", (30,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

    return frame
