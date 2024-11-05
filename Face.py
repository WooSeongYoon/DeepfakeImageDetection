import cv2
import os
from mtcnn import MTCNN
import numpy as np

# 얼굴에서 눈, 코, 입을 마스크 형태로 남기는 함수
def create_face_mask(face_image):
    detector = MTCNN()
    faces = detector.detect_faces(face_image)

    if not faces:
        return None

    # 첫 번째 얼굴에 대해서만 처리
    face = faces[0]
    keypoints = face['keypoints']

    # 마스크 초기화
    mask = np.zeros_like(face_image)

    # 눈, 코, 입 좌표를 이용하여 마스크 생성
    for feature in keypoints.values():
        cv2.circle(mask, (feature[0], feature[1]), 15, (255, 255, 255), -1)

    # 원본 얼굴 이미지와 마스크 결합
    masked_face = cv2.bitwise_and(face_image, mask)

    return masked_face

# 지정하는 프레임마다 얼굴을 탐지하고 이미지로 저장
def save_faces_from_video(video_path, output_folder, interval):
    # 이미지 저장 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 이전 파일 삭제
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            os.remove(file_path)

    # MTCNN 얼굴 탐지 모델 초기화
    detector = MTCNN()

    # 비디오 캡처 객체 초기화
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    # 지정하는 프레임마다 얼굴을 탐지하고 이미지로 저장
    while True:
        ret, frame = cap.read()  # 프레임 읽기

        # 동영상이 끝나면 루프 종료
        if not ret:
            break

        frame_count += 1

        # interval 프레임마다 얼굴 탐지
        if frame_count % interval == 0:
            faces = detector.detect_faces(frame)

                # 탐지된 얼굴 영역을 이미지로 저장
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                face_img = frame[y:y + height, x:x + width]  # 얼굴 영역 추출

                # 얼굴에서 눈, 코, 입을 마스크 형태로 남기기
                masked_face = create_face_mask(face_img)

                if masked_face is not None:
                    masked_filename = os.path.join(output_folder, f'masked_face_{frame_count}_{i}.jpg')
                    cv2.imwrite(masked_filename, masked_face)

            # 현재 프레임의 얼굴 탐지 결과 출력 (선택적)
                print(f'Frame {frame_count}: {len(faces)} faces detected')

    # 비디오 캡처 객체 해제
    cap.release()

# 이 함수는 호출할 때 비디오 경로, 출력 폴더 및 프레임 간격을 인자로 전달받습니다.
