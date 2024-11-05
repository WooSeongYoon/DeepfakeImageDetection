# 얼굴에서 눈, 코, 입에만 마스크를 적용하는 함수
import cv2
import os
import numpy as np
from mtcnn import MTCNN

# 지정하는 프레임마다 얼굴을 탐지하고 이미지로 저장
def save_faces_from_video(video_path, output_folder, interval):
    # 입력 비디오 파일이 존재하는지 확인
    # 이미지 저장 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 이미지 저장 폴더가 이미 존재하면 이미지 파일 삭제
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            os.remove(file_path)

    # MTCNN 얼굴 탐지 모델 초기화
    detector = MTCNN()

    # 비디오 캡처 객체 초기화
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    face_count = 0

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
                x, y = max(0, x), max(0, y)  # 좌표가 음수일 경우 처리
                face_img = frame[y:y+height, x:x+width]  # 얼굴 영역 추출

                # 얼굴 이미지를 파일로 저장
                face_filename = os.path.join(output_folder, f'face_{frame_count}_{i}.jpg')
                cv2.imwrite(face_filename, face_img)
                face_count += 1

            # 현재 프레임의 얼굴 탐지 결과 출력 (선택적)
            print(f'Frame {frame_count}: {len(faces)} faces detected')

    # 비디오 캡처 객체 해제
    cap.release()

    print(f'Total {face_count} faces were saved in "{output_folder}" folder.')

# 얼굴에서 눈, 코, 입에만 마스크를 적용하는 함수
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

# 지정된 폴더 내의 모든 이미지를 처리하는 함수
def process_images_in_folder(input_folder, output_folder):
    # 출력 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더에서 모든 파일을 읽어옴
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 이미지 파일인지 확인 (jpg, png 등 확장자 확인)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 이미지 로드
            image = cv2.imread(file_path)

            # 얼굴 마스크 적용
            masked_image = create_face_mask(image)

            if masked_image is not None:
                # 출력 파일 경로
                output_path = os.path.join(output_folder, filename)

                # 마스크 처리된 이미지 저장
                cv2.imwrite(output_path, masked_image)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"No face detected in {filename}")