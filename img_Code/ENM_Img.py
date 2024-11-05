import cv2
import dlib
import numpy as np
import os

# 입력 및 출력 디렉토리 설정
input_dir = 'deepData/elon_true' # 이미지가 있는 폴더
output_dir = 'Re_Img/true1'  # 결과 이미지를 저장할 폴더

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 얼굴 검출기 및 랜드마크 예측 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 입력 폴더에 있는 모든 이미지 파일을 처리
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일 확장자 필터링
        image_path = os.path.join(input_dir, filename)
        
        # 이미지를 읽어옵니다.
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴을 검출합니다.
        faces = detector(gray)

        # 검출된 얼굴에 대해 작업합니다.
        for face in faces:
            landmarks = predictor(gray, face)
            
            # 눈, 코, 입에 해당하는 랜드마크 포인트를 가져옵니다.
            points = []
            
            # 왼쪽 눈(36~41번 포인트)
            points.extend([landmarks.part(n) for n in range(36, 42)])
            
            # 오른쪽 눈(42~47번 포인트)
            points.extend([landmarks.part(n) for n in range(42, 48)])
            
            # 코(27~35번 포인트)
            points.extend([landmarks.part(n) for n in range(27, 36)])
            
            # 입(48~60번 포인트)
            points.extend([landmarks.part(n) for n in range(48, 61)])
            
            # 랜드마크 포인트들을 배열로 변환합니다.
            points = np.array([(p.x, p.y) for p in points])
            
            # 마스크를 생성합니다.
            mask = np.zeros_like(gray)
            
            # 랜드마크 포인트들을 기준으로 다각형을 그려 해당 부분만 남기고 나머지를 마스킹합니다.
            cv2.fillConvexPoly(mask, points, 255)
            
            # 마스크를 컬러 이미지로 변환합니다.
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 원본 이미지와 마스크를 AND 연산하여 눈, 코, 입만 남깁니다.
            result = cv2.bitwise_and(image, mask)
            
            # 결과 이미지를 저장할 경로 설정
            output_path = os.path.join(output_dir, f'result_{filename}')
            
            # 결과 이미지를 저장합니다.
            cv2.imwrite(output_path, result)

            print(f'Result image saved at: {output_path}')

print('All images processed and saved.')