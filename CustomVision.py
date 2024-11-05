import requests
import numpy as np
import os

def predict_image(image_path):
    # Custom Vision API 정보 설정
    prediction_key = "d6bc657cca1d4325977b60ee1f8c6e79"  # 예측 키
    endpoint = "https://21928296vision-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/f020151b-1afd-453b-bd3a-ef95e5436549/classify/iterations/Iteration1/image"

    # 이미지 파일 읽기
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # HTTP 요청 헤더 설정
    headers = {
        "Prediction-Key": prediction_key,
        "Content-Type": "application/octet-stream"
    }

    # HTTP POST 요청 보내기
    response = requests.post(endpoint, headers=headers, data=image_data)

    # 결과 출력
    if response.status_code == 200:
        predictions = response.json()
        for prediction in predictions['predictions']:
            print(f"태그: {prediction['tagName']}, 확률: {prediction['probability']:.2f}")
            return prediction['tagName'], prediction['probability']

    else:
        print(f"Error: {response.status_code}")
        print(response.text)

        return None

import numpy as np
import os

def deepfake_Score(predictions, output_folder, labels):
    score = []
    deepfake = [value for label, value in predictions if label == labels]
    returnstring = []
    if deepfake:
        # labels 레이블을 가진 값 중에서 최대값을 찾기
        maxScore = max(deepfake)
        max_index = predictions.index((labels, max(deepfake)))
        max_filename = os.listdir(output_folder)[max_index]

        # labels 레이블을 가진 값 중에서 최소값을 찾기
        minScore = min(deepfake)
        min_index = predictions.index((labels, min(deepfake)))
        min_filename = os.listdir(output_folder)[min_index]

        # labels 레이블을 가진 값 중에서 평균값을 찾기
        avgScore = np.mean(deepfake)


        print(f'{labels} 최대 확률: {maxScore}')
        print(f"{labels} 확률이 가장 높은 이미지 파일: {max_filename}")
        score.append(maxScore)
        print(f'{labels} 최소 확률: {minScore}')
        print(f"{labels} 확률이 가장 낮은 이미지 파일: {min_filename}")
        score.append(minScore)
        print(f'{labels} 평균 확률: {avgScore}')
        score.append(int(avgScore*100))
        return score