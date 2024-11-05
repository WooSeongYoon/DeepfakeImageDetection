from pymongo import MongoClient
import os
from flask import Flask, request, render_template
import CustomVision
import Face

app = Flask(__name__)

# MongoDB 연결
client = MongoClient('몽고 DB 연결 코드')
db = client['database']
collection = db['myCollection']


# 업로드된 파일을 저장할 폴더 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    if video.filename == '':
        return "파일 이름이 없습니다.", 400

    # 동영상 저장
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    print(video_path)

    output_folder = './result'
    interval = 50

    predictions = []      # 확률 리스트

    # 비디오로부터 얼굴 이미지 추출
    Face.save_faces_from_video(video_path, output_folder, interval)
    
    # ouput_folder에 저장된 얼굴 이미지들을 Custom Vision API로 분류
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        prediction = CustomVision.predict_image(file_path)
        predictions.append(prediction)

    print(f"{filename}: {prediction}")
    # Deepfake 확률 계산
    result1 = CustomVision.deepfake_Score(predictions, output_folder, 'Deepfake')
    result2 = CustomVision.deepfake_Score(predictions, output_folder, 'Not Deepfake')
    print(result1)
    if result1 == None:
        return render_template('result.html', result3 = "딥페이크 영상이 아닌것으로 판단 되었습니다.") 
    elif result2 == None:
        return render_template('result.html', result3 = "딥페이크 영상으로 판단 되었습니다.") 
    if result1[2] > result2[2]:
        df_result = "딥페이크 영상으로 판단 되었습니다."
    elif result1[2] > result2[2]:
        df_result = "딥페이크 영상이 아닌것으로 판단 되었습니다."
    else:
        if (result1[0]+result1[1]) > (result2[0]+result2[1]):
            df_result = "중간값을 기준으로 딥페이크 영상으로 판단 되었습니다."
        else:
            df_result = "중간값을 기준으로 딥페이크 영상이 아닌것으로 판단 되었습니다."
    return render_template('result.html', result3 = df_result)


if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1")
