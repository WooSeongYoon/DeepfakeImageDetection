# DeepfakeImageDetection
Azure(Custom Vision) API을 활용한 딥페이크 영상 분석

!개요

![웹페이지_입력](https://github.com/user-attachments/assets/094f682a-61a9-4673-9a83-187c73e0a056)   
해당 이미지는 app.py파일을 실행한 이미지입니다.   
웹페이지에서 "파일 선택" 버튼을 누르고 동영상을 업로드한 후에 "업로드" 버튼을 눌러 딥페이크 분석을 진행합니다.   
업로드 버튼을 누르면 Face.py파일이 실행되어 동영상이 지정한 프레임 단위로 전처리를 진행하여 이미지로 저장합니다.   
![이미지 전처리_결과](https://github.com/user-attachments/assets/16d75261-ca51-489d-a24f-26908fc8720e)   

이미지가 모두 저장되면 CustomVision.py파일이 실행되어 딥페이크 분석을 진행합니다.   
딥페이크 분석은 사전에 학습한 Azure Custom Vision API를 사용하여 확률을 통해 딥페이크 영상 여부를 판단합니다.   

**딥페이크 판단**   
Deepfake 확률만 존재: 딥페이크 영상.   
Non Deepfake 확률만 존재: 딥페이크 영상이 아님.   
Deepfake와 Non Deepfake의 활률이 둘다 존재: 각각의 평균값을 반환하여 평균값이 높은 쪽으로 판단.   

판단을 완료하면 아래와 같이 웹페이지를 통해 확인이 가능합니다.   
![웹페이지_결과](https://github.com/user-attachments/assets/d802b01d-6d21-4762-8b79-907ce386c269)   

진행한 결과는 Azure에서 아래와 같이 확인 가능합니다.   
![Azure_결과 (1)](https://github.com/user-attachments/assets/44c7e25f-95fc-45d0-8e46-6e95e7d2bec7)
