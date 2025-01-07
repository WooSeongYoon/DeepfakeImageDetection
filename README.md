# DeepfakeImageDetection
Azure(Custom Vision) API을 활용한 딥페이크 영상 분석   
개발 인원: 우성윤, 안동휘, 박규섭

## 개발 배경
 현재 딥페이크 영상으로 인한 피해가 점점 증가하고 있습니다. 유명인들의 얼굴을 합성하여 영상을 제작하고 이를 사기에 활용하는 사례가 다수 발생하고 있으며, 최근에는 미성년자 일반인을 딥페이크에 합성하여 음란물로 만드는 사건도 증가하고 있습니다. 경찰이 딥페이크 탐지 소프트웨어를 개발하여 선거 범죄 및 음란물 확인을 진행하고 있습니다. 특히 2024년도에 30억 원에 가까운 연구개발 예산이 편성이 되었습니다.   
 하지만, 일반인이 동영상을 일일 경찰에 신고하여 확인받은 후에 영상을 시청할 수는 없습니다. 그렇기에 저희가 프로그램을 통해서 개인이 판단하기 어렵지만, 일차적으로 확인하여 선제적으로 대처하고 피해를 예방할 수 있게 하도록 “딥페이크 분석 웹사이트”를 설계 및 개발하였습니다.   
 
## 설계 및 구형
1. 데이터셋 제작
youtube에 있는 유명인들의 딥페이크 영상과 DeepFaceLab2.0을 이용해 AI로 직접 제작한 딥페이크를 만들어서 제작 하였습니다.   
![image](https://github.com/user-attachments/assets/2e35a447-22a9-4e09-8d40-c9d44578f66f)

2. 데이터셋 전처리
FaceNet에 기반한 MTCNN 모델을 활용하여 이미지를 아래 표와 같은 형태로 이미지에서 얼굴 특징점 위치를 리턴받아 원본 이미지에서 눈, 코, 입 부분만 동영상 50프레임당 한 장씩 추출하였습니다.   
![image](https://github.com/user-attachments/assets/65120949-5cac-4791-b969-c77b2b8b26c3)

3. 딥페이크 판독
Microsoft Azure Custom Vision을 이용해 데이터셋을 학습하고, 이를 통해 학습된 모델은 아래와 같은 결과를 리턴하게 됩니다.   
![image](https://github.com/user-attachments/assets/958c1040-485e-4195-8441-e743a3445cf9)
Azure API 결과   
이러한 Not DeepFake, DeepFake 확률을 최대 확률, 최소 확률, 평균 확률 ,총합 확률을 종합하여 딥페이크를 판단하게 합니다.

4. 웹 서버 및 웹
사전에 기술한 운영 방향성에 맞게끔 상기의 과정을 누구나 이용 할 수 있게 하기 위해서 flask를 통해 웹서버를 구축 하였고, 간단한 비디오 업로드 웹사이트를 제작하여 딥페이크를 판단해 볼 수 있게 제작 하였습니다.


## 프로그램 실행
1. 웹페이지 내에 영상 업로드
![웹페이지_입력](https://github.com/user-attachments/assets/094f682a-61a9-4673-9a83-187c73e0a056)   
해당 이미지는 app.py파일을 실행한 이미지입니다.   
웹페이지에서 "파일 선택" 버튼을 누르고 동영상을 업로드한 후에 "업로드" 버튼을 눌러 딥페이크 분석을 진행합니다.   
업로드 버튼을 누르면 Face.py파일이 실행되어 동영상이 지정한 프레임 단위로 전처리를 진행하여 이미지로 저장합니다.   
![이미지 전처리_결과](https://github.com/user-attachments/assets/16d75261-ca51-489d-a24f-26908fc8720e)   


2. 업로드된 영상 분석
이미지가 모두 저장되면 CustomVision.py파일이 실행되어 딥페이크 분석을 진행합니다.   
딥페이크 분석은 사전에 학습한 Azure Custom Vision API를 사용하여 확률을 통해 딥페이크 영상 여부를 판단합니다.   
**딥페이크 판단**   
Deepfake 확률만 존재: 딥페이크 영상.   
Non Deepfake 확률만 존재: 딥페이크 영상이 아님.   
Deepfake와 Non Deepfake의 활률이 둘다 존재: 각각의 평균값을 반환하여 평균값이 높은 쪽으로 판단.   

3. 결과 출력
판단을 완료하면 아래와 같이 웹페이지를 통해 확인이 가능합니다.   
![웹페이지_결과](https://github.com/user-attachments/assets/d802b01d-6d21-4762-8b79-907ce386c269)   
진행한 결과는 Azure에서 아래와 같이 확인 가능합니다.   
![Azure_결과 (1)](https://github.com/user-attachments/assets/44c7e25f-95fc-45d0-8e46-6e95e7d2bec7)

## 결론
한국 남성 및 백인 남성은 약70% 이상의 정확도로 딥페이크를 탐지 할 수 있는 것으로 확인 했으나, 두 인종을 제외한 남성과 모든 인종의 여성은  약50% 정도의 탐지율을 보여주어 상대적으로 매우 탐지율이 떨어지는 모습을 보였고, 단순히 화질이 낮은 경우에도 딥페이크로 오탐하는 문제가 발생 하였기 때문에 아직까지는 많은 개선이 필요해 보입니다.   

다만, 웹서버와 Microsoft 사의 AzureAI를 연동하거나, 여러 AI 모델과 각종 라이브러리를 학습하는 등의 강의 에서는 미처 배우지 못했던 부분들을 졸업 작품이라는 틀 아래에서 작업 할 수 있게 되어 학업 증진에는 매우 많은 도움이 되었다고 생각하고, 이러한 기회를 주셔서 감사하게 생각합니다.   

추후에는 이러한 문제를 해결 하기 위해 더욱 많은 데이터 셋과 학습 시간은 물론이고 다른 국내외 기술과 논문을 적절히 참고하여 프로세스 자체를 개선해보고, 실제로 서비스 하기 위해서 여러 가지 방법을 찾아보고 적절히 적용해 볼 예정입니다.   

## 성과
![24-2_DU공학제](https://github.com/user-attachments/assets/d23b9cb4-c842-4667-bedc-beef46421e3e)
