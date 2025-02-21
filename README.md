# skeleton_Vision
### 📌 개요
본 프로젝트는 **YOLOv5, HRNet, EfficientNet** 등을 활용하여 **사람의 자세를 분석하고 평가**하는 AI 모델을 개발한 자료입니다. YOLOv5를 이용해 **객체를 검출**하고, HRNet으로 **자세 추정(Human Pose Estimation)** 을 수행한 뒤, EfficientNet을 기반으로 **다중 라벨 자세 분류(Multi-Label Classification)** 를 진행했습니다. 마지막으로, **EfficientNet-Lite 모델을 모바일 환경에 적용**하여 **실시간 자세 평가**가 가능하도록 구현하였습니다.

### 📌 시연 영상



https://github.com/user-attachments/assets/a467a745-6ba0-4137-81d4-5c3c89a652b2



### ✅ TASK 01 - YOLOv5 기반 객체 검출
#### 📍 목표
- YOLOv5s를 활용한 데이터 전처리 및 검출 모델 학습

#### 📍 데이터 처리
- 15,967개 데이터 (Training: 80%, Validation: 10%, Test: 10%)로 분할
- 누락된 Label 값 제거 및 커스텀 데이터 적용

#### 📍 학습 환경
- **Google Colab** 사용 (Tesla T4 GPU)

#### 📍 결과
- 커스텀 데이터 학습을 완료하여 후속 작업(Task 02)에 사용될 **Cropped 이미지 생성**

---

### ✅ TASK 02 - HRNet 기반 자세 추정 (Human Pose Estimation)
#### 📍 사용 모델
- **HRNet (High-Resolution Network)**

#### 📍 특징
- CNN 기반의 고해상도 유지, 적은 연산량으로 높은 정확도 제공

#### 📍 진행 과정
1. **Task 01에서 생성된 Cropped 이미지**를 입력 데이터로 활용
2. **Pretrained weight (`pose_hrnet_w48_384x288.pth`)** 적용하여 Pose Estimation 수행
3. 이미지에 **Skeleton을 그려 결과 시각화**

#### 📍 결과 분석
- 대부분의 이미지에서 **정확한 Skeleton 예측 성공**
- 일부 장애물이나 가려진 부분에서도 안정적인 결과 도출
- 단, **상반신만 크롭된 이미지의 경우 하반신 Keypoint까지 예측되는 문제 발생**

#### 📍 개선 방안
- **상반신 크롭된 이미지를 별도로 분류**하여 HPE 모델을 재학습하면 더 높은 성능 기대 가능

---

### ✅ TASK 03 - EfficientNet을 활용한 Multi-Label Classification
#### 📍 목표
- **사람의 자세를 자동 분류하는 모델 개발**

#### 📍 특징
- **Multi-Class가 아닌 Multi-Label Classification** 접근법 적용
- **EfficientNet-b0 모델 활용** (적은 파라미터로 높은 성능)

#### 📍 분류할 라벨
- **목 비틀림(Neck Twisted)**, **몸통 굽힘(Trunk Bending)**, **몸통 비틀림(Trunk Twisted)** 등

#### 📍 결과
- **Label별 Confusion Matrix 분석 진행**
- **EfficientNet_b0 모델을 기반으로 라벨 분류 성능 평가 수행**

---

### ✅ TASK 04 - 인간공학적 평가 기법을 적용한 자세 평가
#### 📍 목표
- **인간공학적 평가 방법론(OWAS, REBA, RULA 등)과 AI 모델을 결합하여 자세 평가 점수 산출**

#### 📍 진행 과정
- **OWAS 방식 적용** (허리, 상체, 하체, 무게 고려)
- **총 252개 자세 코드 기반으로 Binary Search를 활용한 실시간 점수 계산 방식 도입**

#### 📍 결론
- 기존 방법보다 **OWAS가 Label과 일치하는 부분이 많아 해당 방법으로 평가 진행**

---

### ✅ TASK 05 - 모바일 환경에서 EfficientNet-Lite 모델 적용
#### 📍 목표
- **AI 모델을 모바일 환경에서 동작하도록 변환**

#### 📍 진행 과정
1. **EfficientNet-Lite 모델을 PyTorch JIT 변환하여 경량화**
2. **Android Flutter 기반으로 UI 개발**
3. **이미지 모드 & 스트리밍 모드 제공**

#### 📍 결과
- **PyTorch 모델을 `model.pt`로 저장**하여 모바일 환경에서도 효율적으로 동작하도록 구현

<br>

### 📌 결과물
|값 측정 페이지|결과 값 출력 페이지|
|-----------------|----|
| ![image](https://github.com/user-attachments/assets/0d0de8d9-fcb6-40db-8a3b-a3de65be5ee3) ![image](https://github.com/user-attachments/assets/392486b4-9992-4dc8-b50f-80bc813acef3) |![image](https://github.com/user-attachments/assets/66d31473-cd7e-4de3-978e-51b9ec19e4b3)|

<br>

### 📌 세부설명
![image](https://github.com/user-attachments/assets/056ce94f-77cb-4bb0-9352-7e11a6617f77)
![image](https://github.com/user-attachments/assets/a6c5c061-e23b-400d-af7c-a01584001024)
![image](https://github.com/user-attachments/assets/8551226c-4aa5-43f8-89b1-00a6d9ec6ec5)
![image](https://github.com/user-attachments/assets/43d8a271-2d98-4b19-b6e0-3bf962109d16)
![image](https://github.com/user-attachments/assets/aa50545a-e0ac-4405-94d4-ceef936129c1)
![image](https://github.com/user-attachments/assets/494ddb5c-d9b0-4419-a4b9-9f3d76f2d6b0)
![image](https://github.com/user-attachments/assets/3da766f5-72be-49ff-bcf3-62decb3ee483)
![image](https://github.com/user-attachments/assets/d1bee0d8-f2d9-4b88-8a3e-6b5dbe013670)
![image](https://github.com/user-attachments/assets/3f79884a-f9ec-4c93-a39f-1fde6c2d6b52)
![image](https://github.com/user-attachments/assets/44da19be-4a89-4294-8fa5-ea473b77f1c9)
![image](https://github.com/user-attachments/assets/d28a7950-383e-4469-80c1-63366c1da317)
![image](https://github.com/user-attachments/assets/93028206-b96a-431b-996a-a57ff99e0ac0)
![image](https://github.com/user-attachments/assets/f1abfd14-c6e0-4496-9a3c-9617760aa861)
![image](https://github.com/user-attachments/assets/e78893c3-d08c-4736-bed7-b96209d14da3)
![image](https://github.com/user-attachments/assets/eb892bcf-50dc-4fd8-b8ec-a8b728dc4127)
![image](https://github.com/user-attachments/assets/122f1062-781e-4424-9664-6c8a629adccc)
![image](https://github.com/user-attachments/assets/06bfb4ec-5f35-4469-a5cf-82b6596d42eb)
![image](https://github.com/user-attachments/assets/4eb51dd4-14ec-42fe-81ac-ab7da2277b6a)
![image](https://github.com/user-attachments/assets/d9ab2f3b-4219-4ed9-85f2-95b10eef2cea)
![image](https://github.com/user-attachments/assets/e60d0fcc-3b8b-4ffa-ab3f-e28c4e111063)
![image](https://github.com/user-attachments/assets/efb8fa01-1ff7-464d-a45d-b993eda82007)
![image](https://github.com/user-attachments/assets/6b241b8f-9e48-4c1e-8e2f-a4cf7fcf3985)
![image](https://github.com/user-attachments/assets/28e64ab6-a402-44a1-bfc7-9abca89f9f11)
![image](https://github.com/user-attachments/assets/fc0af57b-d6a5-4ddf-a054-6b0e79ccb21c)
![image](https://github.com/user-attachments/assets/d386bf17-0c69-4a9c-9ab5-b835adda0bf8)
![image](https://github.com/user-attachments/assets/bae78c70-5d4c-4171-9890-6245a9120c2f)
![image](https://github.com/user-attachments/assets/c2308e04-0987-4161-b86d-28a6c81be49e)
![image](https://github.com/user-attachments/assets/ea865904-d3f4-44ed-a00d-4ebe1b9d74e8)
![image](https://github.com/user-attachments/assets/c6e0629f-020a-4d20-bd29-a25ceeb2a977)
![image](https://github.com/user-attachments/assets/99a95695-2811-41f6-84c6-b6146ebea02e)
![image](https://github.com/user-attachments/assets/2022227b-d709-4ed3-89e8-8f3fd04be0cf)
![image](https://github.com/user-attachments/assets/e280b296-c30d-4e01-91e5-9b60ea655a0b)
![image](https://github.com/user-attachments/assets/3332f63d-d642-4006-ad33-60ce0dbf769c)
![image](https://github.com/user-attachments/assets/7d7780a8-658a-4784-b7b9-951ff5fd6e9c)
![image](https://github.com/user-attachments/assets/6d859e9d-7078-499b-90fc-be7b16940408)
![image](https://github.com/user-attachments/assets/455bf9ed-af63-4627-91f8-0de3c5282945)
![image](https://github.com/user-attachments/assets/355bdc78-c7d3-4559-ac72-265bbea8c2a0)
