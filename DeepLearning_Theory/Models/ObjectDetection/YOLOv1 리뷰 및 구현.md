

## YOLOv1 Architecture 

![|900](Data/Models/yolov1/1.png)

448x448 크기의 이미지를 입력받아 여러층의 Layer를 거쳐 이미지에 있는 객체 위치와 객체의 정체를 알아내는 구조이다. 

*Object Detection* 을 수행하는 모델은 크게 2가지 구조로 나뉜다.
- Backbone 
- Head 

Backbone은 입력받은 이미지의 특성을 추출하는 역할을 하고 head 에서는 특성이 추출된 특성 맵을 받아 object detection 역할을 수행한다. 



## Backbone 

Backbone은 특성 추출이 목적이기 때문에 특성 추출에 최적화된 모델, 즉 classification을 목적으로 만들어진 모델을 사용한다. 
Yolo의 저자들은 기존의 Backbone(Ex. VGG16)을 사용하지 않고 DarkNet이라는 모델을 만들었다. 



## Head 

![|900](Data/Models/yolov1/3.png)

448x448 해상도의 이미지를 입력받을 때 7, 7, 30 사이즈의 3차원 텐서를 출력으로 내놓는다.

측, 출력값의 셀 하나가 원본 이미지의 64x64영역을 대표하고 이 영역에서 검출된 30개의 데이터가 담겨있다는 뜻이다. (448 / 7 = 64)

30개의 데이터는 다음과 같다. 
1. 해당 영역을 중점으로 하는 객체의 Bounding Box 2개 (x, y, w, h, confidence)
2. 해당 영역을 중점으로 하는 객체의 class score 20개 

한 셀에서 2개의 bounding box를 검출하기 때문에 총 검출되는 박스는 7 x 7 x 2 = 98개 이다. 
이 98개의 박스는 각각의 confidence를 가지고 있다. confidence 는 bounding box를 얼마나 신뢰할 수 있는가를 나타낸 점수라고 볼 수 있다. 

confidence = Pr(Object) * IoU

x, y는 해당 셀에 대해 normalize된 값이고 w, h는 전체 이미지에 대해 normalize된 값이다. 
예를 들어 (0, 0)셀에서 나온 bounding box의 [x, y, w, h] 가 [0.5, 0.5, 0.2, 0.2]라면 변환 했을 때 x = 31, y = 31, w = 448 * 0.2 = 96, h= 96이다.

(0, 0) 셀은 원본 이미지의 (0, 0) <-> (63, 63)인 사각형을 대표하기 때문이다. 

20개의 class score는 해당 영역에서 검출된 객체가 어떤 클래스의 객체일 확률을 클래스 별로 나타낸 것이다. 20은 YOLO를 훈련시킬 때 사용할 PASCAL VOC 2007 dataset에 있는 클래스가 20종류라 20을 사용한 것이다. 

![|900](Data/Models/yolov1/4.png)


- 입력 이미지 먼저, 448x448 픽셀 크기의 이미지를 입력으로 받는다. 예를 들어, 강아지와 고양이가 있는 사진이라고 가정해보자.
- 그리드 분할 이 이미지를 7x7 그리드로 나눕니다. 각 그리드 셀은 64x64 픽셀(448/7=64) 영역을 담당한다.
- 그리드 셀의 출력 각 그리드 셀은 30개의 값을 출력한다. 이 30개의 값을 자세히 살펴보면:

a) 바운딩 박스 1 (5개 값):
	- x1: 박스 중심의 x 좌표 (0~1 사이 값, 셀 내에서의 상대적 위치)
	- y1: 박스 중심의 y 좌표 (0~1 사이 값, 셀 내에서의 상대적 위치)
	- w1: 박스의 너비 (0~1 사이 값, 전체 이미지 대비 상대적 크기)
	- h1: 박스의 높이 (0~1 사이 값, 전체 이미지 대비 상대적 크기)
	- c1: 신뢰도 점수 (0~1 사이 값)

b) 바운딩 박스 2 (5개 값):
	- x2, y2, w2, h2, c2 (위와 동일한 의미)

c) 클래스 점수 (20개 값):
	- 각 클래스에 대한 확률 (0~1 사이 값) 예: [개, 고양이, 새, 말, 양, 소, 코끼리, 곰, 얼룩말, 기린, 백팩, 우산, 핸드백, 넥타이, 여행가방, 프리스비, 스키, 스노우보드, 스포츠 공, 연]

4. 구체적인 예시 그리드의 (2,3) 위치에 있는 셀을 예로 들어보면. 이 셀이 다음과 같은 값을 출력했다고 가정해보자:

[0.3, 0.4, 0.5, 0.6, 0.8, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


이를 해석하면:

a) 첫 번째 바운딩 박스:

- x1 = 0.3, y1 = 0.4 (셀 내에서의 상대적 위치)
- w1 = 0.5, h1 = 0.6 (전체 이미지 대비 크기)
- c1 = 0.8 (높은 신뢰도)

b) 두 번째 바운딩 박스:

- x2 = 0.2, y2 = 0.3 (셀 내에서의 상대적 위치)
- w2 = 0.4, h2 = 0.5 (전체 이미지 대비 크기)
- c2 = 0.7 (꽤 높은 신뢰도)

c) 클래스 점수:

- [0.9, 0.05, 0, 0, ..., 0] 이는 첫 번째 클래스(개)에 대해 90% 확률, 두 번째 클래스(고양이)에 대해 5% 확률을 나타낸다.

5. 실제 좌표 계산 (2,3) 셀의 실제 좌표는 (128, 192)에서 시작한다. (2_64, 3_64). 첫 번째 바운딩 박스의 실제 좌표는:

- 중심 x = 128 + (64 * 0.3) = 147.2
- 중심 y = 192 + (64 * 0.4) = 217.6
- 너비 w = 448 * 0.5 = 224
- 높이 h = 448 * 0.6 = 268.8

6. 최종 해석 이 셀은 개(90% 확률)를 포함하고 있을 가능성이 높으며, 그 개의 위치는 대략 (147, 218)을 중심으로 하고 크기가 224x269 픽셀인 영역에 있을 것으로 예측한다.

이런 방식으로 모든 7x7=49개의 셀에 대해 예측을 수행하고, 이를 종합하여 전체 이미지에서의 객체 검출 결과를 얻게 된다.


## 활성화 함수

저자는 Linear Activation function과 Leaky ReLU를 사용했다. 
Linear Activation function는 맨 마지막 Layer에 사용했다. 즉 마지막 레이어를 Logit으로 출력한다. 
Leaky ReLU는 마지막 Layer를 제외한 모든 레이어에서 사용했다. 



## Loss function 

손실 함수는 multi-task loss를 사용한다. 

![|900](Data/Models/yolov1/2.png)

위에서 2줄은 bbox의 위치에 대한 손실(localization loss), 중간 3, 4번째 줄은 confidence score에 관한 손실(confidence loss), 마지막 한줄은 class score에 관한 에러이다 (classification loss)

$\Sigma ^{S^2}$ 에서 $S^2$ 은 전체 cell의 갯수 = 49 이고 B는 각 셀에서 출력하는 bounding box의 갯수 = 2이다. 
즉, localization loss, confidence loss는 해당 셀에 실제 객체의 중점이 있을 때 해당 셀에서 출력한 2개의 bounding box 중 Ground Truth Box와 IoU가 더 높은 bounding box와 Ground Truth Box와의 loss를 계산한 것들이다. 

그리고 classification loss는 해당 셀에서 실제 객체의 중점이 있을 때 해당 셀에서 얻은 class score와 label data 사이의 loss를 나타낸 값이다. 



## 훈련

1. Backbone: ImageNet 2012 dataset으로 1주일간 훈련 
2. Head : Weight decay = 0.0005, momentum = 0.9, batch size = 65, epoch = 135로 설정. learning rate를 0.001로 맞춘 뒤 epoch=75까지 0.01로 조금씩 상승시킴. 그 후 30회는 0.001로 훈련시키고 마지막 30회는 0.0001로 훈련 
3. 데이터 증강(Data Augmentation): 전체 이미지 사이즈의 20%만큼 random scaling수행. 그 후 translation도 하며 원본 이미지의 1.5배 만큼 HSV 증가시킴 




## 논문 구현 

kaggle datasets download -d aladdinpersson/pascalvoc-yolo

#### Dataset 

$Label_{cell} \; = \; [C_1, \; C_2, \; ..., \; C_{20}, \; p_c, \; x, \; y, \; w, \; h]$


## model.py
``` python

import torch
import torch.nn as nn

architecture_config = [
    # (kernel_size, filters, stride, padding) 
    (7, 64, 2, 3),
    "M", # Maxpooling2d
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # 첫번째 conv, 두번째 conv, 반복 횟수
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)), # 7 * 7 * (20 + 2 * 5) = 1,470
        )
```

## loss.py


### YOLO 손실 함수의 구성 요소
YOLO 손실 함수는 다음과 같이 네 가지 주요 부분으로 나뉜다:
1. **박스 좌표 손실 (Box Coordinates Loss)**
2. **객체 신뢰도 손실 (Object Confidence Loss)**
3. **배경 신뢰도 손실 (No Object Confidence Loss)**
4. **클래스 예측 손실 (Class Prediction Loss)**


### 1. 박스 좌표 손실 (Box Coordinates Loss)

$\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]$
- 이 수식은 실제 박스와 예측 박스 간의 위치와 크기 차이를 계산합니다. 크기는 너비와 높이의 제곱근으로 계산된다.

#### 코드 구현:
```python
predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

# 각 예측된 바운딩 박스와 실제 바운딩 박스 간의 IoU(교차 영역 비율)을 계산
iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

# 두 예측 중 IoU가 높은 박스를 선택
iou_maxes, bestbox = torch.max(ious, dim=0)
exists_box = target[..., 20].unsqueeze(3) # 실제 박스가 존재하는 위치

box_predictions = exists_box * (
    bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
)
box_targets = exists_box * target[..., 21:25]

box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
    torch.abs(box_predictions[..., 2:4] + 1e-6)
)
box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
box_loss = self.mse(
    torch.flatten(box_predictions, end_dim=-2),
    torch.flatten(box_targets, end_dim=-2),
)
```
- `box_predictions`와 `box_targets`에서 박스의 너비(w)와 높이(h)에 대해 제곱근을 취하는 것이 수식의 $\sqrt{w_i} - \sqrt{\hat{w}_i}$ 와 $\sqrt{h_i} - \sqrt{\hat{h}_i}$에 해당
- 최종 `box_loss`는 이러한 차이들을 제곱하여 합산한 값



### 2. 객체 신뢰도 손실 (Object Confidence Loss)

$\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$
- 이 수식은 박스에 객체가 있을 때 그 박스의 신뢰도 차이를 계산

#### 코드 구현:
```python
pred_box = (
    bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
)
object_loss = self.mse(
    torch.flatten(exists_box * pred_box),
    torch.flatten(exists_box * target[..., 20:21]),
)
```
- `pred_box`는 두 박스 예측 중 IoU가 높은 박스의 신뢰도 점수를 선택
- `object_loss`는 실제 박스의 신뢰도(`target[..., 20:21]`)와 예측된 신뢰도(`pred_box`)의 차이를 MSE로 계산



### 3. 배경 신뢰도 손실 (No Object Confidence Loss)

$\lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$
- 이 수식은 박스에 객체가 없을 때의 신뢰도 차이를 계산
#### 코드 구현:
```python
no_object_loss = self.mse(
    torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
)
no_object_loss += self.mse(
    torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
)
```
- 객체가 없는 경우(`(1 - exists_box)`)에 대한 신뢰도 차이를 두 예측값(predictions[..., 20:21]과 predictions[..., 25:26])에 대해 각각 계산



### 4. 클래스 예측 손실 (Class Prediction Loss)

$\sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$
- 이 수식은 객체가 있는 박스의 클래스 예측 오류를 계산
#### 코드 구현:
```python
class_loss = self.mse(
    torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
    torch.flatten(exists_box * target[..., :20], end_dim=-2),
)
```
- `class_loss`는 실제 클래스 레이블(`target[..., :20]`)과 예측된 클래스 레이블(`predictions[..., :20]`)의 차이를 MSE로 계산


``` python
import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    YOLO 모델(v1)의 손실을 계산하기 위한 클래스입니다.
    """
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # MSE 손실 함수를 초기화합니다.

        # S, B, C는 각각 이미지를 나누는 그리드의 크기, 바운딩 박스의 수, 클래스의 수를 의미합니다.
        self.S = S
        self.B = B
        self.C = C

        # 논문에서 제안된 손실 가중치입니다.
        self.lambda_noobj = 0.5  # 객체가 없는 박스의 손실 가중치
        self.lambda_coord = 5    # 박스 좌표의 손실 가중치

    def forward(self, predictions, target):
        # 예측 값과 실제 값을 입력받아 손실을 계산합니다.
        # 예측 값은 (BATCH_SIZE, S*S*(C+B*5))의 형태로 입력됩니다.
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 각 예측된 바운딩 박스와 실제 바운딩 박스 간의 IoU(교차 영역 비율)을 계산합니다.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # 두 예측 중 IoU가 높은 박스를 선택합니다.
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # 실제 박스가 존재하는 위치

        # 박스 좌표에 대한 손실 계산:
        # 최대 IoU를 가진 예측 박스만 사용하여 손실을 계산합니다.
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        # 박스의 너비와 높이의 제곱근을 취합니다.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # 박스 좌표의 손실을 MSE로 계산합니다.
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # 객체가 존재하는 박스의 신뢰도 손실:
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # 객체가 없는 박스의 손실:
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # 클래스 예측 손실:
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # 총 손실은 각 손실에 대한 가중치를 적용하여 합산합니다.
        loss = (
            self.lambda_coord * box_loss  # 박스 좌표 손실
            + object_loss  # 객체 손실
            + self.lambda_noobj * no_object_loss  # 객체 없음 손실
            + class_loss  # 클래스 손실
        )

        return loss
```



## dataset.py

``` python
import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    """
    파스칼 VOC 데이터셋을 로드하기 위한 PyTorch 데이터셋 클래스입니다.
    """
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        클래스의 생성자(초기화 메소드)입니다. 데이터셋을 로드할 때 필요한 설정을 초기화합니다.

        매개변수:
        csv_file: 이미지와 레이블 파일의 이름이 저장된 CSV 파일의 경로입니다.
        img_dir: 이미지 파일들이 저장된 디렉토리의 경로입니다.
        label_dir: 레이블 파일들이 저장된 디렉토리의 경로입니다.
        S: 이미지를 SxS 그리드로 나누는 값입니다.
        B: 각 그리드 셀 별 예측할 최대 바운딩 박스의 수입니다.
        C: 데이터셋의 클래스 수입니다.
        transform: 이미지와 레이블에 적용할 변환 함수(예: 리사이징, 크롭 등).
        """
        self.annotations = pd.read_csv(csv_file)  # CSV 파일을 읽어서 데이터 프레임으로 저장합니다.
        self.img_dir = img_dir  # 이미지 파일 디렉토리 경로를 저장합니다.
        self.label_dir = label_dir  # 레이블 파일 디렉토리 경로를 저장합니다.
        self.transform = transform  # 변환 함수를 저장합니다.
        self.S = S  # 그리드 사이즈 S를 저장합니다.
        self.B = B  # 바운딩 박스 수 B를 저장합니다.
        self.C = C  # 클래스 수 C를 저장합니다.

    """
    Python에서 __len__과 __getitem__ 같은 메서드는 특별한 용도로 사용되며 "매직 메서드" 또는 "던더(double underscore) 메서드"라고 불립니다. 이 메서드들은 Python의 데이터 모델을 구현하여 Python 객체가 내장 타입처럼 행동하도록 해줍니다. __ (더블 언더스코어)는 이 메서드들이 특별한 메서드임을 나타내며, 직접 호출하기보다는 Python 인터프리터에 의해 자동으로 호출되도록 설계되었습니다.

__len__(self)
__len__ 메서드는 객체의 길이를 반환합니다. 예를 들어, 리스트, 튜플, 문자열 등의 길이를 알고 싶을 때 len() 함수를 사용하는데, 이 len() 함수가 내부적으로 해당 객체의 __len__ 메서드를 호출합니다. 사용자 정의 클래스에서 __len__ 메서드를 정의하면, 그 클래스의 인스턴스에 대해서도 len() 함수를 사용할 수 있습니다.
    """
    def __len__(self):
        """
        데이터셋의 전체 길이(데이터 개수)를 반환합니다.

        반환값:
        int: 데이터셋에 있는 총 데이터(이미지)의 수입니다.
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        주어진 인덱스에 해당하는 데이터(이미지와 레이블)를 로드하고 반환합니다.

        매개변수:
        index: 데이터셋에서 가져올 데이터의 인덱스입니다.

        반환값:
        tuple: 처리된 이미지와 해당 이미지의 레이블 행렬을 포함하는 튜플입니다.
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []  # 박스 정보를 저장할 리스트를 초기화합니다.
        with open(label_path) as f:
            for label in f.readlines():
                # 레이블 파일에서 각 줄을 읽어서 클래스 레이블과 바운딩 박스 정보를 파싱합니다.
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])  # 박스 정보를 리스트에 추가합니다.

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)  # 이미지 파일을 열어서 Image 객체로 로드합니다.
        boxes = torch.tensor(boxes)  # 박스 정보 리스트를 텐서로 변환합니다.

        if self.transform:
            # 변환 함수가 설정되어 있다면 이미지와 박스에 적용합니다.
            image, boxes = self.transform(image, boxes)

        # 레이블 행렬을 생성합니다. 행렬의 크기는 (S, S, C + 5*B)입니다.
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i, j는 그리드 내의 셀 위치를 나타냅니다.
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # 셀 내에서의 박스 너비와 높이를 계산합니다.
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # 특정 셀(i, j)에 아직 객체가 없다면 객체 정보를 저장합니다.
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # 객체 존재 표시
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates  # 박스 좌표 저장
                label_matrix[i, j, class_label] = 1  # 클래스 레이블에 대한 원-핫 인코딩 설정

        return image, label_matrix  # 처리된 이미지와 레이블 행렬을 반환합니다.
```


Python에서 `__len__`과 `__getitem__` 같은 메서드는 특별한 용도로 사용되며 "매직 메서드" 또는 "던더(double underscore) 메서드"라고 불립니다. 이 메서드들은 Python의 데이터 모델을 구현하여 Python 객체가 내장 타입처럼 행동하도록 해준다. 
`__` (더블 언더스코어)는 이 메서드들이 특별한 메서드임을 나타내며, 직접 호출하기보다는 Python 인터프리터에 의해 자동으로 호출되도록 설계되었다.

### `__len__(self)`
`__len__` 메서드는 객체의 길이를 반환한다. 예를 들어, 리스트, 튜플, 문자열 등의 길이를 알고 싶을 때 `len()` 함수를 사용하는데, 
이 `len()` 함수가 내부적으로 해당 객체의 `__len__` 메서드를 호출한다. 
사용자 정의 클래스에서 `__len__` 메서드를 정의하면, 그 클래스의 인스턴스에 대해서도 `len()` 함수를 사용할 수 있다.

### 예시
```python
class MyCollection:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

coll = MyCollection([1, 2, 3, 4])
print(len(coll))  # 출력: 4
```
위 예시에서 `MyCollection` 클래스는 내부 데이터의 길이를 반환하는 `__len__` 메서드를 가지고 있습니다. `len(coll)`을 호출하면, `coll.__len__()`이 호출된다.

### `__getitem__(self, index)`
`__getitem__` 메서드는 객체의 특정 인덱스에 접근할 때 호출된다. 이 메서드를 구현함으로써, 객체를 리스트나 딕셔너리처럼 인덱싱할 수 있게 된다. 
즉, 객체에서 `obj[index]` 형태로 요소를 조회할 때 `__getitem__` 메서드가 호출된다.

### 예시
```python
class MyCollection:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]

coll = MyCollection([1, 2, 3, 4])
print(coll[2])  # 출력: 3
```
위 예시에서 `MyCollection` 클래스는 인덱스를 통해 내부 데이터에 접근하는 `__getitem__` 메서드를 구현하고 있습니다. `coll[2]`을 호출하면 `coll.__getitem__(2)`이 실행되어 인덱스 2의 값을 반환한다.

### PyTorch에서의 사용
PyTorch의 `Dataset` 클래스를 상속받아 사용자 정의 데이터셋을 만들 때, `__len__`은 데이터셋의 전체 데이터 수를 반환하는 용도로, 
`__getitem__`은 특정 인덱스의 데이터(이미지, 레이블 등)를 로드하고 전처리하는 데 사용된다. 
PyTorch에서 데이터 로더(DataLoader)를 통해 데이터셋을 반복할 때, 이 메서드들이 자동으로 호출되어 데이터를 순차적으로 로드하게 된다.




#### `self.annotations.iloc[index, 1]`
`self.annotations`는 CSV 파일에서 읽은 데이터를 저장하는 pandas의 DataFrame 객체입니다. `.iloc`는 이 DataFrame에서 위치 기반 인덱싱을 사용하여 데이터를 접근할 때 사용됩니다. 여기서 `index`는 접근하려는 행 번호를 나타내고, `1`은 두 번째 열을 의미합니다. 즉, `self.annotations.iloc[index, 1]`는 주어진 `index`의 행에서 두 번째 열의 데이터를 가져오는 것으로, 이 데이터는 레이블 파일의 이름을 포함하고 있습니다.

```python
label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
```
이 코드는 `self.label_dir` (레이블 파일들이 저장된 디렉토리 경로)와 `self.annotations.iloc[index, 1]` (해당 인덱스의 레이블 파일 이름)을 결합하여 전체 파일 경로를 만듭니다. 이 경로는 이후에 파일을 열고 데이터를 읽기 위해 사용됩니다.

#### 파일 읽기와 데이터 파싱
```python
with open(label_path) as f:
    for label in f.readlines():
        # 레이블 파일에서 각 줄을 읽어서 클래스 레이블과 바운딩 박스 정보를 파싱합니다.
        class_label, x, y, width, height = [
            float(x) if float(x) != int(float(x)) else int(x)
            for x in label.replace("\n", "").split()
        ]
```
이 코드는 `label_path`에 있는 파일을 열고 (`open(label_path) as f`), 파일의 각 줄을 반복하며 처리합니다 (`for label in f.readlines()`). `f.readlines()`는 파일의 모든 줄을 읽어서 리스트로 반환합니다. 각 줄 (`label`)은 클래스 레이블과 바운딩 박스의 위치 및 크기 정보를 담고 있습니다.

- `label.replace("\n", "").split()`: 각 줄에서 줄바꿈 문자 (`\n`)를 제거하고, 공백을 기준으로 문자열을 분리하여 리스트로 만듭니다. 예를 들어, 줄이 `"1 0.5 0.5 0.1 0.1\n"`이면, 이 코드는 `['1', '0.5', '0.5', '0.1', '0.1']`를 생성합니다.
- 리스트 내포를 사용하여 각 문자열을 실수(`float`)로 변환하고, 만약 그 실수가 정수값을 가질 경우 정수(`int`)로 다시 변환합니다. 이는 값이 소수점 없이 표현될 수 있을 때, 예를 들어 `1.0`을 `1`로 처리하기 위함입니다. 

```python
class_label, x, y, width, height = [
    float(x) if float(x) != int(float(x)) else int(x)
    for x in label.replace("\n", "").split()
]
```
여기서 `class_label`은 객체의 클래스 인덱스, `x`와 `y`는 바운딩 박스의 중심 위치, `width`와 `height`는 바운딩 박스의 너비와 높이를 나타냅니다. 이 값들은 모두 이미지 크기에 대한 상대적 비율로 주어집니다.

이 파싱된 데이터는 각 객체의 정보를 담고 있는 `boxes` 리스트에 추가됩니다:
```python
boxes.append([class_label, x, y, width, height])
```
이 리스트는 나중에 데이터셋의 각 항목을 반환할 때, 이미지와 함께 처리되어 신경망 학습에 사용됩니다.


이 코드 블록은 이미지와 해당 이미지에 대한 레이블 데이터를 로드하고, 그 데이터를 신경망에서 사용할 수 있도록 전처리하는 과정을 나타냅니다. 이 과정을 단계별로 자세히 살펴보겠습니다:

### 이미지 파일 경로 설정 및 이미지 로드
```python
img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
image = Image.open(img_path)
```
- `img_path`는 `os.path.join` 함수를 사용하여 `self.img_dir` (이미지가 저장된 디렉토리 경로)와 `self.annotations.iloc[index, 0]` (DataFrame에서 현재 인덱스의 이미지 파일 이름)을 결합하여 전체 파일 경로를 생성합니다.
- `Image.open(img_path)`는 Python Imaging Library (PIL)의 `Image` 모듈을 사용하여 해당 경로의 이미지 파일을 열고 `image` 객체로 로드합니다. 이 객체는 후처리 과정에서 이미지 데이터를 다루기 위해 사용됩니다.

### 바운딩 박스 데이터의 텐서 변환
```python
boxes = torch.tensor(boxes)
```
- `boxes` 리스트, 이전에 각 바운딩 박스의 데이터를 저장하기 위해 사용했던 것을 PyTorch의 텐서로 변환합니다. 이 텐서는 이미지와 함께 변환 작업을 거친 후 모델의 입력으로 사용됩니다.

### 이미지와 바운딩 박스의 변환 적용
```python
if self.transform:
    image, boxes = self.transform(image, boxes)
```
- `self.transform`은 데이터 전처리를 위해 정의된 변환 함수(예: 리사이징, 크롭, 색상 조정 등)가 있을 경우 이를 이미지와 바운딩 박스 데이터에 적용합니다. 이 과정은 모델 학습에 적합한 형태로 데이터를 표준화하거나 증강시키는 데 사용됩니다.

### 레이블 행렬 생성
```python
label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
```
- `label_matrix`는 이미지를 `S x S` 그리드로 나누고 각 그리드 셀에 대한 정보를 저장하는 텐서입니다. 차원은 `(self.S, self.S, self.C + 5*self.B)`로, 클래스 레이블, 객체 존재 여부, 바운딩 박스의 좌표와 신뢰도를 포함합니다.

### 각 바운딩 박스 처리 및 레이블 행렬 채우기
```python
for box in boxes:
    class_label, x, y, width, height = box.tolist()
    class_label = int(class_label)
    i, j = int(self.S * y), int(self.S * x)
    x_cell, y_cell = self.S * x - j, self.S * y - i
    width_cell, height_cell = width * self.S, height * self.S

    if label_matrix[i, j, 20] == 0:
        label_matrix[i, j, 20] = 1
        box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        label_matrix[i, j, 21:25] = box_coordinates
        label_matrix[i, j, class_label] = 1
```
- 각 바운딩 박스를 반복 처리하며, 해당 박스의 클래스 레이블과 위치 정보를 추출합니다.
- `i, j`는 바운딩 박스의 중심이 위치하는 그리드 셀의 인덱스입니다.
- `x_cell, y_cell`은 그리드 셀 내에서의 상대적 위치, `width_cell, height_cell`은 그리드 셀에 대한 상대적 너비와 높이를 계산합니다.
- `label_matrix[i, j, 20]`은 해당 셀에 객체가 있는지 없는지를 나타내며, 객체가 있다면 (`0`에서 `1`로 변경) 해당 셀에 바운딩 박스의 좌표와 클래스 레이블을 저장합니다. 이 과정은 각 셀에 최대 한 개의 객체만을 고려합니다 (다중 객체를 고려하지 않는 단순화된 설정).

### 데이터 반환
```python
return image, label_matrix
```
- 전처리가 완료된 이미지와 해당 이미지에 대한 레이블 행렬을 반환합니다. 이 정보는 학습 또는 검증 과정에서 모델에 입력으로 제공됩니다.

이 예시에서는 `VOCDataset` 클래스의 `__getitem__` 메서드에서 특정 이미지에 대한 레이블 행렬을 생성하는 과정을 자세히 설명하겠습니다. 이 과정에서 이미지는 7x7 그리드로 나누어지며, 각 그리드 셀에 대한 정보를 저장합니다.

### 예시 데이터 설정
- 이미지 크기를 448x448으로 가정합니다 (YOLO의 일반적 설정).
- 한 이미지에 대해 다음과 같은 바운딩 박스가 있다고 가정하겠습니다:
  - 객체 1: 클래스 레이블 0, 중심 좌표 (112, 224), 너비 112, 높이 224
  - S (그리드 크기) = 7 이므로, 이미지는 7x7 그리드로 나눠집니다.

각 그리드 셀의 크기는 64x64 (448 / 7 = 64)입니다.

### 바운딩 박스의 처리
```python
boxes = torch.tensor([
    [0, 112 / 448, 224 / 448, 112 / 448, 224 / 448]  # class_label, x, y, width, height
])
```

#### 각 바운딩 박스 계산
```python
label_matrix = torch.zeros((7, 7, 25))  # C=20 클래스 + 5B (B=1)
for box in boxes:
    class_label, x, y, width, height = box.tolist()
    class_label = int(class_label)
    
    # 이미지 크기에 따른 상대 위치 계산
    i, j = int(7 * y), int(7 * x)  # 그리드 내 셀 위치
    x_cell, y_cell = 7 * x - j, 7 * y - i  # 셀 내 좌표
    width_cell, height_cell = width * 7, height * 7  # 셀 크기 대비 바운딩 박스 크기
    
    if label_matrix[i, j, 20] == 0:
        label_matrix[i, j, 20] = 1
        box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        label_matrix[i, j, 21:25] = box_coordinates
        label_matrix[i, j, class_label] = 1
```

### 상세 설명
- `(x, y, width, height)`는 이미지 크기에 대해 정규화된 값입니다.
- `i, j`는 바운딩 박스의 중심이 속하는 그리드 셀의 위치를 계산합니다. 예를 들어, x=0.25, y=0.5인 경우, `i = int(7 * 0.5) = 3`, `j = int(7 * 0.25) = 1`.
- `x_cell`과 `y_cell`은 그리드 셀 내에서의 바운딩 박스 중심 위치입니다. 예를 들어, `x = 0.25` (실제 위치 112)에서 `j = 1` 이므로, `x_cell = 7 * 0.25 - 1 = 0.75`.
- `width_cell`과 `height_cell`은 그리드 셀에 대한 바운딩 박스의 너비와 높이의 상대적 비율입니다.
- 바운딩 박스 정보 (`x_cell, y_cell, width_cell, height_cell`)와 클래스 레이블은 `label_matrix`의 해당 셀에 저장됩니다.

이 과정을 통해 각 이미지에 대해 각 그리드 셀에 저장할 정보를 설정하며, 이 정보는 YOLO 모델이 객체를 감지하고 분류하는 데 사용됩니다.


## train.py

``` python
import torch  # PyTorch, 딥러닝 모델을 만들고 훈련할 때 사용하는 라이브러리입니다.
import torchvision.transforms as transforms  # 이미지 데이터를 변환(예: 크기 변경, 텐서 변환)하는 도구입니다.
import torch.optim as optim  # 모델을 최적화할 때 사용하는 방법을 제공합니다. 여기서는 Adam 최적화를 사용합니다.
import torchvision.transforms.functional as FT
from tqdm import tqdm  # 진행 상태 바를 보여주는 라이브러리입니다. 훈련 진행 상태를 시각적으로 확인할 수 있게 해 줍니다.
from torch.utils.data import DataLoader  # 데이터를 모델에 배치(batch) 단위로 공급하는 역할을 합니다.
from model import Yolov1  # Yolo v1 모델을 가져옵니다.
from dataset import VOCDataset  # VOCDataset 클래스를 가져옵니다. 이 클래스는 Pascal VOC 데이터셋을 로드하는 역할을 합니다.
from utils import (  # 유틸리티 함수들을 가져옵니다.
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss  # YoloLoss 클래스를 가져옵니다. 이 클래스는 손실 함수를 정의합니다.

seed = 123  # 랜덤 시드를 설정합니다. 이 값으로 인해 코드를 여러 번 실행해도 같은 결과를 얻을 수 있습니다.
torch.manual_seed(seed)

# 하이퍼파라미터 설정 부분
LEARNING_RATE = 2e-5  # 학습률: 모델이 배우는 속도를 조절합니다.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU를 사용할 수 있으면 'cuda', 아니면 'cpu'를 사용합니다.
BATCH_SIZE = 16  # 한 번에 처리하는 데이터의 수입니다.
WEIGHT_DECAY = 0  # 가중치 감소: 모델의 복잡성을 제한하여 오버피팅을 방지합니다.
EPOCHS = 1000  # 전체 데이터셋을 몇 번 학습할지를 정하는 횟수입니다.
NUM_WORKERS = 8  # 데이터 로딩에 사용할 프로세스 수입니다.
PIN_MEMORY = True  # 데이터를 GPU 메모리에 고정시킬지 여부입니다.
LOAD_MODEL = True  # 학습된 모델을 불러올지 여부입니다.
LOAD_MODEL_FILE = "overfit.pth.tar"  # 불러올 모델 파일의 이름입니다.
IMG_DIR = "data/images"  # 이미지가 저장된 폴더의 경로입니다.
LABEL_DIR = "data/labels"  # 레이블이 저장된 폴더의 경로입니다.

class Compose(object):
    """
    여러 변환을 하나로 조합해서 적용하는 클래스입니다.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])  # 이미지를 448x448으로 리사이즈하고, 텐서로 변환합니다.

def train_fn(train_loader, model, optimizer, loss_fn):
    """
    훈련을 실행하는 함수입니다.
    """
    loop = tqdm(train_loader, leave=True)  # 진행 상태 바를 생성합니다.
    mean_loss = []  # 각 배치의 손실을 기록할 리스트입니다.

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)  # 데이터를 GPU나 CPU로 보냅니다.
        out = model(x)  # 모델에 데이터를 넣고 출력을 받습니다.
        loss = loss_fn(out, y)  # 출력과 실제 데이터를 비교하여 손실을 계산합니다.
        mean_loss.append(loss.item())  # 손실을 리스트에 추가합니다.
        optimizer.zero_grad()  # 최적화하기 전에 이전에 계산된 기울기를 초기화합니다.
        loss.backward()  # 손실을 기준으로 기울기를 계산합니다 (역전파).
        optimizer.step()  # 모델의 가중치를 갱신합니다.

        loop.set_postfix(loss=loss.item())  # 진행 상태 바에 현재 손실을 표시합니다.

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")  # 평균 손실을 출력합니다.



def main():
    """
    프로그램의 주 실행 함수입니다.
    """
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)  # 모델을 생성하고, GPU나 CPU로 옮깁니다.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Adam 최적화를 설정합니다.
    loss_fn = YoloLoss()  # 손실 함수를 생성합니다.

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)  # 저장된 모델을 불러옵니다.


    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        
        for x, y in test_loader:
           x = x.to(DEVICE)
           for idx in range(8):
               bboxes = cellboxes_to_boxes(model(x))
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

           import sys
           sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
           save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
           import time
           time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
```



## utils.py

``` py
import torch  # PyTorch 라이브러리를 임포트합니다.
import numpy as np  # numpy 라이브러리를 임포트합니다. 수학 연산에 사용됩니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 matplotlib의 pyplot을 임포트합니다.
import matplotlib.patches as patches  # matplotlib에서 그래픽 패치를 그리기 위해 patches를 임포트합니다.
from collections import Counter  # 데이터의 요소 개수를 셀 때 사용하는 Counter 클래스를 임포트합니다.

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    두 박스의 교차 영역 대 합집합(IoU)을 계산합니다.

    매개변수:
    boxes_preds (tensor): 예측된 바운딩 박스들 (BATCH_SIZE, 4)
    boxes_labels (tensor): 실제 바운딩 박스 레이블들 (BATCH_SIZE, 4)
    box_format (str): 'midpoint'는 박스가 중심점, 너비, 높이로 주어진 경우, 'corners'는 박스가 좌측 상단과 우측 하단 좌표로 주어진 경우

    반환값:
    tensor: 모든 예제에 대한 IoU 값
    """

    if box_format == "midpoint":
        # 박스 형식이 중심점 기준일 때 각 박스의 좌측 상단(x1, y1)과 우측 하단(x2, y2) 좌표를 계산합니다.
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        # 박스 형식이 좌표 기준일 때 각 좌표를 직접 사용합니다.
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # 교차 영역의 좌표를 구합니다.
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # 교차 영역이 없을 경우를 위해 clamp(0)을 사용합니다.
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # 각 박스의 영역을 계산합니다.
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # IoU를 계산합니다.
    return intersection / (box1_area + box2_area - intersection + 1e-6)
```

``` python
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    주어진 바운딩 박스 목록에서 Non-Max Suppression을 수행합니다.

    매개변수:
        bboxes (list): 모든 바운딩 박스의 목록이며, 각 바운딩 박스는 [class_pred, prob_score, x1, y1, x2, y2] 형식입니다.
        iou_threshold (float): 예측된 바운딩 박스가 정확하다고 간주되는 IoU 임계값입니다.
        threshold (float): 예측된 바운딩 박스를 제거하기 위한 확률 점수 임계값입니다 (IoU와는 독립적입니다).
        box_format (str): "midpoint" 또는 "corners", 바운딩 박스를 지정하는 형식입니다.

    반환값:
        list: 특정 IoU 임계값을 사용한 후 NMS를 수행한 후의 바운딩 박스 목록
    """

    assert type(bboxes) == list  # 입력된 bboxes가 리스트 타입인지 확인합니다.

    # 확률 임계값보다 높은 바운딩 박스만 필터링합니다.
    bboxes = [box for box in bboxes if box[1] > threshold]
    
    # 확률 점수에 따라 바운딩 박스를 내림차순으로 정렬합니다.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []  # NMS 후에 남은 바운딩 박스를 저장할 리스트입니다.

    # 바운딩 박스 목록이 비어있지 않은 동안 반복합니다.
    while bboxes:
        chosen_box = bboxes.pop(0)  # 가장 확률이 높은 바운딩 박스를 선택하고 목록에서 제거합니다.

        # 선택된 바운딩 박스와 IoU가 임계값보다 낮거나, 다른 클래스에 속하는 바운딩 박스만 남깁니다.
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # 다른 클래스의 경우
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),  # 선택된 박스의 좌표
                torch.tensor(box[2:]),  # 비교하는 박스의 좌표
                box_format=box_format,  # 박스 형식
            )
            < iou_threshold  # IoU 임계값보다 낮아야 합니다.
        ]

        # 선택된 박스를 결과 목록에 추가합니다.
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms  # 처리된 바운딩 박스 목록을 반환합니다.
```

``` python
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    예측된 바운딩 박스와 실제 바운딩 박스를 비교하여 각 클래스별 평균 정밀도(mAP)를 계산하고 이를 평균내어 반환합니다.

    매개변수:
        pred_boxes (list): [train_idx, class_prediction, prob_score, x1, y1, x2, y2] 형식의 예측된 바운딩 박스 목록입니다.
        true_boxes (list): [train_idx, class_prediction, prob_score, x1, y1, x2, y2] 형식의 실제 바운딩 박스 목록입니다.
        iou_threshold (float): 정밀도 계산에 사용할 IoU 임계값입니다.
        box_format (str): 바운딩 박스 좌표 형식입니다 ("midpoint" 또는 "corners").
        num_classes (int): 클래스의 총 수입니다.

    반환값:
        float: 모든 클래스에 대한 평균 mAP 값입니다.
    """

    average_precisions = []  # 각 클래스별로 계산된 정밀도를 저장할 리스트입니다.

    epsilon = 1e-6  # 수치적 안정성을 위한 작은 수입니다.

    # 각 클래스에 대해 반복합니다.
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # 예측 박스 중 현재 클래스에 해당하는 박스만 선택합니다.
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        # 실제 박스 중 현재 클래스에 해당하는 박스만 선택합니다.
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # 각 이미지별로 실제 박스의 수를 카운트합니다.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # 카운트된 결과를 바탕으로 각 이미지별 실제 박스의 존재 여부를 표시합니다.
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # 예측된 박스를 확률 점수에 따라 내림차순으로 정렬합니다.
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # 실제 박스가 없는 클래스는 계산에서 제외합니다.
        if total_true_bboxes == 0:
            continue

        # 모든 예측된 박스에 대해 반복하여 True Positive와 False Positive를 계산합니다.
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            best_iou = 0
            best_gt_idx = None

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold and best_gt_idx is not None:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1  # True Positive
                    amount_bboxes[detection[0]][best_gt_idx] = 1  # 해당 박스는 처리됨을 표시
                else:
                    FP[detection_idx] = 1  # False Positive
            else:
                FP[detection_idx] = 1  # IoU 임계값을 넘지 못하면 False Positive

        # 누적 합을 계산하여 Recall과 Precision을 계산합니다.
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # Recall 계산
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)  # Precision 계산
        precisions = torch.cat((torch.tensor([1]), precisions))  # 첫 번째 값으로 1 추가
        recalls = torch.cat((torch.tensor([0]), recalls))  # 첫 번째 값으로 0 추가

        # 정밀도와 재현율 곡선 아래의 면적을 계산하여 클래스별 평균 정밀도를 구합니다.
        average_precisions.append(torch.trapz(precisions, recalls))

    # 모든 클래스에 대한 평균 정밀도의 평균을 계산하여 반환합니다.
    return sum(average_precisions) / len(average_precisions)
```

``` python
def plot_image(image, boxes):
    """
    이미지 위에 예측된 바운딩 박스를 그립니다.

    매개변수:
    image (array): 바운딩 박스를 그릴 이미지입니다.
    boxes (list): 각 바운딩 박스의 정보를 담은 리스트입니다. 각 박스는 [class_pred, prob_score, x, y, w, h] 형식입니다.
    """

    # 이미지 데이터를 numpy 배열로 변환합니다.
    im = np.array(image)
    # 이미지의 높이, 너비, 색상 채널 수를 구합니다.
    height, width, _ = im.shape

    # matplotlib 라이브러리를 사용하여 이미지를 표시할 그림과 축을 생성합니다.
    fig, ax = plt.subplots(1)
    # 축에 이미지를 표시합니다.
    ax.imshow(im)

    # 모든 바운딩 박스에 대하여 반복합니다.
    for box in boxes:
        box = box[2:]  # 바운딩 박스의 좌표 정보만 가져옵니다.
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        
        # 바운딩 박스의 중심점 (x, y)와 너비 (w), 높이 (h)를 사용하여 왼쪽 상단 모서리의 좌표를 계산합니다.
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        
        # 계산된 위치와 크기 정보를 사용하여 직사각형 (Rectangle) 객체를 생성합니다.
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),  # 직사각형의 왼쪽 상단 모서리 위치
            box[2] * width,  # 직사각형의 너비
            box[3] * height,  # 직사각형의 높이
            linewidth=1,  # 직사각형 테두리의 두께
            edgecolor="r",  # 직사각형 테두리의 색상
            facecolor="none"  # 직사각형 내부의 색상 (없음)
        )
        
        # 생성한 직사각형을 그림의 축에 추가합니다.
        ax.add_patch(rect)

    # 그림을 표시합니다.
    plt.show()
```

``` python
def get_bboxes(
    loader,  # 데이터를 로드하는 DataLoader 객체입니다.
    model,   # 평가할 객체 탐지 모델입니다.
    iou_threshold,  # IoU 임계값입니다. 이 값을 넘는 박스만을 유효한 것으로 간주합니다.
    threshold,  # 예측 확률 임계값입니다. 이 값을 넘는 예측만을 유효한 것으로 간주합니다.
    pred_format="cells",  # 예측 데이터 형식입니다. 여기서는 'cells'로 지정되어 있습니다.
    box_format="midpoint",  # 바운딩 박스의 형식입니다. 'midpoint'는 중심 좌표와 크기로 박스가 정의됩니다.
    device="cuda",  # 계산을 수행할 디바이스입니다. 'cuda'는 GPU를 의미합니다.
):
    all_pred_boxes = []  # 모든 예측 박스를 저장할 리스트입니다.
    all_true_boxes = []  # 모든 실제 박스를 저장할 리스트입니다.

    model.eval()  # 모델을 평가 모드로 설정합니다. 이렇게 하면 훈련 중에 사용되는 일부 메커니즘이 비활성화됩니다.

    train_idx = 0  # 각 이미지에 대한 고유 인덱스를 추적합니다.

    for batch_idx, (x, labels) in enumerate(loader):  # DataLoader에서 배치를 순차적으로 가져옵니다.
        x = x.to(device)  # 이미지 데이터를 계산 디바이스로 옮깁니다.
        labels = labels.to(device)  # 레이블 데이터를 계산 디바이스로 옮깁니다.

        with torch.no_grad():  # 기울기 계산을 중단하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
            predictions = model(x)  # 모델을 사용하여 이미지에 대한 예측을 수행합니다.

        batch_size = x.shape[0]  # 배치 내의 이미지 수를 가져옵니다.
        true_bboxes = cellboxes_to_boxes(labels)  # 실제 레이블 데이터를 사용 가능한 형태로 변환합니다.
        bboxes = cellboxes_to_boxes(predictions)  # 모델의 예측을 사용 가능한 형태로 변환합니다.

        for idx in range(batch_size):  # 배치 내의 각 이미지에 대해 반복합니다.
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )  # 비최대 억제를 사용하여 중복 박스를 제거합니다.

            for nms_box in nms_boxes:  # 비최대 억제를 통과한 박스에 대해 반복합니다.
                all_pred_boxes.append([train_idx] + nms_box)  # 예측 박스 리스트에 추가합니다.

            for box in true_bboxes[idx]:  # 실제 박스에 대해 반복합니다.
                if box[1] > threshold:  # 확률 임계값보다 높은 박스만 추가합니다.
                    all_true_boxes.append([train_idx] + box)  # 실제 박스 리스트에 추가합니다.

            train_idx += 1  # 다음 이미지로 인덱스를 증가시킵니다.

    model.train()  # 모델을 다시 훈련 모드로 전환합니다.
    return all_pred_boxes, all_true_boxes  # 모든 예측 박스와 실제 박스를 반환합니다.
```

``` python
def convert_cellboxes(predictions, S=7):
    """
    YOLO 모델로부터의 출력을 전체 이미지 비율로 변환합니다.
    이 함수는 이미지가 SxS 그리드로 나뉘어진 것을 전제로 하며,
    각 셀에 대한 예측값을 전체 이미지에 대한 비율로 변환합니다.

    매개변수:
    predictions (tensor): 모델의 출력, 각 셀에 대한 예측 포함
    S (int): 이미지를 나누는 그리드의 크기 (기본값 7)

    반환값:
    tensor: 전체 이미지에 대한 예측 박스의 좌표와 클래스, 확률을 포함한 텐서
    """

    predictions = predictions.to("cpu")  # 예측을 CPU로 이동
    batch_size = predictions.shape[0]  # 배치 크기
    predictions = predictions.reshape(batch_size, S, S, 30)  # 예측을 (배치 크기, S, S, 30) 형태로 재구성

    bboxes1 = predictions[..., 21:25]  # 첫 번째 바운딩 박스 좌표
    bboxes2 = predictions[..., 26:30]  # 두 번째 바운딩 박스 좌표
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )  # 첫 번째 및 두 번째 박스의 신뢰도 점수

    best_box = scores.argmax(0).unsqueeze(-1)  # 가장 높은 신뢰도를 가진 박스 선택
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2  # 선택된 박스 좌표 계산

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)  # 각 셀의 인덱스 생성
    x = 1 / S * (best_boxes[..., :1] + cell_indices)  # 전체 이미지 상의 x 좌표 계산
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  # 전체 이미지 상의 y 좌표 계산
    w_y = 1 / S * best_boxes[..., 2:4]  # 너비와 높이 계산

    converted_bboxes = torch.cat((x, y, w_y), dim=-1)  # 변환된 좌표를 하나의 텐서로 결합
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)  # 가장 확률이 높은 클래스
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )  # 가장 높은 신뢰도 점수

    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )  # 최종 결과 텐서 생성

    return converted_preds  # 변환된 예측 결과 반환
```

``` python
def cellboxes_to_boxes(out, S=7):
    """
    YOLO 모델의 출력을 전체 이미지 좌표로 변환하여 각 객체에 대한 바운딩 박스 정보를 반환합니다.

    매개변수:
    out (tensor): 모델의 출력, 각 셀에 대한 예측이 포함된 텐서입니다.
    S (int): 이미지를 나누는 그리드의 크기 (기본값 7)

    반환값:
    list: 전체 이미지 좌표로 변환된 바운딩 박스 목록입니다.
    """

    # `convert_cellboxes` 함수를 사용하여 셀 기반 좌표를 전체 이미지 좌표로 변환합니다.
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)

    # 클래스 예측 값을 정수형으로 변환합니다. 이는 클래스 인덱스를 나타냅니다.
    converted_pred[..., 0] = converted_pred[..., 0].long()

    all_bboxes = []  # 각 이미지별로 모든 바운딩 박스를 저장할 리스트입니다.

    # 배치 내의 각 예제에 대해 반복합니다.
    for ex_idx in range(out.shape[0]):
        bboxes = []  # 한 예제의 모든 바운딩 박스를 저장할 리스트입니다.

        # 각 그리드 셀에 대해 반복합니다.
        for bbox_idx in range(S * S):
            # 현재 바운딩 박스의 정보를 추출하여 리스트로 변환합니다.
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])

        # 현재 예제의 모든 바운딩 박스를 추가합니다.
        all_bboxes.append(bboxes)

    return all_bboxes  # 모든 예제의 바운딩 박스 목록을 반환합니다.
```

``` python
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    학습 중인 모델의 상태를 파일로 저장합니다.

    매개변수:
    state (dict): 저장할 모델의 상태를 포함한 딕셔너리. 보통 모델의 파라미터, 최적화 상태 등이 포함됩니다.
    filename (str): 체크포인트 파일의 이름. 기본값은 'my_checkpoint.pth.tar' 입니다.

    반환값:
    없음
    """
    print("=> Saving checkpoint")  # 체크포인트 저장 시작 메시지 출력
    torch.save(state, filename)  # torch.save 함수를 사용하여 상태 딕셔너리를 파일로 저장


def load_checkpoint(checkpoint, model, optimizer):
    """
    저장된 체크포인트 파일로부터 모델의 상태를 불러옵니다.

    매개변수:
    checkpoint (dict): 불러올 체크포인트 파일에서 읽은 상태 정보가 담긴 딕셔너리입니다.
    model (torch.nn.Module): 상태를 불러올 모델 객체입니다.
    optimizer (torch.optim.Optimizer): 모델의 최적화를 담당하는 옵티마이저 객체입니다.

    반환값:
    없음
    """
    print("=> Loading checkpoint")  # 체크포인트 불러오기 시작 메시지 출력
    model.load_state_dict(checkpoint["state_dict"])  # 모델의 상태를 체크포인트에서 불러옵니다.
    optimizer.load_state_dict(checkpoint["optimizer"])  # 옵티마이저의 상태를 체크포인트에서 불러옵니다.
```