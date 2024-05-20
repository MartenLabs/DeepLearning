
\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{url}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{lipsum}
\usepackage{multicol}
\usepackage{xcolor}
\usepackage{natbib}
\usepackage{kotex}
\usepackage{hyperref}
\usepackage{wrapfig}
\usepackage[dvips]{epsfig}

\usepackage{enumitem}% http://ctan.org/pkg/enumitem
\usepackage{algorithm, algorithmicx, algpseudocode}

\usepackage[font=small]{caption}
\addtolength{\abovecaptionskip}{-3mm}
\addtolength{\textfloatsep}{-5mm}
\setlength\columnsep{20pt}

\usepackage[a4paper,left=1.50cm, right=1.50cm, top=1.50cm, bottom=1.50cm]{geometry}


\usepackage[rightcaption]{sidecap}



\author{}

\title{경량 딥러닝 아키텍처 기반 저해상도 열화상 실시간 객체 검출 기술 - Low cost MCU 적용 연구}

\begin{document}
	
	\begin{center}
		{\Large \textbf{저해상도 열화상 카메라를 활용한 MCU 기반 실시간 객체 검출 기술 연구}} \\ 
		\vspace{1em}
		{\large Jun Hee Lee}\\
		\vspace{1em}
		\textit{Tech University of Korea}
	\end{center}
	

	\begin{center}
		\rule{150mm}{0.2mm}
	\end{center}		

	\begin{abstract}
	
            본 논문에서는 STM32와 같은 초저사양 MCU(Micro-Controller Unit) 환경에서 동작할 수 있는 효율적인 열화상 이미지 기반 Object detection 딥러닝 모델을 제안한다. 저해상도 열화상 이미지의 특성상 겹쳐진 사람을 정확히 분류하는 것이 어려운 과제이다. 이를 해결하기 위해 본 논문은 박스 좌표 예측, 클래스 분류, F1 점수를 결합한 복합 손실 함수를 제안한다. 제안된 손실 함수는 CutMix와 하드 네거티브 마이닝, 하드 포지티브 마이닝 기법을 적용하여 데이터셋 불균형 문제를 완화하고, 어려운 샘플에 더 집중할 수 있도록 설계되었다. 이론적 증명과 실험 결과를 통해 제안 손실 함수의 타당성과 효과를 입증하였다. 본 연구는 초저사양 임베디드 기기에서도 정확하고 효율적인 열화상 기반 Object detection이 가능함을 보여준다. \\
	
	% \textbf{Collaborators}: 
	\end{abstract}

	\begin{center}
		\rule{150mm}{0.2mm}
	\end{center}		

	\vspace{5mm}

 \begin{multicols*}{2}

\section{Introduction}
 
% \subsection{Formatting} 
 
사물인터넷, 지능형 감시, 웨어러블 디바이스 등의 분야에서 초저사양 MCU에서 동작할 수 있는 효율적인 딥러닝 모델의 필요성이 증가하고 있다. 특히, 열화상 카메라는 다양한 환경에서 작동할 수 있어 주목받고 있지만, 열화상 이미지의 낮은 해상도와 명확하지 않은 물체 경계로 인해 기존 딥러닝 모델을 적용하기 어렵다.
본 연구에서는 STM32와 같은 초저사양 MCU 환경에서 동작 가능한 효율적인 열화상 기반 객체 검출 딥러닝 모델을 제안한다. 저해상도 열화상에서 겹쳐진 물체를 정확히 분류하기 위해 복합 손실 함수를 제안하고, 데이터 불균형과 어려운 샘플 문제를 해결하기 위해 CutMix와 하드 네거티브 마이닝 기법을 적용한다. 제안 모델의 이론적 타당성을 증명하고 실험을 통해 효과를 검증하여, 초저사양 임베디드 기기에서의 정확하고 효율적인 열화상 기반 객체 검출 가능성을 제시한다.  \newline

\noindent 본 연구의 주요 기여는 다음과 같다:
\begin{itemize} \setlength{\itemsep}{-1mm}
    \item 저해상도, 단일 채널 열화상에서의 실시간 객체 검출에 최적화된 경량 CNN 아키텍처를 제안한다. 
    \item 제안된 모델이 열화상 도메인에서 사용 가능한 제한된 정보에도 불구하고 관심 객체를 정확하게 감지할 수 있음을 입증한다. 
    \item Low-cost MCU 환경에서 효율적이고 빠른 검출이 가능한 열화상 기반 객체 검출 모델을 제안한다. 
    \item 저해상도 열화상 이미지에서 겹쳐진 사람을 정확히 분류하기 위한 손실 함수를 제안한다.
\end{itemize}

\noindent 본 연구는 초저사양 임베디드 기기에서의 효율적인 열화상 기반 사람 검출을 가능하게 함으로써, 지능형 감시, 재난 구조, 건강 관리 등 다양한 응용 분야에 기여할 수 있을 것으로 기대된다. 또한, 제안 손실 함수와 데이터 불균형 및 어려운 샘플 문제에 대한 접근 방식은 유사한 도전 과제를 다루는 다른 연구에도 활용될 수 있을 것이다. \\

\section{Related Work} 

\subsection{MobileNet}
MobileNet[1]은 모바일 및 임베디드 비전 애플리케이션을 위해 설계된 경량 CNN 아키텍처이다. 이 모델은 depthwise separable convolution을 사용하여 모델 크기와 연산량을 대폭 줄였다. Depthwise separable convolution은 채널별 공간 컨볼루션과 pointwise 컨볼루션을 결합하여 기존 컨볼루션과 유사한 성능을 유지하면서도 파라미터 수와 연산량을 크게 감소시킨다. 또한, MobileNet은 폭 승수(width multiplier)와 해상도 승수(resolution multiplier)를 도입하여 모델의 폭과 입력 해상도를 조정함으로써 정확도와 리소스 사용량 간의 균형을 맞출 수 있도록 하였다. MobileNet은 ImageNet 데이터셋에서 높은 정확도를 달성하였으며, 다양한 모바일 애플리케이션에 성공적으로 적용되었다.

\subsection{Single Shot MultiBox Detector}
SSD[2]는 객체 검출을 위한 단일 샷 검출기로, 실시간 처리 속도와 높은 정확도를 동시에 달성하였다. SSD는 다양한 크기의 디폴트 박스(default box)를 사용하여 객체의 위치와 크기를 예측한다. 이를 위해 CNN의 여러 레이어에서 특징 맵을 추출하고, 각 특징 맵에 대해 디폴트 박스를 적용하여 객체의 존재 여부와 위치를 예측한다. SSD는 객체 검출 문제를 회귀 문제와 분류 문제로 동시에 처리하며, Non-Maximum Suppression을 사용하여 중복된 박스를 제거한다. 또한, SSD는 하드 네거티브 마이닝을 적용하여 배경과 객체를 구분하는 능력을 향상시켰다. SSD는 PASCAL VOC와 COCO 데이터셋에서 우수한 성능을 보였으며, 실시간 객체 검출 애플리케이션에 널리 사용되고 있다.
\newline

본 연구에서는 MobileNet의 경량화 기법과 SSD의 Single shot 검출 아이디어를 결합하여, 초저사양 MCU 환경에 적합한 열화상 기반 객체 검출 모델을 설계하고자 한다. MobileNet의 depthwise separable convolution을 활용하여 모델 크기와 연산량을 최소화하고, SSD의 다중 스케일 특징 맵과 디폴트 박스를 사용하여 다양한 크기의 객체를 효과적으로 검출할 수 있을 것으로 기대된다. 또한, 하드 네거티브 마이닝을 적용하여 열화상 이미지에서의 객체와 배경 구분 능력을 개선하고자 한다.

\end{multicols*}

\section{Proposed Method}


\vspace{1cm}

% \begin{figure}[h]
%     \centering
%     \includegraphics[width=1\linewidth]{figures/BackBone.png}
%     \caption{Backbone architecture}
%     \label{fig:enter-label}
% \end{figure}


\vspace{1cm}

\subsection{Overview}
본 연구에서 제안하는 객체 검출 모델은 저해상도 열화상 이미지에서의 객체 검출 성능을 높이기 위해 설계되었다. 모델의 전체 구조는 그림 \ref{fig:model_overview}와 같이 Backbone 네트워크, Feature Pyramid Network (FPN), Detection Head의 세 부분으로 구성된다.

\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{figures/Architecture.png}
    \caption{제안하는 객체 검출 모델의 전체 구조}
    \label{fig:model_overview}
\end{figure}

Backbone 네트워크는 입력 이미지로부터 다양한 스케일의 특징 맵을 추출하는 역할을 한다. 그림 \ref{fig:backbone}은 Backbone 네트워크의 상세 구조를 보여준다. 입력 이미지는 먼저 Conv 모듈을 통과하여 초기 특징을 추출한다. 이후 MaxPool 모듈로 노이즈를 줄이고, Spatial Attention 모듈을 통해 특징 맵의 공간적 정보를 강화한다. \newline

\noindent Backbone 네트워크는 세 가지 스케일의 특징 맵 ($p1$, $p2$, $p3$)을 출력한다. 첫 번째 스케일의 특징 맵 $p1$은 Conv 모듈과 SPPFast 모듈을 통해 생성된다. 두 번째 스케일의 특징 맵 $p2$를 얻기 위해, $p1$은 Conv 모듈, MaxPool 모듈, DepthwiseConv 모듈, Spatial Attention 모듈을 거친 후 이전 단계의 특징 맵과 Concatenate된다. 마지막으로 $p2$는 SPPFast 모듈을 통과하여 $p2$ 출력을 생성한다.
세 번째 스케일의 특징 맵 $p3$ 또한 유사한 과정을 거친다. $p2$ 출력은 Conv 모듈, MaxPool 모듈, DepthwiseConv 모듈, Spatial Attention 모듈을 차례로 통과한 후, 이전 단계의 특징 맵과 Concatenate되어 최종 $p3$ 출력을 생성한다.
이러한 Backbone 네트워크 구조는 입력 이미지로부터 다양한 수용 영역을 가지는 특징 맵을 추출할 수 있도록 설계되었다. Spatial Attention 모듈은 특징 맵의 공간적 정보를 강화하여 객체 검출 성능을 향상시킨다. 또한 SPPFast 모듈과 Concatenation 연산을 통해 다양한 스케일의 특징을 효과적으로 융합할 수 있다. \newline

\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{figures/BackBone.png}
    \caption{Backbone 네트워크의 구조}
    \label{fig:backbone}
\end{figure}

\noindent FPN은 Backbone 네트워크에서 추출된 세 가지 스케일의 특징 맵을 입력으로 받아, 상향식 및 하향식 특징 융합을 수행한다. 이를 통해 다양한 스케일의 특징 정보를 효과적으로 통합하여 보다 풍부한 특징 표현을 생성한다. \newline

\noindent Detection Head는 FPN에서 생성된 특징 맵을 기반으로 객체의 위치와 클래스를 예측한다. 각 스케일의 특징 맵은 Depthwise Separable Convolution을 거쳐 채널 수를 조정하고, 최종적으로 각 앵커 박스에 대한 경계 상자 좌표와 클래스 확률을 출력한다.

\noindent 모델의 학습 과정에서는 분류 손실 (Focal Loss), 박스 손실 (Huber Loss), 확신 손실 (Binary Cross-entropy Loss), F-beta Loss를 결합한 최종 손실 함수를 사용한다. 이 손실 함수는 Hard Negative Mining을 통해 어려운 배경 샘플에 대한 학습을 강화하고, 각 손실 함수의 가중치를 조절하여 모델의 성능을 최적화한다.

\noindent 제안된 모델 아키텍처는 저해상도 열화상 이미지의 특성을 고려하여 설계되었다. Backbone 네트워크에서는 다양한 모듈을 통해 입력 이미지로부터 의미 있는 특징을 추출하고, FPN에서는 이를 효과적으로 융합하여 풍부한 특징 표현을 생성한다. Detection Head에서는 이를 바탕으로 객체의 위치와 클래스를 정확히 예측한다.
\newpage

\subsubsection{객체 검출 모델 아키텍처 (Object Detection Model Architecture)}
제안하는 객체 검출 모델은 크게 세 가지 부분으로 구성된다: Backbone 네트워크, Feature Pyramid Network (FPN), 그리고 Detection Head. 각 부분은 저해상도 열화상 이미지에서의 객체 검출 성능을 높이기 위해 설계되었으며, 그 구조와 기능은 다음과 같다.

\paragraph{Backbone 네트워크}
Backbone 네트워크는 입력 이미지로부터 다양한 스케일의 특징 맵 (feature map)을 추출하는 역할을 한다. 본 연구에서는 경량화된 Backbone 구조를 제안하여 모델의 계산 효율성을 높이고자 하였다.

\noindent Backbone 네트워크의 첫 단계는 입력 이미지를 Conv 모듈을 통과시켜 초기 특징을 추출하는 것이다. 추출된 특징은 MaxPool 모듈을 통해 노이즈가 감소되고, Convolution Block Attention Module을 거쳐 채널 및 공간적 정보가 강화된다.
이후 Conv 모듈과 SPPFast (Spatial Pyramid Pooling Fast) 모듈을 통해 첫 번째 스케일의 특징 맵 $p1$이 생성된다. $p1$은 다시 Conv 모듈, MaxPool 모듈, DepthwiseConv 모듈, Spatial Attention 모듈을 차례로 통과한 후, 이전 단계의 특징 맵과 Concatenate되어 두 번째 스케일의 특징 맵 $p2$를 생성한다. $p2$ 또한 SPPFast 모듈을 통과하여 최종 $p2$ 출력이 된다. \newline

\noindent 마지막으로, $p2$ 출력은 Conv 모듈, MaxPool 모듈, DepthwiseConv 모듈, Spatial Attention 모듈을 거쳐 세 번째 스케일의 특징 맵 $p3$를 생성한다. 이 과정에서도 이전 단계의 특징 맵과의 Concatenation이 수행된다. 

\noindent 이러한 Backbone 네트워크 구조는 입력 이미지로부터 다양한 수용 영역을 가지는 특징 맵을 효과적으로 추출할 수 있도록 설계되었다. Spatial Attention 모듈은 특징 맵의 공간적 정보를 강화하여 객체 검출 성능을 향상시키며, SPPFast 모듈과 Concatenation 연산을 통해 다양한 스케일의 특징이 효과적으로 융합된다.
최종적으로, Backbone 네트워크는 세 가지 스케일 ($p1$, $p2$, $p3$)의 특징 맵을 출력하며, 이는 FPN으로 전달되어 추가적인 특징 융합이 이루어진다. \newline


\paragraph{Feature Pyramid Network (FPN)}
FPN은 Backbone 네트워크에서 추출된 다양한 스케일의 특징 맵을 융합하여 보다 풍부한 특징 표현을 생성하는 역할을 한다. 본 연구에서는 FPN의 개념을 차용하되, 저해상도 열화상 이미지에 적합하도록 수정된 구조를 제안한다.
제안된 FPN 구조는 세 가지 스케일의 특징 맵 ($p1$, $p2$, $p3$)을 입력으로 받는다. 먼저, 각 특징 맵은 $1 \times 1$ Convolution을 통해 채널 수를 조정한다. 그 다음, $p3$ 특징 맵은 2배 업샘플링되어 $p2$의 크기에 맞춰지고, $p1$ 특징 맵은 2배 다운샘플링되어 $p2$의 크기에 맞춰진다. 이렇게 크기가 조정된 $p1$, $p2$, $p3$ 특징 맵은 채널 방향으로 연결 (concatenation)되어 정보를 융합한다.
융합된 $p2$ 특징 맵은 $3 \times 3$ Depthwise Convolution을 거쳐 $p2_{out}$을 생성한다. 한편, 융합된 $p2$ 특징 맵은 2배 다운샘플링되어 $p3$의 크기에 맞춰지고, $p3$ 특징 맵과 연결된다. 이렇게 생성된 $p3$ 특징 맵은 $3 \times 3$ Depthwise Convolution을 거쳐 $p3_{out}$을 생성한다.
제안된 FPN 구조는 다양한 스케일의 특징 정보를 효과적으로 융합할 뿐만 아니라, Depthwise Convolution을 사용하여 연산량을 줄이면서도 특징 맵의 공간적 정보를 보존할 수 있다. 이는 저해상도 열화상 이미지에서 작은 객체나 경계가 불명확한 객체를 검출하는데 도움이 되는것을 실험적으로 확인하였다. \newline

\clearpage

\paragraph{Detection Head}
Detection Head는 FPN에서 생성된 특징 맵을 기반으로 실제 객체의 위치와 클래스를 예측하는 역할을 한다. 본 연구에서는 간단하면서도 효과적인 Detection Head 구조를 제안한다. \newline

\noindent Detection Head는 FPN의 출력인 두 가지 스케일의 특징 맵 ($p2_{out}$, $p3_{out}$)을 입력으로 받는다. 각 특징 맵은 독립적으로 처리되며, 동일한 구조의 레이어를 거친다.

\noindent 먼저, 특징 맵은 Depthwise Separable Convolution을 두 번 거친다. 첫 번째 Depthwise Separable Convolution은 $3 \times 3$ 커널을 사용하며 출력 채널 수는 128이다. 두 번째 Depthwise Separable Convolution은 $1 \times 1$ 커널을 사용하며, 출력 채널 수는 각 위치 (position)에서의 앵커 박스 (anchor box) 수와 예측해야 할 값의 수에 따라 결정된다.

\noindent 각 앵커 박스마다 4개의 경계 상자 좌표 값 (bounding box coordinates)과 $num\_classes$개의 클래스 확률 값을 예측해야 하므로, 총 $num\_anchors\_per\_location \times (4 + num\_classes)$개의 채널을 출력한다. 이렇게 생성된 출력은 $(N, H \times W \times num\_anchors\_per\_location, 4 + num\_classes)$ 크기의 텐서로 reshape된다. 여기서 $N$은 배치 크기, $H$와 $W$는 특징 맵의 높이와 너비이다.
최종적으로, 모든 스케일의 출력을 연결 (concatenation)하여 $(N, num\_total\_anchors, 4 + num\_classes)$ 크기의 텐서를 생성한다. 이 텐서가 모델의 최종 출력이 되며, 각 앵커 박스에 대한 경계 상자 좌표와 클래스 확률을 나타낸다. \newline

\noindent 이러한 Detection Head 구조는 Depthwise Separable Convolution을 사용하여 파라미터 수를 줄이면서도 효과적으로 객체의 위치와 클래스를 예측할 수 있다. 또한, 다양한 스케일의 특징 맵을 독립적으로 처리하고 연결하는 방식을 통해, 서로 다른 크기의 객체를 유연하게 검출할 수 있다.
제안된 객체 검출 모델 아키텍처는 Backbone 네트워크, 수정된 FPN 구조, 그리고 효율적인 Detection Head의 조화로운 구성을 통해 저해상도 열화상 이미지에서의 객체 검출 성능을 높이고자 한다. Backbone 네트워크에서는 입력 이미지로부터 다양한 스케일의 특징 맵을 추출하고, FPN에서는 이러한 특징 맵들을 융합하여 보다 풍부하고 정제된 특징 표현을 생성한다. 마지막으로 Detection Head에서는 FPN의 출력을 기반으로 객체의 위치와 클래스를 효과적으로 예측한다. \newline

\noindent 이러한 통합적인 접근 방식을 통해, 제안된 모델은 저해상도 열화상 이미지의 특성을 고려하면서도 높은 객체 검출 성능을 달성하며 특히, FPN 구조의 개선과 Detection Head의 최적화를 통해, 작은 객체나 경계가 불명확한 객체에 대한 검출 정확도를 높일 수 있다.

\clearpage

\subsection{CutMix Data Augmentation}

객체 탐지(Object Detection) 문제는 이미지 내에 존재하는 물체의 위치와 클래스를 동시에 예측하는 작업이다. 이때 학습 데이터셋 내에서 물체가 존재하는 경우(Positive 샘플)와 물체가 존재하지 않는 경우(Negative 샘플)가 모두 포함되어 있다. 그런데 실제 데이터셋에서는 Positive 샘플의 비율이 상대적으로 낮은 경우가 많이 발생한다. 이는 데이터 수집 과정에서 비어있는 배경 이미지가 많이 포함되거나, 작은 물체들이 있는 이미지가 다수 포함되기 때문이다. \newline

\noindent Positive 샘플의 비율이 낮으면 객체 탐지 모델이 Positive 샘플을 충분히 학습하지 못하게 되어 성능 저하가 발생할 수 있다. 모델은 Positive 샘플을 통해 물체의 특징을 제대로 학습해야 하는데, 이러한 샘플이 부족하면 모델이 일반화 능력을 잘 갖추지 못하게 된다. 결과적으로 테스트 데이터에 대해 낮은 정확도를 보이게 되는 것이다.

\noindent이러한 문제를 해결하기 위해 CutMix 알고리즘을 적용하여 Positive 샘플의 비율을 증가시켰다. CutMix는 데이터 증강(Data Augmentation) 기법 중 하나로, 기존 이미지에서 물체 영역을 잘라내어 다른 이미지에 붙여넣는 방식이다. 이를 통해 기존 이미지에 새로운 Positive 샘플을 추가할 수 있게 된다.

\noindent구체적으로는 데이터셋 내 이미지들 중 Positive 샘플이 있는 이미지를 선택하여, 그 이미지에서 물체 영역을 잘라낸다. 그리고 다른 이미지에 그 영역을 붙여넣는다. 이때 기존 이미지의 물체 영역과 겹치지 않도록 IoU(Intersection over Union) 임계값을 설정하여 제약을 둔다. 이렇게 하면 기존 이미지에 새로운 Positive 샘플이 추가되면서도, 기존 물체 정보는 유지할 수 있다. \newline

\noindent 이 과정을 반복하여 원하는 수준으로 Positive 샘플의 비율을 높일 수 있다. 결과적으로 학습 데이터셋 내 Positive 샘플의 수가 증가하게 되어, 객체 탐지 모델이 Positive 샘플을 더 많이 학습할 수 있게 된다. 이를 통해 모델의 일반화 능력이 향상되고, 테스트 데이터에 대한 정확도 또한 높아질 것으로 기대할 수 있다. \newline



\noindent CutMix 알고리즘의 과정은 다음과 같다.

\begin{algorithm}
\caption{CutMix for Object Detection}
\begin{algorithmic}[1]

\Require Dataset $\mathcal{D} = \{(x_i, y_i)\}{i=1}^N$, where $x_i$ is an image and $y_i = \{(b_j, c_j)\}{j=1}^{M_i}$ is a set of bounding boxes $b_j$ and class labels $c_j$ for $M_i$ objects in $x_i$. Hyperparameters: maximum number of objects $M$, IoU threshold $\tau$. 

\Ensure Augmented dataset $\mathcal{D}'$ 
\State $\mathcal{D}' \gets \emptyset$

\For{$(x, y)$ in $\mathcal{D}$} 
    \State $x' \gets x$
    \State $y' \gets y$
    \State $M' \gets |y'|$
    \While{$M' < M$}
        \State Sample a donor image $(x_d, y_d)$ from $\mathcal{D}$ such that $|y_d| > 0$
        \State Sample an object $(b_d, c_d)$ from $y_d$
        \State $\mathcal{B} \gets \{b \in y' \mid \text{IoU}(b, b_d) < \tau\}$
        \If{$\mathcal{B} \neq \emptyset$}
            \State $x'[\text{region}(b_d)] \gets x_d[\text{region}(b_d)]$
            \State $y' \gets y' \cup \{(b_d, c_d)\}$
            \State $M' \gets M' + 1$
        \EndIf
    \EndWhile
    \State $\mathcal{D}' \gets \mathcal{D}' \cup \{(x', y')\}$
\EndFor
\State \Return $\mathcal{D}'$
\end{algorithmic}
\end{algorithm}


\begin{figure}[h]
    \centering
    \includegraphics[width=0.4\linewidth]{figures/cutmix2.jpg}
    \hspace{1cm}
    \includegraphics[width=0.4\linewidth]{figures/cutmix1.jpg} 
    \caption{CutMix Augmentation}
\end{figure}

\newpage




\subsection{Hard Negative Mining}

하드 네거티브 마이닝은 객체 탐지 모델의 성능을 개선하기 위한 기법 중 하나이다. 이 기법은 모델이 잘못 예측한 어려운 네거티브 샘플(Hard Negative Sample)에 더 집중하도록 함으로써 모델의 일반화 능력을 향상시킨다. \newline

\noindent 객체 탐지 문제에서 네거티브 샘플(Negative Sample)은 이미지 내에 물체가 없는 경우를 의미한다. 전통적인 학습 방식에서는 모든 네거티브 샘플에 대해 동일한 가중치를 부여하여 학습을 진행한다. 그러나 이렇게 되면 쉬운 네거티브 샘플에 비해 어려운 네거티브 샘플에 대한 학습이 부족해질 수 있다. \newline

\noindent 하드 네거티브 마이닝은 이러한 문제를 해결하기 위해 고안되었다. 이 기법은 모델이 잘못 예측한 어려운 네거티브 샘플에 더 많은 가중치를 부여하여 학습함으로써, 모델이 이러한 샘플에 더 잘 적응하도록 한다.  \newline

\noindent 하드 네거티브 마이닝 알고리즘은 다음과 같이 동작한다.

\begin{algorithm}
\caption{Hard Negative Mining}
\begin{algorithmic}[1]
\Require Training set $\mathcal{T} = \{(x_i, y_i)\}_{i=1}^N$, where $x_i$ is an image and $y_i = \{(b_j, c_j)\}_{j=1}^{M_i}$ is a set of bounding boxes $b_j$ and class labels $c_j$ for $M_i$ objects in $x_i$. Hyperparameters: negative-positive ratio $r$, maximum number of hard negatives $k$.
\State Train the model on $\mathcal{T}$
\State Compute classification loss $L_{cls}$ and F1 score loss $L_{f1}$ on $\mathcal{T}$
\State Compute negative classification loss $L_{cls, neg}$ on negative samples
\State $N_{pos} \gets \sum_{i=1}^N \mathbb{1}[M_i > 0]$  \Comment{Number of positive samples} \hfill 
\State $k \gets \lfloor r \times N_{pos} \rfloor$        \Comment{Maximum number of hard negatives} \hfill 
\State $L_{hard\_neg} \gets \emptyset$                   \Comment{Hard negative losses} \hfill 
\State $L_{cls, neg} \gets \text{sort\_descending}(L_{cls, neg})$ \Comment{Sort negative losses in descending order}
\For{$i=1$ to $k$}
    \State $L_{hard\_neg} \gets L_{hard\_neg} \cup \{(L_{cls, neg}[i], L_{f1, neg}[i])\}$
\EndFor
\State $L_{pos} \gets \frac{1}{N_{pos}} \sum_{(x, y) \in \mathcal{T}, M > 0} (L_{cls, pos} + L_{f1, pos} + L_{box, pos})$
\State $L_{hard\_neg} \gets \frac{1}{k} \sum_{(l_c, l_f) \in L_{hard\_neg}} (l_c + l_f)$
\State $L_{cls} \gets \sqrt{\max(L_{pos, cls} \times L_{hard\_neg, cls}, \epsilon)}$
\State $L_{f1} \gets \sqrt{\max(L_{pos, f1} \times L_{hard\_neg, f1}, \epsilon)}$
\State $L_{combined} \gets \sqrt{\max(L_{cls} \times L_{f1}, \epsilon)}$
\State $L_{final} \gets w_{cls} \times L_{combined} + w_{box} \times L_{pos, box}$
\State Update the model using $L_{final}$
\end{algorithmic}
\end{algorithm}


\noindent 본 논문에서 제안한 손실 함수에서는 하드 네거티브 마이닝 기법을 적용하였다. 구체적인 과정은 다음과 같다.

\begin{enumerate}
    \item 모델을 통해 분류 손실($L_{cls}$)과 F1 점수 손실($L_{f1}$)을 계산한다.
    \item 네거티브 샘플에 대한 분류 손실($L_{cls, neg}$)을 계산한다.
    \item $L_{cls, neg}$가 높은 상위 k개의 샘플을 하드 네거티브 샘플로 선택한다. 여기서 k는 $N_{pos} \times r$로 결정되며, $N_{pos}$는 포지티브 샘플의 수, $r$은 하이퍼파라미터(네거티브-포지티브 비율)이다.
    \item 하드 네거티브 샘플에 대한 $L_{cls, hard\_neg}$와 $L_{f1, hard\_neg}$를 계산하고, 이를 포지티브 샘플의 손실과 결합하여 최종 손실 함수를 구성한다.
\end{enumerate}

\noindent 이렇게 함으로써 모델은 어려운 네거티브 샘플에 더 집중하게 되어 일반화 능력이 향상된다. 특히 저해상도 열화상 이미지에서 겹친 물체 분류 문제의 경우, 배경과 물체를 구분하기 어려운 상황이 많기 때문에 하드 네거티브 마이닝 기법이 효과적으로 물체를 분류하는데 기여하는것을 확인하였다. \newline

\clearpage


\subsection{Hard Positive Mining}

하드 네거티브 마이닝과 마찬가지로 하드 포지티브 마이닝도 객체 탐지 모델의 성능을 향상시키기 위한 기법 중 하나이다. 이 기법은 모델이 예측하기 어려운 포지티브 샘플(Hard Positive Sample)에 더 집중하도록 함으로써 모델의 일반화 능력을 향상시킨다. \newline

\noindent 객체 탐지 문제에서 포지티브 샘플(Positive Sample)은 이미지 내에 물체가 있는 경우를 의미한다. 그러나 물체의 크기가 작거나, 물체가 겹쳐 있거나, 배경과 유사한 색상을 가지고 있는 등의 이유로 모델이 예측하기 어려운 포지티브 샘플이 존재할 수 있다. \newline

\noindent 하드 포지티브 마이닝은 이러한 어려운 포지티브 샘플에 더 많은 가중치를 부여하여 학습함으로써, 모델이 이러한 샘플에 더 잘 적응하도록 한다. \newline

\noindent 하드 포지티브 마이닝 알고리즘은 다음과 같이 동작한다.

\begin{algorithm}
\caption{Hard Positive Mining}
\begin{algorithmic}[1]
\Require Training set $\mathcal{T} = \{(x_i, y_i)\}_{i=1}^N$, where $x_i$ is an image and $y_i = \{(b_j, c_j)\}_{j=1}^{M_i}$ is a set of bounding boxes $b_j$ and class labels $c_j$ for $M_i$ objects in $x_i$. Hyperparameters: hard positive ratio $r$, maximum number of hard positives $k$.
\State Train the model on $\mathcal{T}$
\State Compute classification loss $L_{cls}$ on $\mathcal{T}$
\State Compute positive classification loss $L_{cls, pos}$ on positive samples
\State $N_{pos} \gets \sum_{i=1}^N \mathbb{1}[M_i > 0]$ \Comment{Number of positive samples} \hfill
\State $k \gets \lfloor r \times N_{pos} \rfloor$ \Comment{Maximum number of hard positives} \hfill
\State $L_{hard\_pos} \gets \emptyset$ \Comment{Hard positive losses} \hfill
\State $L_{cls, pos} \gets \text{sort\_descending}(L_{cls, pos})$ \Comment{Sort positive losses in descending order}
\For{$i=1$ to $k$}
\State $L_{hard\_pos} \gets L_{hard\_pos} \cup \{L_{cls, pos}[i]\}$
\EndFor
\State $L_{hard\_pos} \gets \frac{1}{k} \sum_{l \in L_{hard\_pos}} l$
\State $L_{cls, pos} \gets L_{hard\_pos}$
\State Update the model using $L_{cls, pos}$
\end{algorithmic}
\end{algorithm}

\noindent 본 논문에서 제안한 손실 함수에서는 하드 네거티브 마이닝과 하드 포지티브 마이닝 기법을 함께 적용하였다. 구체적인 과정은 다음과 같다.

\begin{enumerate}
\item 모델을 통해 분류 손실($L_{cls}$), F1 점수 손실($L_{f1}$), confidence 손실($L_{conf}$)을 계산한다.
\item 네거티브 샘플에 대한 confidence 손실($L_{conf, neg}$)을 계산하고, 이 중 상위 $k_{neg}$개의 샘플을 하드 네거티브 샘플로 선택한다. 여기서 $k_{neg}$는 $N_{pos} \times r_{neg}$로 결정되며, $N_{pos}$는 포지티브 샘플의 수, $r_{neg}$은 하이퍼파라미터(네거티브-포지티브 비율)이다.
\item 포지티브 샘플에 대한 분류 손실($L_{cls, pos}$)을 계산하고, 이 중 상위 $k_{pos}$개의 샘플을 하드 포지티브 샘플로 선택한다. 여기서 $k_{pos}$는 $N_{pos} \times r_{pos}$로 결정되며, $r_{pos}$는 하이퍼파라미터(하드 포지티브 비율)이다.
\item 하드 네거티브 샘플에 대한 $L_{cls, hard\_neg}$, $L_{conf, hard\_neg}$와 하드 포지티브 샘플에 대한 $L_{cls, hard\_pos}$, $L_{f1, hard\_pos}$를 계산하고, 이를 포지티브 샘플의 손실과 결합하여 최종 손실 함수를 구성한다.
\end{enumerate}

\noindent 이렇게 함으로써 모델은 어려운 네거티브 샘플과 포지티브 샘플에 더 집중하게 되어 일반화 능력이 향상된다. 특히 저해상도 열화상 이미지에서 겹친 물체 분류 문제의 경우, 배경과 물체를 구분하기 어려운 상황이 많고, 물체의 크기가 작거나 물체가 겹쳐 있는 경우가 많기 때문에 하드 네거티브 마이닝과 하드 포지티브 마이닝 기법이 효과적으로 물체가 겹쳐 있는 경우와 노이즈 오탐지율을 낮추는데 기여하는것을 실험적으로 확인하였다. \newline

\clearpage



\subsection{Loss Function}

본 연구에서는 객체 탐지 문제에 대해 복합적인 손실 함수를 제안한다. 이 손실 함수는 박스 좌표 예측(Box Regression), 복합 클래스 분류(Focal Loss, Confidence Loss)를 사용하여 Positive와 Hard Negative 샘플을 구분하여 다룬다. 
\newline

\subsubsection{Box Loss}
박스 손실(Box Loss) $L_{\text{box}}$는 예측된 박스 $y_{\text{pred}}$와 실제 박스 $y_{\text{true}}$ 간의 차이를 기반으로 계산된다. \newline

$$
L_{\text{box}} = \sum_{i}\begin{cases}
0.5 \times (y_{\text{true},i} - y_{\text{pred},i})^2 & \text{if } |y_{\text{true},i} - y_{\text{pred},i}| \leq \delta\\
|y_{\text{true},i} - y_{\text{pred},i}| - 0.5 \times \delta & \text{otherwise}
\end{cases}
$$ \newline

\noindent 여기서 $\delta$는 하이퍼파라미터이다. 이 손실 함수는 작은 오차에 대해서는 제곱 오차를, 큰 오차에 대해서는 절대 오차를 사용하여 박스 좌표 예측 성능을 향상시킨다. \newline

\subsubsection{Focal Loss}
Focal Loss는 객체 검출 모델에서 각 앵커 박스가 어떤 클래스에 속하는지를 예측하는 데 사용되는 손실 함수이다. 이 손실 함수는 클래스 불균형 문제를 해결하고, Hard Negative Mining을 수행하여 어려운 음성 샘플에 대한 학습을 강화하는 데 중점을 둔다. Focal Loss는 다음과 같이 정의된다: \newline

$$FL(p_t) = -\alpha_t \times (1-p_t)^\gamma \times \log(p_t)$$ \newline

$$
\alpha_t = \begin{cases}
\alpha & \text{if } y_{\text{true}} = 1\\
1 - \alpha & \text{otherwise}
\end{cases}
$$

$$
p_t = \begin{cases}
y_{\text{pred}} & \text{if } y_{\text{true}} = 1\\
1 - y_{\text{pred}} & \text{otherwise}
\end{cases}
$$ \newline

\noindent $p_t$는 모델이 예측한 실제 클래스에 대한 확률을 나타낸다. 만약 앵커 박스가 실제 클래스에 속한다면 $p_t = y_{\text{pred}}$이고, 그렇지 않다면 $p_t = 1-y_{\text{pred}}$이다. $\alpha_t$는 클래스 불균형을 다루기 위한 가중치로, 실제 클래스가 양성일 때는 $\alpha$, 음성일 때는 $1-\alpha$의 값을 가진다. $\gamma$는 손실 함수의 포커싱 파라미터로, 어려운 샘플에 대한 손실 함수의 가중치를 조절한다. \newline
\noindent Focal Loss의 핵심은 $(1-p_t)^\gamma$ 항을 통해 어려운 샘플에 대한 손실 함수의 가중치를 높이는 것이다. 이 항은 모델이 잘못 분류한 샘플, 즉 $p_t$가 작은 샘플에 대해 손실 함수의 값을 크게 만든다. 반면, 모델이 정확하게 분류한 샘플, 즉 $p_t$가 큰 샘플에 대해서는 손실 함수의 값을 작게 만든다. 이를 통해 모델은 쉬운 샘플보다 어려운 샘플에 더 집중하게 되며, Hard Negative Mining의 효과를 얻을 수 있다. \newline

\noindent $\gamma$ 값을 조절함으로써, Focal Loss는 어려운 샘플에 대한 학습 강도를 조절할 수 있다. $\gamma$ 값이 클수록 어려운 샘플에 대한 손실 함수의 가중치가 커지므로, 모델은 이러한 샘플에 더 집중하게 된다. 하지만 $\gamma$ 값이 너무 크면 손실 함수가 불안정해질 수 있으므로, 일반적으로 5 이상으로 설정하지 않는 것이 좋다. \newline
\noindent $\alpha$ 값은 클래스 불균형 문제를 해결하는 데 사용된다. 객체 검출에서는 일반적으로 배경 클래스의 샘플 수가 객체 클래스의 샘플 수보다 훨씬 많다. 이러한 불균형은 모델이 배경 클래스에 편향되어 학습되는 결과를 초래할 수 있다. $\alpha$ 값을 조절함으로써, Focal Loss는 이러한 불균형을 완화할 수 있다. 일반적으로 $\alpha$ 값은 0.5 이상으로 설정하지 않는 것이 좋은데, 이는 $\alpha$ 값이 너무 크면 모델이 양성 샘플에만 집중하여 Precision이 낮아질 수 있기 때문이다. \newline
\noindent 저해상도 열화상 이미지에서 객체 검출 시, Focal Loss를 사용하면 어려운 음성 샘플에 대한 분류 성능을 높일 수 있다. 열화상 이미지에서는 객체의 특징이 명확하지 않고, 배경과의 구분이 모호한 경우가 많아 객체 검출이 어렵다. 이러한 환경에서 Focal Loss를 사용하면, 모델이 어려운 음성 샘플에 더 집중하여 학습할 수 있으므로 객체 검출 성능을 향상시킬 수 있다. \newline

\noindent 또한, Focal Loss는 클래스 불균형 문제를 해결하는 데에도 도움이 된다. 열화상 이미지에서는 배경 클래스의 샘플 수가 객체 클래스의 샘플 수보다 훨씬 많은 경우가 일반적이다. Focal Loss를 사용하면 이러한 불균형을 완화하여, 모델이 객체 클래스에 더 집중할 수 있도록 유도할 수 있다. \newline

\subsubsection{Confidence Loss}
Confidence Loss는 객체 검출 모델에서 각 앵커 박스가 객체를 포함하는지 여부를 예측하는 데 사용되는 손실 함수이다. 이 손실 함수의 주요 목적은 모델의 Recall과 Precision 간의 균형을 조절하여 객체 검출 성능을 최적화하는 것이다. Confidence Loss는 다음과 같이 정의된다: \newline

$$L_{conf} = -\alpha y \log(\sigma(x)) - \beta (1-y) \log(1-\sigma(x))$$ \newline

\noindent 여기서 $y$는 실제 객체 존재 여부를 나타내는 이진 변수로, 객체가 있는 경우 1, 없는 경우 0의 값을 가진다. $x$는 모델이 예측한 객체 존재 확률을 나타내며, $\sigma(x)$는 시그모이드 함수를 통해 확률 값으로 변환된 결과이다. $\alpha$와 $\beta$는 손실 함수의 가중치로, 각각 모델의 Recall과 Precision에 영향을 준다. \newline

\noindent $\alpha$ 값을 크게 설정할수록 모델은 더 많은 앵커 박스를 객체로 분류하려는 경향을 보인다. 이는 실제 객체를 포함하는 앵커 박스를 놓치지 않도록 하는 데 도움이 되므로, Recall이 높아지는 효과가 있다. 반면, $\beta$ 값을 크게 설정할수록 모델은 객체가 아닌 앵커 박스를 배경으로 분류하려는 경향을 보인다. 이는 실제로 객체가 아닌 앵커 박스를 객체로 잘못 분류하는 것을 방지하므로, Precision이 높아지는 효과가 있다. \newline

\noindent Confidence Loss에서 $\alpha$와 $\beta$ 값을 적절히 조절함으로써, 모델은 객체 검출 시 Recall과 Precision 간의 균형을 유지할 수 있다. 이는 저해상도 열화상 이미지와 같이 객체 검출이 어려운 환경에서 특히 중요하다. 열화상 이미지에서는 객체의 경계가 명확하지 않고, 배경과의 대비가 낮아 객체를 정확히 검출하기 어렵기 때문이다. Confidence Loss를 사용하면 이러한 어려운 환경에서도 모델이 객체의 존재 여부를 더 정확하게 예측할 수 있게 된다. \newline

\noindent Confidence Loss의 효과를 극대화하기 위해서는 $\alpha$와 $\beta$ 값을 데이터셋의 특성과 모델의 구조에 맞게 적절히 설정하는 것이 중요하다. $\alpha$와 $\beta$ 값을 너무 크게 설정하면 모델이 특정 클래스에 편향되어 학습할 수 있으므로 주의해야 한다. \newline

\noindent 저해상도 열화상 이미지에서 객체 검출 시, Confidence Loss를 사용하면 모델이 객체의 존재 여부를 더 정확하게 예측할 수 있다. 이는 열화상 이미지의 특성상 객체와 배경의 구분이 모호한 경우가 많기 때문에, 객체 검출 성능을 높이는 데 큰 도움이 된다. 또한, Confidence Loss는 모델의 Recall과 Precision 간의 균형을 조절할 수 있는 유연성을 제공하므로, 다양한 응용 분야에서 활용될 수 있다. \newline 

\noindent 결론적으로, 저해상도 열화상 이미지에서 객체 검출 시, Confidence Loss와 Focal Loss를 함께 사용하면 모델의 성능을 크게 향상시킬 수 있다. Confidence Loss는 객체의 존재 여부를 예측하는 데 도움을 주고, Focal Loss는 어려운 음성 샘플에 대한 분류 성능을 높이는 데 기여한다. 이 두 손실 함수를 적절히 조합하고, 하이퍼파라미터를 최적화함으로써, 열화상 이미지에서의 객체 검출 정확도를 높일 수 있다.  \newline

\subsubsection{F-beta Loss}

F-beta Loss는 객체 검출 모델의 성능을 평가하는 데 사용되는 F-beta 점수를 손실 함수로 활용한 것이다. F-beta 점수는 모델의 정밀도(Precision)와 재현율(Recall)을 결합한 지표로, 객체 검출 문제에서 널리 사용된다. F-beta Loss는 모델이 F-beta 점수를 직접 최적화하도록 학습시킴으로써, 정밀도와 재현율 간의 균형을 조절할 수 있다. \newline

\noindent F-beta 점수는 다음과 같이 정의된다: \newline
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$ \newline
\noindent 여기서 $\beta$는 정밀도와 재현율 간의 상대적 중요도를 조절하는 하이퍼파라미터이다. $\beta$ 값이 1보다 크면 재현율에 더 큰 가중치를 부여하고, 1보다 작으면 정밀도에 더 큰 가중치를 부여한다. 일반적으로 $\beta=1$인 경우를 F1 점수라고 하며, 정밀도와 재현율을 동등하게 고려한다. \newline

\noindent 이 손실 함수를 사용할 때는 $\beta$ 값을 적절히 설정하는 것이 중요하다. 저해상도 열화상 이미지에서 객체 검출 문제의 경우, 작은 객체나 겹친 객체를 검출하는 것이 중요하므로 재현율에 더 큰 가중치를 부여하는 것이 좋다. 따라서 $\beta$ 값을 1보다 크게 설정하는 것이 일반적이다. \newline

\noindent F-beta Loss의 장점은 모델이 정밀도와 재현율의 조화를 고려하며 학습할 수 있다는 점이다. 이는 실제 응용 환경에서 중요한 특성이다. 예를 들어, 자율 주행 차량의 객체 검출 시스템에서는 재현율이 높아야 한다. 즉, 실제 객체를 최대한 놓치지 않아야 한다. 반면, 의료 영상 분석에서는 정밀도가 높아야 한다. 즉, 잘못된 양성 예측을 최소화해야 한다. F-beta Loss를 사용하면 이러한 도메인 특성을 고려하여 모델을 학습시킬 수 있다. \newline

\noindent 또한, F-beta Loss는 불균형 데이터셋에서도 효과적이다. 객체 검출 문제에서는 배경 클래스의 샘플 수가 객체 클래스의 샘플 수보다 훨씬 많은 경우가 일반적이다. 이런 상황에서 F-beta Loss를 사용하면 소수 클래스의 샘플에 더 큰 가중치를 부여할 수 있으므로, 모델이 소수 클래스를 더 잘 검출할 수 있게 된다. \newline
\noindent 결론적으로, F-beta Loss는 객체 검출 모델의 성능을 향상시키는 데 매우 유용한 손실 함수이다. 특히 저해상도 열화상 이미지와 같이 객체 검출이 어려운 환경에서는 F-beta Loss의 효과가 더욱 두드러진다. 정밀도와 재현율의 균형을 적절히 조절함으로써, 다양한 응용 분야에 맞는 객체 검출 모델을 학습시킬 수 있다. 본 연구에서는 F-beta Loss를 다른 손실 함수와 함께 사용하여, 저해상도 열화상 이미지에서의 객체 검출 성능을 크게 향상시켰다.


\subsubsection{최종 손실 함수 (Final Loss Function)}
최종 손실 함수는 객체 검출 모델의 성능을 종합적으로 최적화하기 위해 설계된 함수로, 분류 손실 (Classification Loss), F-beta 손실 (F-beta Loss), 박스 손실 (Box Loss), 그리고 확신 손실 (Confidence Loss)을 결합하여 계산된다. 이 네 가지 손실 함수는 객체 검출 모델의 서로 다른 측면을 다루며, 각각의 역할은 다음과 같다.
\noindent 분류 손실은 모델이 객체의 클래스를 정확하게 예측할 수 있도록 학습시키는 데 사용된다. 이 손실 함수는 일반적으로 Focal Loss를 사용하여 계산되며, 이는 클래스 불균형 문제를 해결하고 어려운 샘플에 대한 학습을 강화하는 데 효과적이다. \newline

\noindent F-beta 손실은 모델의 정밀도(Precision)와 재현율(Recall)의 조화평균인 F-beta 점수를 최적화하는 데 사용된다. 이 손실 함수는 객체 검출 문제에서 널리 사용되는 평가 지표인 F-beta 점수를 직접 최적화함으로써, 모델이 정밀도와 재현율의 균형을 맞출 수 있도록 돕는다. \newline

\noindent 박스 손실은 모델이 객체의 경계 상자 (bounding box)를 정확하게 예측할 수 있도록 학습시키는 데 사용된다. 이 손실 함수는 일반적으로 Huber Loss를 사용하여 계산되며, 이는 이상치에 강건한 특성을 가진다. \newline

\noindent 확신 손실은 모델이 객체의 존재 여부를 정확하게 예측할 수 있도록 학습시키는 데 사용된다. 이 손실 함수는 $\alpha$와 $\beta$ 값을 적절히 조절함으로써, 모델은 객체 검출 시 Recall과 Precision 간의 균형을 유지할 수 있다. \newline

\noindent 최종 손실 함수는 분류 손실, F-beta 손실, 박스 손실, 그리고 확신 손실을 결합하여 계산된다. 구체적으로, 분류 손실과 F-beta 손실의 기하평균을 계산하고, 이를 박스 손실 및 확신 손실과 가중합하여 최종 손실을 구한다: \newline
$$L_{cls} = \sqrt{L_{cls\_positive} \times L_{fbeta\_positive} + \epsilon}$$ \newline
$$L_{final} = w_{cls} \times L_{cls} + w_{box} \times L_{box\_positive} + L_{conf\_positive}$$ \newline
\noindent 여기서 $w_{cls}$와 $w_{box}$는 분류 손실과 박스 손실에 대한 가중치이며, $\epsilon$은 수치 안정성을 위한 작은 상수이다.
\noindent 최종 손실 함수의 중요한 특징 중 하나는 Hard Negative Mining과 Hard Positive Mining을 수행한다는 것이다. Hard Negative Mining은 배경 (negative) 샘플 중에서 모델이 가장 어려워하는 샘플에 집중하여 학습하는 기법이며, Hard Positive Mining은 객체 (positive) 샘플 중에서 모델이 가장 어려워하는 샘플에 집중하여 학습하는 기법이다. 이를 통해 모델은 어려운 샘플에 대한 구분 능력을 향상시킬 수 있다. \newline

\noindent Hard Negative Mining을 위해, 배경 샘플에 대한 확신 손실을 계산하고, 손실 값이 큰 상위 $k_{neg}$개의 샘플을 선택한다. 이 $k_{neg}$는 양성 샘플 수의 $neg\_pos\_ratio$배로 계산된다. 선택된 Hard Negative 샘플들은 $hard\_negative\_mask$를 통해 표시된다.
Hard Positive Mining을 위해, 객체 샘플에 대한 분류 손실을 계산하고, 손실 값이 큰 상위 $k_{pos}$개의 샘플을 선택한다. 이 $k_{pos}$는 양성 샘플 수의 $hard\_pos\_ratio$배로 계산된다. 선택된 Hard Positive 샘플들은 $hard\_positive\_mask$를 통해 표시된다.
$hard\_negative\_mask$와 $hard\_positive\_mask$는 양성 샘플 마스크 ($positive\_mask$)와 함께 사용되어, 분류 손실, F-beta 손실, 그리고 확신 손실을 계산할 때 고려된다. 이를 통해 모델은 어려운 배경 샘플과 객체 샘플에 대한 구분 능력을 향상시킬 수 있다. \newline

\noindent 최종적으로, 최종 손실 함수는 각 손실 함수의 값을 양성 샘플, Hard Negative 샘플, 그리고 Hard Positive 샘플의 수로 정규화한다. 이는 배치 (batch) 내의 샘플 수 변화에 강건한 손실 값을 얻기 위한 것이다.
\noindent 이러한 구현을 통해, 최종 손실 함수는 객체 검출 모델의 성능을 종합적으로 최적화할 수 있다. 분류 손실, F-beta 손실, 박스 손실, 그리고 확신 손실을 적절히 조합하고, Hard Negative Mining과 Hard Positive Mining을 통해 어려운 샘플에 대한 학습을 강화함으로써, 모델은 저해상도 열화상 이미지에서도 높은 객체 검출 성능을 달성할 수 있다. \newline

\noindent 특히, 저해상도 열화상 이미지는 객체 검출 task에 있어 많은 도전 과제를 안고 있다. 객체의 경계가 흐릿하고, 배경과의 대비가 낮으며, 객체 간의 겹침이 빈번하게 발생하기 때문이다. 이러한 상황에서 일반적인 손실 함수를 사용하는 것은 모델의 성능을 제한할 수 있다.
그러나 분류 손실에 Focal Loss를 적용하고, F-beta 손실을 도입함으로써, 모델은 클래스 불균형 문제를 해결하고, 정밀도와 재현율의 균형을 맞추며, 어려운 샘플에 대한 학습을 강화할 수 있다. 이는 저해상도 열화상 이미지에서 작은 객체나 겹쳐진 객체를 정확히 검출하는 데 큰 도움을 준다.
또한, 박스 손실에 Huber Loss를 사용함으로써, 모델은 객체의 경계 상자를 보다 정확하게 예측할 수 있다. Huber Loss는 이상치에 강건한 특성을 가지고 있어, 열화상 이미지에서 흔히 발생하는 부정확한 경계 상자 어노테이션에 대해서도 안정적인 학습이 가능하다. \newline

\noindent 확신 손실은 모델이 객체의 존재 여부를 정확히 예측하도록 돕는다. 저해상도 열화상 이미지에서는 객체와 배경을 구분하는 것이 쉽지 않기 때문에, 이 손실 함수는 모델이 객체 존재 여부를 보다 명확히 판단할 수 있게 해준다.
여기에 Hard Negative Mining과 Hard Positive Mining을 적용하면, 모델은 저해상도 열화상 이미지에서 자주 등장하는 어려운 배경 샘플과 객체 샘플에 대한 구분 능력을 크게 향상시킬 수 있다. 이는 전체적인 객체 검출 성능의 향상으로 이어진다. \newline

\noindent 결론적으로, 분류 손실, F-beta 손실, 박스 손실, 확신 손실을 적절히 조합하고, Hard Negative Mining과 Hard Positive Mining을 통해 어려운 샘플에 대한 학습을 강화한 최종 손실 함수는 저해상도 열화상 이미지에서의 객체 검출 문제를 효과적으로 다룰 수 있다.  \newpage

\subsection{Result}




\clearpage

\begin{thebibliography}{20}
    \bibitem{howard2017mobilenets} 
    A. G. Howard et al., "Mobilenets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.
    
    \bibitem{liu2016ssd} 
    Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., Berg, A. C. (2016). SSD: Single shot multibox detector. In European conference on computer vision (pp. 21-37). Springer, Cham.

    \bibitem{}
    S. Woo et al., "CBAM: Convolutional block attention module," in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 3-19.
\end{thebibliography}


\end{document}

