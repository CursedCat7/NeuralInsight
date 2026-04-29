# NeuralInsight: The Transparent MLP Visualizer

*다른 언어로 보기: [English](README.md)*

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**NeuralInsight**는 다층 퍼셉트론(MLP)의 동작 원리와 블랙박스를 시각적으로 풀어내는 스터디용 웹앱입니다. 행렬 표기법을 통해 역전파 미분 과정을 동적으로 렌더링하고 가중치 업데이트를 실시간으로 시각화하여 학습과정 원리를 쉽게 보고 공부하기 위해 만들었습니다.

**라이브 데모**: Streamlit 클라우드에 배포하여 온라인 웹으로 사용이 가능합니다! [Streamlit 클라우드에서 실행해보기](https://neuralinsight.streamlit.app/)

> **참고**: 이 프로젝트는 현재 한국어만 지원하고 있습니다. 차후 영문 및 다국어 지원 기능이 추가 개발될 예정입니다.

## 주요 기능 (Features)

- **동적 아키텍처 지원**: `3,5,4,2`와 같이 원하는 네트워크 구조를 자유롭게 정의하면, 노드와 가중치 행렬 표가 즉시 알맞게 생성됩니다.
- **수식 자동 유도 및 렌더링**: 설정된 네트워크 구조에 맞춰 순전파(Forward pass) 행렬 연산($Z = WX + b$)과 역전파 체인룰 수식을 $\LaTeX$ 포맷으로 자동 생성하고 렌더링합니다.
- **인터랙티브 데이터 에디터**: 엑셀 형태의 표에서 가중치(Weights)와 편향(Biases), 입력값을 직접 클릭해 수정하고, 수학적 연산 흐름이 어떻게 변하는지 실시간으로 확인할 수 있습니다.
- **학습 애니메이션 및 히트맵**: 연속 학습 모드에서 사용자가 설정한 속도에 맞춰 동적인 Loss 그래프와 Plotly 가중치 히트맵 애니메이션을 제공합니다.
- **커스텀 데이터셋 로드**: 보유하고 있는 CSV 데이터셋을 업로드하여, 다양한 옵티마이저(SGD, Momentum, Adam)와 활성화 함수 환경에서 모델이 어떻게 수렴하는지 관찰할 수 있습니다.

## 로컬 설치 및 셋업 (Setup)

1. 레포지토리 클론:
```bash
git clone https://github.com/CursedCat7/NeuralInsight.git
cd NeuralInsight
```

2. 필수 패키지를 설치:
```bash
pip install -r requirements.txt
```

*(참고: 네트워크 그래프를 정상적으로 렌더링하려면 로컬구동환경에 *Graphviz가 설치되어 있어야 합니다.)*

## 실행 방법 (Usage)

로컬 환경에서 ㅎ Streamlit 애플리케이션을 실행:

```bash
streamlit run app.py
```

명령어를 실행하면 브라우저가 열리며 `http://localhost:8501`에서 웹앱이 시작됩니다.

## 기여 (Contributing)

오픈소스 커뮤니티를 통해 이 프로젝트를 더 나은 교육용 툴로 만들기 위한 모든 기여를 **진심으로 환영합니다**.

1. 프로젝트를 포크(Fork).
2. 기능 브랜치를 생성. (`git checkout -b feature/AmazingFeature`)
3. 변경 사항을 커밋. (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시. (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성해 주세요! 검토 후 반영하겠습니다!

## 라이선스 (License)

이 프로젝트는 교육용으로 개발하였으며, MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 확인하세요.
