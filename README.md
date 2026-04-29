# NeuralInsight: The Transparent MLP Visualizer

*Read this in other languages: [한국어(Korean)](README_ko.md)*

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An interactive, math-driven web application that demystifies Multi-Layer Perceptrons (MLPs). It dynamically renders backpropagation calculus using matrix notation and visualizes weight updates in real-time.

**Live Demo**: [Try it on Streamlit Cloud](https://neuralinsight.streamlit.app/)

> **Note**: This project currently only supports Korean. English and additional multilingual support will be developed in future updates.

## Features

- **Dynamic Architecture**: Define any network structure (e.g., `3,5,4,2`) and watch the nodes, edges, and weight matrices adapt instantly.
- **Math & Matrix Automation**: Automatically generates step-by-step LaTeX equations ($Z = WX + b$) for forward passes and chain-rule derivations for backpropagation based on your custom architecture.
- **Interactive Data Editor**: Tweak weights, biases, and input values directly within an Excel-like grid to see how the mathematical flow changes.
- **Training Animation & Heatmaps**: Train the network continuously with real-time Plotly heatmaps and dynamic loss charts.
- **Custom Datasets**: Upload your own CSV datasets to observe how the model converges under different optimizers (SGD, Momentum, Adam) and activation functions.

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/CursedCat7/NeuralInsight.git
cd NeuralInsight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

*(Note: Graphviz must be installed on your system to render the network graphs properly.)*

## Usage

Run the Streamlit application locally:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
