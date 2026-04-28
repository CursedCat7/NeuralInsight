import streamlit as st
import numpy as np
from mlp_engine import MLP
import pandas as pd
import time
import plotly.express as px
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="NeuralInsight", layout="wide")

st.sidebar.header("Configuration")
st.sidebar.subheader("네트워크 아키텍처 (Architecture)")
input_dim = st.sidebar.number_input("입력층 노드 수 (Input)", min_value=1, max_value=5, value=2)
num_hidden = st.sidebar.number_input("은닉층 개수 (Hidden Layers)", min_value=0, max_value=3, value=1)
hidden_dims = []
for i in range(num_hidden):
    h_dim = st.sidebar.number_input(f"은닉층 {i+1} 노드 수", min_value=1, max_value=8, value=2)
    hidden_dims.append(h_dim)
output_dim = st.sidebar.number_input("출력층 노드 수 (Output)", min_value=1, max_value=5, value=1)

layers = [input_dim] + hidden_dims + [output_dim]
layers_str = ",".join(map(str, layers))

current_input_size = layers[0]
current_output_size = layers[-1]

act_choice = st.sidebar.selectbox("활성화 함수 (Activation)", ["sigmoid", "relu", "tanh", "softmax"])
loss_choice = st.sidebar.selectbox("손실 함수 (Loss)", ["mse", "cross_entropy"])
opt_choice = st.sidebar.selectbox("최적화 알고리즘 (Optimizer)", ["sgd", "momentum", "adam"])
lr = st.sidebar.number_input("Learning Rate (η)", value=0.1, step=0.01)

if 'step' not in st.session_state:
    st.session_state.step = 0

if 'mlp' not in st.session_state or st.session_state.get('layers_str') != layers_str or \
   st.session_state.get('act') != act_choice or st.session_state.get('loss') != loss_choice or \
   st.session_state.get('opt') != opt_choice:
    st.session_state.layers_str = layers_str
    st.session_state.act = act_choice
    st.session_state.loss = loss_choice
    st.session_state.opt = opt_choice
    st.session_state.mlp = MLP(layers=layers, activation=act_choice, loss=loss_choice, optimizer=opt_choice)
    st.session_state.step = 0

st.sidebar.markdown("---")
st.sidebar.subheader("Weights & Biases (가중치/편향)")
st.sidebar.info("표 안의 숫자를 직접 클릭하여 편집하세요.")

W_list = []
b_list = []
num_layers = len(layers)
for i in range(num_layers - 1):
    in_size = layers[i]
    out_size = layers[i+1]
    w_key = f'W{i+1}_df'
    b_key = f'b{i+1}_df'
    
    if w_key not in st.session_state or st.session_state[w_key].shape != (out_size, in_size):
        # Initialize
        init_w = np.random.randn(out_size, in_size) * 0.1
        if i == 0 and layers_str == "2,2,1":
            init_w = np.array([[0.1, 0.2], [0.3, 0.4]])
        elif i == 1 and layers_str == "2,2,1":
            init_w = np.array([[0.5, 0.6]])
            
        st.session_state[w_key] = pd.DataFrame(init_w, columns=[f'in_{j+1}' for j in range(in_size)], index=[f'out_{k+1}' for k in range(out_size)])
        st.session_state[b_key] = pd.DataFrame(np.zeros((out_size, 1)), columns=['Bias'], index=[f'out_{k+1}' for k in range(out_size)])
    
    st.sidebar.markdown(f"**$W^{{({i+1})}}$ (Layer {i} $\\rightarrow$ {i+1})**")
    edited_W = st.sidebar.data_editor(st.session_state[w_key], key=f"w{i+1}_editor")
    st.sidebar.markdown(f"**$b^{{({i+1})}}$ (Layer {i+1} Bias)**")
    edited_b = st.sidebar.data_editor(st.session_state[b_key], key=f"b{i+1}_editor")
    
    W_list.append(edited_W.values)
    b_list.append(edited_b.values)

st.sidebar.markdown("---")
st.sidebar.subheader("Input & Target (단일 데이터 편집)")
st.sidebar.info("Step-by-Step 모드에서 사용할 X, Y 값을 편집하세요.")

if 'X_df' not in st.session_state or st.session_state['X_df'].shape[0] != current_input_size:
    init_X = np.ones((current_input_size, 1)) * 0.5
    if layers_str == "2,2,1":
        init_X = np.array([[0.5], [0.1]])
    st.session_state['X_df'] = pd.DataFrame(init_X, columns=['Value'], index=[f'x_{i+1}' for i in range(current_input_size)])

if 'Y_df' not in st.session_state or st.session_state['Y_df'].shape[0] != current_output_size:
    init_Y = np.ones((current_output_size, 1)) * 0.7
    if layers_str == "2,2,1":
        init_Y = np.array([[0.7]])
    st.session_state['Y_df'] = pd.DataFrame(init_Y, columns=['Value'], index=[f'y_{i+1}' for i in range(current_output_size)])

col_x, col_y = st.sidebar.columns(2)
with col_x:
    st.markdown("**Input ($X$)**")
    edited_X = st.data_editor(st.session_state['X_df'], key="x_editor")
with col_y:
    st.markdown("**Target ($Y$)**")
    edited_Y = st.data_editor(st.session_state['Y_df'], key="y_editor")

X_manual = edited_X.values
Y_manual = edited_Y.values

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Loader")
uploaded_file = st.sidebar.file_uploader("연속 학습용 CSV 업로드", type=['csv'])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        X_train = df_uploaded.iloc[:, :current_input_size].values.T
        Y_train = df_uploaded.iloc[:, current_input_size:current_input_size+current_output_size].values.T
        st.sidebar.success(f"{X_train.shape[1]}개 데이터 로드 완료! (연속 학습 전용)")
        X = X_manual
        Y = Y_manual
    except Exception as e:
        st.sidebar.error("CSV 파싱 오류")
        X = X_manual
        Y = Y_manual
        X_train, Y_train = X, Y
else:
    X = X_manual
    Y = Y_manual
    X_train, Y_train = X, Y

if st.sidebar.button("Apply Parameters", use_container_width=True):
    st.session_state.mlp.set_params(W_list, b_list)
    st.session_state.step = 0
    st.sidebar.success("Parameters applied successfully!")

# Ensure forward and backward are calculated for step-by-step
st.session_state.mlp.forward(X)
st.session_state.mlp.backward(Y)

def numpy_to_latex_bmatrix(arr):
    lines = []
    for row in arr:
        lines.append(" & ".join([f"{x:.4f}" for x in row]))
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"

def draw_mlp_d3_html(mlp, layers, step):
    nodes = []
    edges = []
    
    width = 800
    height = 500
    layer_width = width / (len(layers) + 1)
    max_nodes = max(layers)
    
    for l_idx, num_nodes in enumerate(layers):
        x = (l_idx + 1) * layer_width
        node_spacing = height / (num_nodes + 1)
        
        is_input = (l_idx == 0)
        is_output = (l_idx == len(layers) - 1)
        
        if is_input:
            color = "#E8F5E9" if step not in [1, 6, 7] else "#C8E6C9"
            stroke = "#4CAF50"
            glow_type = "blue" if step in [1, 6] else ("red" if step == 7 else "")
        elif is_output:
            color = "#FFF3E0" if step not in [1,2,3,6,7] else "#FFE0B2"
            stroke = "#FF9800"
            glow_type = "blue" if step in [1, 6] else ("red" if step in [2, 3, 7] else "")
        else:
            color = "#E3F2FD" if step not in [1,4,6,7] else "#BBDEFB"
            stroke = "#2196F3"
            glow_type = "blue" if step in [1, 6] else ("red" if step in [4, 7] else "")
            
        for n_idx in range(num_nodes):
            y = (n_idx + 1) * node_spacing
            y_offset = (height - (num_nodes + 1) * node_spacing) / 2
            y += y_offset
            
            node_id = f"L{l_idx}_N{n_idx}"
            
            if is_input:
                label_html = f"x<tspan baseline-shift='sub' font-size='12'>{n_idx+1}</tspan>"
                val_text = f"Input: {mlp.cache['activations'][0][n_idx, 0]:.2f}"
            else:
                if is_output:
                    label_html = f"ŷ<tspan baseline-shift='sub' font-size='12'>{n_idx+1}</tspan>"
                else:
                    label_html = f"h<tspan baseline-shift='super' font-size='10'>({l_idx})</tspan><tspan baseline-shift='sub' font-size='12'>{n_idx+1}</tspan>"
                
                z_val = mlp.cache['zs'][l_idx-1][n_idx, 0]
                a_val = mlp.cache['activations'][l_idx][n_idx, 0]
                val_text = f"z: {z_val:.2f} | a: {a_val:.2f}"
            
            nodes.append({
                "id": node_id,
                "x": x,
                "y": y,
                "label": label_html,
                "val_text": val_text,
                "color": color,
                "stroke": stroke,
                "glow_type": glow_type
            })
            
    for l_idx in range(len(layers) - 1):
        W = mlp.weights[l_idx]
        is_active = (step in [1, 6, 7]) or (step == 5 - l_idx)
        is_forward = (step in [1, 6])
        
        for out_n in range(layers[l_idx+1]):
            for in_n in range(layers[l_idx]):
                weight_val = W[out_n, in_n]
                source = f"L{l_idx}_N{in_n}"
                target = f"L{l_idx+1}_N{out_n}"
                
                edges.append({
                    "source": source,
                    "target": target,
                    "weight": float(weight_val),
                    "is_active": is_active,
                    "is_forward": is_forward
                })
                
    data = {"nodes": nodes, "edges": edges}
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: transparent; }}
            .node-circle {{ transition: all 0.3s; }}
            @keyframes pulse-blue {{
                0% {{ filter: drop-shadow(0 0 5px rgba(33, 150, 243, 0.4)); }}
                50% {{ filter: drop-shadow(0 0 20px rgba(33, 150, 243, 0.9)); }}
                100% {{ filter: drop-shadow(0 0 5px rgba(33, 150, 243, 0.4)); }}
            }}
            @keyframes pulse-red {{
                0% {{ filter: drop-shadow(0 0 5px rgba(244, 67, 54, 0.4)); }}
                50% {{ filter: drop-shadow(0 0 20px rgba(244, 67, 54, 0.9)); }}
                100% {{ filter: drop-shadow(0 0 5px rgba(244, 67, 54, 0.4)); }}
            }}
            .glow-blue {{ animation: pulse-blue 1.5s infinite; }}
            .glow-red {{ animation: pulse-red 1.5s infinite; }}
            .label-text {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 16px; font-weight: bold; fill: #333; text-anchor: middle; }}
            .val-text {{ font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #444; text-anchor: middle; font-weight: 500; }}
            .weight-text {{ font-family: monospace; font-size: 13px; fill: #555; text-anchor: middle; font-weight: bold; paint-order: stroke; stroke: rgba(255,255,255,0.9); stroke-width: 4px; }}
        </style>
    </head>
    <body>
        <div id="graph-container"></div>
        <script>
            const data = {json.dumps(data)};
            const width = 800;
            const height = 500;
            
            const svg = d3.select("#graph-container")
                          .append("svg")
                          .attr("width", "100%")
                          .attr("height", "600px")
                          .attr("viewBox", `0 0 ${{width}} ${{height}}`)
                          .attr("preserveAspectRatio", "xMidYMid meet");
                          
            const defs = svg.append("defs");
            defs.append("marker")
                .attr("id", "arrow-fwd")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 32)
                .attr("refY", 0)
                .attr("markerWidth", 5)
                .attr("markerHeight", 5)
                .attr("orient", "auto")
                .append("path")
                .attr("fill", "#2196F3")
                .attr("d", "M0,-5L10,0L0,5");
                
            defs.append("marker")
                .attr("id", "arrow-bwd")
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", -22)
                .attr("refY", 0)
                .attr("markerWidth", 5)
                .attr("markerHeight", 5)
                .attr("orient", "auto-start-reverse")
                .append("path")
                .attr("fill", "#F44336")
                .attr("d", "M0,-5L10,0L0,5");
                
            const link = svg.selectAll(".link")
                .data(data.edges)
                .enter().append("g")
                .attr("class", "link");
                
            link.append("line")
                .attr("x1", d => data.nodes.find(n => n.id === d.source).x)
                .attr("y1", d => data.nodes.find(n => n.id === d.source).y)
                .attr("x2", d => data.nodes.find(n => n.id === d.target).x)
                .attr("y2", d => data.nodes.find(n => n.id === d.target).y)
                .attr("stroke", d => d.is_active ? (d.is_forward ? "#2196F3" : "#F44336") : "#CFD8DC")
                .attr("stroke-width", d => Math.max(1, Math.min(8, Math.abs(d.weight) * 3)))
                .attr("stroke-dasharray", d => d.is_active && !d.is_forward ? "6,4" : "none")
                .attr("marker-end", d => d.is_active && d.is_forward ? "url(#arrow-fwd)" : "none")
                .attr("marker-start", d => d.is_active && !d.is_forward ? "url(#arrow-bwd)" : "none")
                .style("opacity", d => d.is_active ? 0.9 : 0.4);
                
            const activeLinks = link.filter(d => d.is_active);
            activeLinks.append("circle")
                .attr("r", 5)
                .attr("fill", d => d.is_forward ? "#1976D2" : "#D32F2F")
                .style("filter", "drop-shadow(0 0 4px rgba(0,0,0,0.3))")
                .append("animateMotion")
                .attr("dur", "1.2s")
                .attr("repeatCount", "indefinite")
                .attr("path", d => {{
                    const src = data.nodes.find(n => n.id === d.source);
                    const tgt = data.nodes.find(n => n.id === d.target);
                    if (d.is_forward) return `M ${{src.x}} ${{src.y}} L ${{tgt.x}} ${{tgt.y}}`;
                    else return `M ${{tgt.x}} ${{tgt.y}} L ${{src.x}} ${{src.y}}`;
                }});

            link.append("text")
                .attr("class", "weight-text")
                .attr("x", d => (data.nodes.find(n => n.id === d.source).x * 0.65 + data.nodes.find(n => n.id === d.target).x * 0.35))
                .attr("y", d => (data.nodes.find(n => n.id === d.source).y * 0.65 + data.nodes.find(n => n.id === d.target).y * 0.35) - 6)
                .text(d => d.weight.toFixed(2));

            const node = svg.selectAll(".node")
                .data(data.nodes)
                .enter().append("g")
                .attr("class", "node")
                .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                
            node.append("circle")
                .attr("r", 26)
                .attr("fill", d => d.color)
                .attr("stroke", d => d.stroke)
                .attr("stroke-width", 3)
                .attr("class", d => d.glow_type === "blue" ? "node-circle glow-blue" : (d.glow_type === "red" ? "node-circle glow-red" : "node-circle"));
                
            node.append("text")
                .attr("class", "label-text")
                .attr("dy", "5")
                .html(d => d.label);
                
            const valBox = node.append("g")
                .attr("transform", "translate(0, 38)");
                
            valBox.append("rect")
                .attr("x", -45)
                .attr("y", -12)
                .attr("width", 90)
                .attr("height", 24)
                .attr("rx", 12)
                .attr("fill", "rgba(255, 255, 255, 0.9)")
                .attr("stroke", "#B0BEC5")
                .attr("stroke-width", 1.5)
                .style("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.1))");
                
            valBox.append("text")
                .attr("class", "val-text")
                .attr("dy", "4")
                .text(d => d.val_text);
                
        </script>
    </body>
    </html>
    """
    return html

st.title("NeuralInsight: The Transparent MLP Visualizer")
st.markdown("Developed by [@CursedCat7](https://github.com/CursedCat7) | [GitHub Repository](https://github.com/CursedCat7/NeuralInsight.git)")
st.markdown(f"**현재 아키텍처:** {layers_str} 구조")

tab1, tab2 = st.tabs(["Step-by-Step 구조 이해", "Continuous Training (연속 학습) 모드"])

with tab1:
    st.progress((st.session_state.step) / 5.0)
    
    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Previous Step", use_container_width=True) and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()
    with col_next:
        if st.button("Next Step", use_container_width=True) and st.session_state.step < 5:
            st.session_state.step += 1
            st.rerun()

    step = st.session_state.step
    cache = st.session_state.mlp.cache
    grads = cache.get('gradients', {})

    step_titles = [
        "초기 상태 (가중치 및 데이터 로드)",
        "순전파 연산 (Forward Pass)",
        "오차 및 출력층 $\delta$ 계산",
        "가중치 기울기 역전파 (Weight Gradients)",
        "은닉층 오차 전파 (Hidden Layer $\delta$)",
        "가중치 업데이트 (Weight Update)"
    ]
    st.markdown(f"### Step {step}: {step_titles[step]}")
    
    vis_col, math_col = st.columns([1.2, 1.0])
    with vis_col:
        html_str = draw_mlp_d3_html(st.session_state.mlp, layers, step)
        components.html(html_str, height=600, scrolling=False)
        
    with math_col:
        with st.expander("수식 기호 사전 (Math Symbol Legend)", expanded=False):
            st.markdown("""
- **$X$**: 입력 데이터 행렬 (Input Data Matrix)
- **$Y$**: 정답 데이터 행렬 (Target Label Matrix)
- **$N$**: 배치 사이즈 (Batch Size, 데이터 샘플의 개수)
- **$W^{(l)}$**: $l$번째 층의 가중치 행렬 (Weight Matrix of layer $l$)
- **$b^{(l)}$**: $l$번째 층의 편향 (Bias of layer $l$)
- **$Z^{(l)}$**: 선형 변환 결과 ($W^{(l)} A^{(l-1)} + b^{(l)}$)
- **$A^{(l)}$**: 활성화 함수를 통과한 출력값 ($\sigma(Z^{(l)})$)
- **$\delta^{(l)}$**: $l$번째 층의 오차(에러) 항목. 수학적으로는 $\\frac{\partial L}{\partial Z^{(l)}}$ (로컬 그래디언트)
- **$L$**: 손실(Loss) 함수 값
- **$\sigma$**: 활성화 함수 (Activation Function)
- **$\sigma'$**: 활성화 함수의 미분값 (Derivative of Activation)
- **$\odot$**: 행렬의 원소별 곱셈 (Hadamard Product)
- **$\eta$**: 학습률 (Learning Rate)
            """)

        with st.expander("선택된 함수의 미분 유도 과정 (Derivations)", expanded=False):
            st.markdown(f"**1. 손실 함수 ({loss_choice.upper()})의 미분**")
            if loss_choice == "mse":
                st.latex(r"L = \frac{1}{2N} \sum (A - Y)^2")
                st.latex(r"\frac{\partial L}{\partial A} = \frac{1}{N} (A - Y)")
            elif loss_choice == "cross_entropy":
                st.latex(r"L = -\frac{1}{N} \sum Y \log(A)")
                st.latex(r"\frac{\partial L}{\partial A} = -\frac{1}{N} \frac{Y}{A}")
            
            st.markdown("---")
            st.markdown(f"**2. 활성화 함수 ({act_choice.upper()})의 미분**")
            if act_choice == "sigmoid":
                st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}")
                st.latex(r"\frac{\partial \sigma}{\partial z} = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}")
                st.latex(r"= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z) \cdot (1 - \sigma(z))")
            elif act_choice == "relu":
                st.latex(r"\text{ReLU}(z) = \max(0, z)")
                st.latex(r"\frac{\partial \text{ReLU}}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}")
            elif act_choice == "tanh":
                st.latex(r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")
                st.latex(r"\frac{\partial \tanh}{\partial z} = \frac{(e^z + e^{-z})^2 - (e^z - e^{-z})^2}{(e^z + e^{-z})^2}")
                st.latex(r"= 1 - \left(\frac{e^z - e^{-z}}{e^z + e^{-z}}\right)^2 = 1 - \tanh^2(z)")
            elif act_choice == "softmax":
                st.latex(r"\sigma(z_i) = \frac{e^{z_i}}{\sum_{k} e^{z_k}}")
                st.latex(r"\frac{\partial \sigma(z_i)}{\partial z_j} = \begin{cases} \sigma(z_i)(1 - \sigma(z_j)) & \text{if } i = j \\ -\sigma(z_i)\sigma(z_j) & \text{if } i \neq j \end{cases}")
            
            if loss_choice == "cross_entropy" and act_choice == "softmax":
                st.markdown("---")
                st.markdown("**3. Softmax + Cross Entropy 복합 미분 (출력층 $\delta^{(L)}$)**")
                st.latex(r"\delta_i = \frac{\partial L}{\partial z_i} = \sum_j \frac{\partial L}{\partial a_j} \frac{\partial a_j}{\partial z_i}")
                st.latex(r"= -\frac{1}{N}\frac{y_i}{a_i} a_i (1-a_i) + \sum_{j \neq i} \left( -\frac{1}{N}\frac{y_j}{a_j} (-a_j a_i) \right)")
                st.latex(r"= \frac{1}{N}\left( -y_i + y_i a_i + \sum_{j \neq i} y_j a_i \right) = \frac{1}{N}\left( -y_i + a_i \sum_k y_k \right)")
                st.latex(r"\text{Since } \sum y_k = 1, \quad \frac{\partial L}{\partial Z} = \frac{1}{N}(A - Y)")

        if step == 0:
            st.info("**0. 전제 및 표기법 (Initialization)**\n\n- **$X$**: 입력 데이터 행렬(Input).\n- **$Y$**: 정답 데이터 행렬(Target).\n- 네트워크가 가변적이더라도 행렬 표기법을 통해 동일한 수식을 일반화하여 전개합니다.")
            st.latex(r"X = " + numpy_to_latex_bmatrix(X))
            st.latex(r"Y = " + numpy_to_latex_bmatrix(Y))
        elif step == 1:
            st.info("**1. 순전파 (Forward Pass)**\n\n- **선형 변환 ($Z$)**: 이전 층의 출력($A$)과 가중치($W$)를 행렬곱한 뒤 편향($b$)을 더합니다.\n- **활성화 ($A$)**: 활성화 함수($\sigma$)를 통과시켜 비선형성을 부여합니다.")
            for l_idx in range(len(layers)-1):
                st.markdown(f"**Layer {l_idx+1}**")
                W_str = numpy_to_latex_bmatrix(st.session_state.mlp.weights[l_idx])
                b_str = numpy_to_latex_bmatrix(st.session_state.mlp.biases[l_idx])
                A_prev = numpy_to_latex_bmatrix(cache['activations'][l_idx])
                Z_cur = numpy_to_latex_bmatrix(cache['zs'][l_idx])
                A_cur = numpy_to_latex_bmatrix(cache['activations'][l_idx+1])
                st.latex(rf"Z^{{({l_idx+1})}} = W^{{({l_idx+1})}} A^{{({l_idx})}} + b^{{({l_idx+1})}}")
                st.latex(rf"= {W_str} {A_prev} + {b_str}")
                st.latex(rf"= {Z_cur}")
                st.latex(rf"A^{{({l_idx+1})}} = \sigma(Z^{{({l_idx+1})}}) = {A_cur}")
        elif step == 2:
            st.info("**2. 오차 및 출력층 $\delta$ (Loss & Output Error)**\n\n- **Loss**: 예측값($A^{(L)}$)과 실제값($Y$)의 차이를 정량화합니다.\n- **출력층 $\delta^{(L)}$**: 손실 함수의 미분과 활성화 함수의 미분을 연쇄 법칙으로 곱하여, 출력 노드의 오차 기여도를 도출합니다.")
            st.latex(rf"L = \text{{{loss_choice}}}")
            st.markdown("**출력층 오차 ($\\delta^{(L)}$) 체인룰 유도:**")
            L_idx = len(layers)-1
            st.latex(rf"\delta^{{({L_idx})}} = \frac{{\partial L}}{{\partial Z^{{({L_idx})}}}} = \frac{{\partial L}}{{\partial A^{{({L_idx})}}}} \times \frac{{\partial A^{{({L_idx})}}}}{{\partial Z^{{({L_idx})}}}}")
            delta_L = numpy_to_latex_bmatrix(cache['deltas'][-1] if 'deltas' in cache else (cache['activations'][-1] - Y))
            st.latex(rf"\delta^{{({L_idx})}} = {delta_L}")
        elif step == 3:
            st.info("**3. 가중치 기울기 (Weight Gradients)**\n\n- 가중치를 얼마나 수정해야 할지 결정하는 단계입니다.\n- 체인룰을 통해 수식을 전개하면 현재 층의 오차($\delta^{(l)}$)와 이전 층의 출력값($A^{(l-1)}$)의 외적(Outer Product)이 됨을 알 수 있습니다.")
            for l_idx in reversed(range(len(layers)-1)):
                layer_num = l_idx + 1
                st.markdown(f"**Layer {layer_num} Gradients 유도:**")
                st.latex(rf"\frac{{\partial L}}{{\partial W^{{({layer_num})}}}} = \frac{{\partial L}}{{\partial Z^{{({layer_num})}}}} \times \frac{{\partial Z^{{({layer_num})}}}}{{\partial W^{{({layer_num})}}}}")
                st.latex(rf"\left( \text{{since }} Z^{{({layer_num})}} = W^{{({layer_num})}} A^{{({layer_num-1})}} + b^{{({layer_num})}} \text{{, we get }} \frac{{\partial Z}}{{\partial W}} = A \right)")
                st.latex(rf"\frac{{\partial L}}{{\partial W^{{({layer_num})}}}} = \delta^{{({layer_num})}} (A^{{({layer_num-1})}})^T")
                dW_str = numpy_to_latex_bmatrix(grads['dW'][l_idx])
                st.latex(rf"= {dW_str}")
        elif step == 4:
            st.info("**4. 은닉층 오차 전파 (Hidden Layer $\delta$)**\n\n- 상위 층에서 발생한 오차를 거꾸로 하위 층으로 흘려보냅니다.\n- 체인룰을 전개하면, 다음 층의 가중치($W^{(l+1)}$)를 통해 오차($\delta^{(l+1)}$)가 분배되고 현재 층의 활성화 미분값 $\sigma'(Z)$이 곱해지는 것을 볼 수 있습니다.")
            if len(layers) > 2:
                for l_idx in reversed(range(1, len(layers)-1)):
                    layer_num = l_idx
                    st.markdown(f"**Layer {layer_num} $\\delta$ 유도:**")
                    st.latex(rf"\delta^{{({layer_num})}} = \frac{{\partial L}}{{\partial Z^{{({layer_num})}}}} = \frac{{\partial L}}{{\partial Z^{{({layer_num+1})}}}} \times \frac{{\partial Z^{{({layer_num+1})}}}}{{\partial A^{{({layer_num})}}}} \times \frac{{\partial A^{{({layer_num})}}}}{{\partial Z^{{({layer_num})}}}}")
                    st.latex(rf"\delta^{{({layer_num})}} = (W^{{({layer_num+1})}})^T \delta^{{({layer_num+1})}} \odot \sigma'(Z^{{({layer_num})}})")
            else:
                st.warning("은닉층이 없는 구조이므로 4단계 연산이 생략됩니다.")
        elif step == 5:
            st.info("**5. 가중치 업데이트 (Weight Update)**\n\n- 경사하강법(Gradient Descent)을 적용하여 새로운 가중치로 갱신합니다.\n- 학습률(Learning Rate)을 곱한 기울기값을 기존 가중치에서 뺍니다.")
            st.latex(r"W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial L}{\partial W}")
            for l_idx in range(len(layers)-1):
                layer_num = l_idx + 1
                st.markdown(f"**Layer {layer_num} 가중치 업데이트 ($W^{{({layer_num})}}$):**")
                st.latex(rf"W^{{({layer_num})}}_{{\text{{new}}}} = W^{{({layer_num})}}_{{\text{{old}}}} - \eta \frac{{\partial L}}{{\partial W^{{({layer_num})}}}}")

with tab2:
    st.markdown("### Continuous Training (연속 학습) 대시보드")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        epochs = st.number_input("총 학습 에폭 (Epochs)", min_value=10, max_value=5000, value=500, step=10)
    with ctrl_col2:
        update_freq = st.number_input("시각화 갱신 주기 (에폭 단위)", min_value=1, max_value=500, value=20, step=1)
    with ctrl_col3:
        anim_speed = st.slider("애니메이션 속도 (초 단위 대기)", min_value=0.0, max_value=0.5, value=0.01, step=0.01)
        
    start_train = st.button("학습 시작 (Train)", use_container_width=True, type="primary")
    
    st.markdown("---")
    
    metrics_placeholder = st.empty()
    
    col_graph, col_loss = st.columns([1.2, 1])
    with col_graph:
        graph_placeholder = st.empty()
    with col_loss:
        loss_placeholder = st.empty()
        
    st.markdown("#### 가중치 및 기울기 분포 (Weights & Gradients)")
    st.info("""
    **📊 그래프 해석 가이드**
    - **$W$ Heatmap (히트맵)**: 해당 층의 가중치 행렬입니다. 붉은색은 양(+), 푸른색은 음(-)의 가중치를 의미하며 색이 진할수록 연결 강도가 셉니다.
    - **$\\nabla W$ Distribution (기울기 분포)**: 역전파된 오차로 인해 계산된 가중치 변화량($dW$)의 히스토그램입니다. x축의 값이 0에만 비정상적으로 몰려있다면 **'기울기 소실(Vanishing Gradient)'** 상태임을 나타냅니다.
    """)
    heatmaps_placeholder = st.empty()
    
    if start_train:
        temp_mlp = MLP(layers=layers, activation=act_choice, loss=loss_choice, optimizer=opt_choice)
        try:
            temp_mlp.set_params(W_list, b_list)
        except:
            pass
            
        losses = []
        prev_loss = None
        
        with st.spinner(f"{epochs} Epochs 학습 중..."):
            for epoch in range(1, epochs + 1):
                loss = temp_mlp.train(X_train, Y_train, epochs=1, learning_rate=lr)[0]
                losses.append(loss)
                
                if epoch % update_freq == 0 or epoch == epochs:
                    loss_delta = 0 if prev_loss is None else (loss - prev_loss)
                    
                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Current Epoch", f"{epoch} / {epochs}")
                        m2.metric("Loss ($L$)", f"{loss:.4f}", f"{loss_delta:.4f}", delta_color="inverse")
                        m3.metric("Learning Rate ($\eta$)", f"{lr}")
                    prev_loss = loss
                    
                    loss_df = pd.DataFrame(losses, columns=['Loss'])
                    loss_placeholder.line_chart(loss_df, height=350)
                    
                    # 1. Forward Pass Rendering
                    html_str_fwd = draw_mlp_d3_html(temp_mlp, layers, step=1)
                    with graph_placeholder:
                        components.html(html_str_fwd, height=400, scrolling=False)
                        
                    if anim_speed > 0:
                        time.sleep(anim_speed)
                        
                    # 2. Backward Pass Rendering
                    html_str_bwd = draw_mlp_d3_html(temp_mlp, layers, step=7)
                    with graph_placeholder:
                        components.html(html_str_bwd, height=400, scrolling=False)
                    
                    with heatmaps_placeholder.container():
                        h_cols = st.columns(len(layers)-1)
                        for l_idx in range(len(layers)-1):
                            with h_cols[l_idx]:
                                st.latex(rf"\text{{Layer }} {l_idx+1} \text{{ Heatmap }} (W^{{({l_idx+1})}})")
                                fig_w = px.imshow(temp_mlp.weights[l_idx], text_auto=True, aspect="auto", 
                                                  color_continuous_scale="RdBu_r", zmin=-2, zmax=2)
                                fig_w.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=200)
                                st.plotly_chart(fig_w, use_container_width=True, key=f"w_{l_idx}_{epoch}")
                                
                                if 'gradients' in temp_mlp.cache and 'dW' in temp_mlp.cache['gradients'] and len(temp_mlp.cache['gradients']['dW']) > l_idx:
                                    st.latex(rf"\text{{Layer }} {l_idx+1} \text{{ Gradients }} (\nabla W^{{({l_idx+1})}})")
                                    dW = temp_mlp.cache['gradients']['dW'][l_idx].flatten()
                                    fig_g = px.histogram(dW, nbins=20)
                                    fig_g.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=150, showlegend=False)
                                    st.plotly_chart(fig_g, use_container_width=True, key=f"g_{l_idx}_{epoch}")
                                    
                    if anim_speed > 0:
                        time.sleep(anim_speed)
        
        st.success("학습 완료!")