import streamlit as st
import numpy as np
import graphviz
from mlp_engine import MLP
import pandas as pd
import time
import plotly.express as px

st.set_page_config(page_title="NeuralInsight", layout="wide")

st.sidebar.header("Configuration")
layers_str = st.sidebar.text_input("네트워크 구조 (ex: 2,4,3,2)", value="2,2,1")
try:
    layers = [int(x.strip()) for x in layers_str.split(',')]
    if len(layers) < 2:
        st.sidebar.error("최소 2개의 레이어(입력, 출력)가 필요합니다.")
        st.stop()
except:
    st.sidebar.error("콤마로 구분된 정수를 입력하세요.")
    st.stop()

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
    
    st.sidebar.markdown(f"**W{i+1} (Layer {i} -> {i+1})**")
    edited_W = st.sidebar.data_editor(st.session_state[w_key], key=f"w{i+1}_editor")
    st.sidebar.markdown(f"**b{i+1} (Layer {i+1} Bias)**")
    edited_b = st.sidebar.data_editor(st.session_state[b_key], key=f"b{i+1}_editor")
    
    W_list.append(edited_W.values)
    b_list.append(edited_b.values)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Loader")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=['csv'])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        X_train = df_uploaded.iloc[:, :current_input_size].values.T
        Y_train = df_uploaded.iloc[:, current_input_size:current_input_size+current_output_size].values.T
        st.sidebar.success(f"{X_train.shape[1]}개 데이터 로드 완료! (연속 학습에 사용됨)")
        X = X_train[:, [0]]
        Y = Y_train[:, [0]]
    except Exception as e:
        st.sidebar.error("CSV 파싱 오류")
        X = np.ones((current_input_size, 1)) * 0.5
        Y = np.ones((current_output_size, 1)) * 0.7
        X_train, Y_train = X, Y
else:
    # Default dummy inputs
    X = np.ones((current_input_size, 1)) * 0.5
    Y = np.ones((current_output_size, 1)) * 0.7
    if layers_str == "2,2,1":
        X = np.array([[0.5], [0.1]])
        Y = np.array([[0.7]])
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

def draw_mlp_dynamic(mlp, layers, step):
    dot = graphviz.Digraph(comment='MLP Structure')
    dot.attr(rankdir='LR', size='10,6', bgcolor='transparent')
    dot.attr('node', shape='plaintext', fontname='Arial')
    
    color_in = '#E8F5E9'
    color_hid = '#E3F2FD'
    color_out = '#FFF3E0'
    color_active_fwd = '#FFF59D'
    color_active_bwd = '#FFCDD2'
    
    def make_html_node(title, z_val=None, a_val=None, x_val=None, bgcolor="white", bordercolor="black"):
        if x_val is not None:
            return f'''<
            <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" BGCOLOR="{bgcolor}" COLOR="{bordercolor}" STYLE="ROUNDED">
                <TR><TD BORDER="0"><B><FONT POINT-SIZE="16">{title}</FONT></B></TD></TR>
                <TR><TD BGCOLOR="#FFFFFF90"><FONT POINT-SIZE="13">입력 = {x_val:.2f}</FONT></TD></TR>
            </TABLE>>'''
        else:
            return f'''<
            <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8" BGCOLOR="{bgcolor}" COLOR="{bordercolor}" STYLE="ROUNDED">
                <TR><TD COLSPAN="2" BORDER="0"><B><FONT POINT-SIZE="16">{title}</FONT></B></TD></TR>
                <TR>
                    <TD BGCOLOR="#FFFFFF90"><FONT POINT-SIZE="13">z<BR/><B>{z_val:.2f}</B></FONT></TD>
                    <TD BGCOLOR="#FFFFFF90"><FONT POINT-SIZE="13">a<BR/><B>{a_val:.2f}</B></FONT></TD>
                </TR>
            </TABLE>>'''
    
    node_names = []
    # Create nodes
    for l_idx, num_nodes in enumerate(layers):
        layer_nodes = []
        is_input = (l_idx == 0)
        is_output = (l_idx == len(layers) - 1)
        
        if is_input:
            color = color_active_fwd if step == 1 else (color_active_bwd if step == 5 else color_in)
        elif is_output:
            color = color_active_fwd if step == 1 else (color_active_bwd if step in [2,3] else color_out)
        else:
            color = color_active_fwd if step == 1 else (color_active_bwd if step == 4 else color_hid)
            
        for n_idx in range(num_nodes):
            node_id = f"L{l_idx}_N{n_idx}"
            layer_nodes.append(node_id)
            if is_input:
                title = f"x{n_idx+1}"
                x_val = mlp.cache['activations'][0][n_idx, 0]
                dot.node(node_id, make_html_node(title, x_val=x_val, bgcolor=color, bordercolor='#4CAF50'))
            else:
                title = f"y_hat{n_idx+1}" if is_output else f"h{l_idx}_{n_idx+1}"
                z_val = mlp.cache['zs'][l_idx-1][n_idx, 0]
                a_val = mlp.cache['activations'][l_idx][n_idx, 0]
                border_color = '#FF9800' if is_output else '#2196F3'
                dot.node(node_id, make_html_node(title, z_val=z_val, a_val=a_val, bgcolor=color, bordercolor=border_color))
        node_names.append(layer_nodes)
        
    # Create edges
    for l_idx in range(len(layers) - 1):
        W = mlp.weights[l_idx]
        is_active = (step == 1) or (step == 5 - l_idx)
        edge_color = '#E53935' if (is_active and step > 1) else ('#2196F3' if is_active else '#B0BEC5')
        edge_penwidth = '3' if is_active else '1'
        
        for out_n in range(layers[l_idx+1]):
            for in_n in range(layers[l_idx]):
                weight_val = W[out_n, in_n]
                dot.edge(node_names[l_idx][in_n], node_names[l_idx+1][out_n], label=f"{weight_val:.2f}", color=edge_color, penwidth=edge_penwidth, fontcolor=edge_color)
                
    return dot

st.title("NeuralInsight: The Transparent MLP Visualizer")
st.markdown("👨‍💻 Developed by [@CursedCat7](https://github.com/CursedCat7) | 📦 [GitHub Repository](https://github.com/CursedCat7/NeuralInsight.git)")
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
        st.graphviz_chart(draw_mlp_dynamic(st.session_state.mlp, layers, step))
        
    with math_col:
        if step == 0:
            st.info("**0. 전제 (설정된 초기 상태)**")
            st.markdown(r"행렬 표기법을 통해 임의의 노드 개수에도 동일한 수식을 전개합니다.")
            st.latex(r"X = " + numpy_to_latex_bmatrix(X))
            st.latex(r"Y = " + numpy_to_latex_bmatrix(Y))
        elif step == 1:
            st.subheader("1. Forward 구조 (순전파)")
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
            st.subheader("2. Loss 함수 및 출력층 $\\delta$")
            st.latex(rf"L = \text{{{loss_choice}}}")
            st.markdown("**출력층 오차 ($\\delta^{(L)}$)**")
            L_idx = len(layers)-1
            delta_L = numpy_to_latex_bmatrix(cache['deltas'][-1] if 'deltas' in cache else (cache['activations'][-1] - Y))
            st.latex(rf"\delta^{{({L_idx})}} = \frac{{\partial L}}{{\partial Z^{{({L_idx})}}}} = {delta_L}")
        elif step == 3:
            st.subheader("3. 역전파 가중치 기울기 (Weight Gradients)")
            for l_idx in reversed(range(len(layers)-1)):
                layer_num = l_idx + 1
                st.markdown(f"**Layer {layer_num} Gradients**")
                dW_str = numpy_to_latex_bmatrix(grads['dW'][l_idx])
                st.latex(rf"\frac{{\partial L}}{{\partial W^{{({layer_num})}}}} = \delta^{{({layer_num})}} (A^{{({layer_num-1})}})^T")
                st.latex(rf"= {dW_str}")
        elif step == 4:
            st.subheader("4. 은닉층 $\\delta$ 전파")
            if len(layers) > 2:
                for l_idx in reversed(range(1, len(layers)-1)):
                    layer_num = l_idx
                    st.markdown(f"**Layer {layer_num} $\\delta$**")
                    st.latex(rf"\delta^{{({layer_num})}} = (W^{{({layer_num+1})}})^T \delta^{{({layer_num+1})}} \odot \sigma'(Z^{{({layer_num})}})")
            else:
                st.info("은닉층이 없어 생략됩니다.")
        elif step == 5:
            st.subheader("5. Weight Update (가중치 업데이트)")
            for l_idx in range(len(layers)-1):
                layer_num = l_idx + 1
                st.markdown(f"**Layer {layer_num} W Update**")
                st.latex(rf"W^{{({layer_num})}}_{{\text{{new}}}} = W^{{({layer_num})}}_{{\text{{old}}}} - \eta \frac{{\partial L}}{{\partial W^{{({layer_num})}}}}")

with tab2:
    st.markdown("### Continuous Training (연속 학습) & Weight Heatmap")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        epochs = st.number_input("학습 에폭 (Epochs)", min_value=10, max_value=2000, value=200, step=10)
        anim_speed = st.slider("애니메이션 속도 (초 단위 대기)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        
        if st.button("학습 시작 (Train)", use_container_width=True):
            temp_mlp = MLP(layers=layers, activation=act_choice, loss=loss_choice, optimizer=opt_choice)
            try:
                temp_mlp.set_params(W_list, b_list)
            except:
                pass
            
            loss_placeholder = st.empty()
            heatmap_placeholder = st.empty()
            
            losses = []
            
            with st.spinner(f"{epochs} Epochs 학습 중..."):
                for epoch in range(epochs):
                    loss = temp_mlp.train(X_train, Y_train, epochs=1, learning_rate=lr)[0]
                    losses.append(loss)
                    
                    if epoch % max(1, epochs//20) == 0 or epoch == epochs - 1:
                        # Update charts
                        loss_df = pd.DataFrame(losses, columns=['Loss'])
                        loss_placeholder.line_chart(loss_df)
                        
                        # Plotly Heatmap for W1
                        fig = px.imshow(temp_mlp.weights[0], text_auto=True, aspect="auto", title="W1 Matrix Heatmap", color_continuous_scale="RdBu_r", zmin=-2, zmax=2)
                        heatmap_placeholder.plotly_chart(fig, use_container_width=True, key=f"heat_{epoch}")
                        
                        if anim_speed > 0:
                            time.sleep(anim_speed)
                
            st.success("학습 완료!")