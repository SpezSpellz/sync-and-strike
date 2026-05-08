import io
import os
import numpy as np
import onnx
import onnxruntime as ort
import shap
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

app = FastAPI()

# --- Model Loading and Surgery ---
MODEL_PATH = "Assets/Models/Basic.onnx"

def get_gemm_leading_to_continuous(model):
    """
    Traces backward from the 'continuous' output to find the first Gemm node.
    """
    # 1. Map every output tensor to the node that produces it
    tensor_to_node = {node.output[0]: node for node in model.graph.node}
    
    # 2. Find the target output tensor name
    target_output_name = None
    for out in model.graph.output:
        if "continuous" in out.name.lower():
            target_output_name = out.name
            break
    
    if not target_output_name:
        return None

    # 3. Trace back
    current_tensor = target_output_name
    while current_tensor in tensor_to_node:
        node = tensor_to_node[current_tensor]
        if node.op_type == "Gemm":
            return node.output[0] # Return the name of the Gemm output tensor
        
        # Move upstream to the first input of the current node
        # (Assuming the main path is the first input, common in action branches)
        current_tensor = node.input[0]
        
    return None

def modify_onnx_outputs(model_path):
    model = onnx.load(model_path)
    intermediate_outputs = []
    
    # Trace the specific Gemm node for continuous actions
    continuous_gemm_output = get_gemm_leading_to_continuous(model)
    if continuous_gemm_output:
        intermediate_outputs.append(continuous_gemm_output)
    
    # Find Softmax nodes for discrete actions
    for node in model.graph.node:
        if node.op_type == "Softmax":
            # You could apply similar back-tracing here if you only wanted
            # Softmaxes leading to "discrete_actions"
            intermediate_outputs.append(node.output[0])
            
    # Add identified tensors as new graph outputs
    for output_name in intermediate_outputs:
        # Check if it's already an output to avoid duplicates
        if any(out.name == output_name for out in model.graph.output):
            continue
            
        intermediate_node = onnx.helper.make_tensor_value_info(
            output_name, onnx.TensorProto.FLOAT, [None, None]
        )
        model.graph.output.append(intermediate_node)
    
    return model.SerializeToString()
# Initialize Session
modified_model_bytes = modify_onnx_outputs(MODEL_PATH)
sess = ort.InferenceSession(modified_model_bytes)
action_names = [x.name for x in sess.get_outputs()]
continuous_actions_index = action_names.index("deterministic_continuous_actions")
discrete_actions_index = action_names.index("deterministic_discrete_actions")
input_name = "obs_0"

# --- SHAP Setup ---
# Mock background data for SHAP (In a real scenario, use a representative dataset)
background_data = np.random.randn(100, 8).astype(np.float32)

def model_predict(data):
    # Wrapper for SHAP to interpret the model
    # We target the specific output index for the desired attribute
    return sess.run(None, {input_name: data.astype(np.float32), "action_masks": np.ones((1, 10), dtype=np.float32)})[0]

explainer = shap.Explainer(model_predict, background_data)

class InferenceRequest(BaseModel):
    data: list # [pos_x, pos_y, vel_x, vel_y, e_pos_x, e_pos_y, e_vel_x, e_vel_y]

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(req: InferenceRequest):
    # Pre-processing: Divide by 10 as per playerController logic
    input_tensor = np.array(req.data, dtype=np.float32).reshape(1, 8) / 10.0
    outputs = sess.run(None, {input_name: input_tensor, "action_masks": np.ones((1, 10), dtype=np.float32)})
    
    # Interpretation logic provided
    # outputs[0] = DiscreteActions, outputs[1] = ContinuousActions
    discrete = outputs[discrete_actions_index][0]
    continuous = outputs[continuous_actions_index][0]
    
    moves = ["Block", "Dash", "Horizontal Slash", "Idle", "Jump", "Super Jump", "Vertical Slash", "Walk Forward"]
    
    res = {
        "isFlipped": bool(discrete[0] == 0),
        "move": moves[int(discrete[1])] if int(discrete[1]) < len(moves) else "Unknown",
        "kb_power": (float(continuous[0]) + 1.0) / 2.0,
        "kb_direction": (float(continuous[1]) + 1.0) * np.pi,
        "jump_power": (float(continuous[2]) + 3.0) / 2.0,
        "jump_direction": (float(continuous[3]) + 1.0) * (np.pi / 2.0)
    }
    return res

@app.post("/explain")
async def explain(req: InferenceRequest):
    input_tensor = np.array(req.data, dtype=np.float32).reshape(1, 8) / 10.0
    shap_values = explainer(input_tensor)
    
    # Generate SHAP waterfall plot
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='svg', bbox_inches='tight')
    plt.close()
    return Response(content=img_buf.getvalue(), media_type="image/svg+xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)