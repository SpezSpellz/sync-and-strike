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
print(action_names)
continuous_actions_index = action_names.index("deterministic_continuous_actions")
discrete_actions_index = action_names.index("deterministic_discrete_actions")
raw_continuous_action_index = action_names.index("/_continuous_distribution/mu/Gemm_output_0")
raw_move_action_index = action_names.index("/_discrete_distribution/Softmax_1_output_0")
raw_flip_action_index = action_names.index("/_discrete_distribution/Softmax_output_0")
input_name = "obs_0"

# --- SHAP Setup ---
# Mock background data for SHAP (In a real scenario, use a representative dataset)
background_data = np.loadtxt('sample.csv', delimiter=',')

def model_predict(x_batch, target_attr, target_idx=0):
    # Ensure x_batch is a 2D numpy array (Batch Size, Features)
    x_batch = np.atleast_2d(x_batch).astype(np.float32)

    # Create the action mask for the ENTIRE batch at once
    # If the batch has 100 samples, we need a (100, 10) mask
    batch_size = x_batch.shape[0]
    masks = np.ones((batch_size, 10), dtype=np.float32)

    # Run inference on the whole batch - NO LOOP
    # We index [4] because you mentioned it's the 5th output
    results = sess.run(None, {
        "obs_0": x_batch,
        "action_masks": masks
    })
    if target_attr == "isFlipped": return results[raw_flip_action_index][:, target_idx].astype(np.float64)
    if target_attr == "move": return results[raw_move_action_index][:, target_idx].astype(np.float64)
    if target_attr == "kb_power": return results[raw_continuous_action_index][:, 0].astype(np.float64)
    if target_attr == "kb_direction": return results[raw_continuous_action_index][:, 1].astype(np.float64)
    if target_attr == "jump_power": return results[raw_continuous_action_index][:, 2].astype(np.float64)
    if target_attr == "jump_direction": return results[raw_continuous_action_index][:, 3].astype(np.float64)
    
    return np.array(results[raw_move_action_index])

class InferenceRequest(BaseModel):
    data: list # [pos_x, pos_y, vel_x, vel_y, e_pos_x, e_pos_y, e_vel_x, e_vel_y]

class ExplainRequest(BaseModel):
    data: list[float]
    target: str
    index: int

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
        "isFlippedIndex": int(discrete[0]),
        "move": moves[int(discrete[1])] if int(discrete[1]) < len(moves) else "Unknown",
        "move_index": int(discrete[1]),
        "kb_power": (float(continuous[0]) + 1.0) / 2.0,
        "kb_direction": (float(continuous[1]) + 1.0) * np.pi,
        "jump_power": (float(continuous[2]) + 3.0) / 2.0,
        "jump_direction": (float(continuous[3]) + 1.0) * (np.pi / 2.0)
    }
    return res

@app.post("/explain")
async def explain(req: ExplainRequest):
    input_tensor = np.array(req.data, dtype=np.float32).reshape(1, 8) / 10.0
    explainer = shap.KernelExplainer(lambda x: model_predict(x, req.target, req.index), background_data)
    shap_v = explainer.shap_values(input_tensor)
    expl_obj = shap.Explanation(
        values=shap_v[0], # The SHAP values for one class
        base_values=explainer.expected_value,
        data=input_tensor[0],
        feature_names=["Self X", "Self Y", "Self Velocity X", "Self Velocity Y", "Enemy X", "Enemy Y", "Enemy Velocity X", "Enemy Velocity Y"]
    )
    
    # Generate SHAP waterfall plot
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(expl_obj, show=False)
    plt.title(f"SHAP: {req.target}")
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='svg', bbox_inches='tight')
    plt.close()
    return Response(content=img_buf.getvalue(), media_type="image/svg+xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
