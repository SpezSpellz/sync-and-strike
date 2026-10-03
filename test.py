import io
import os
import numpy as np
import onnx
import onnxruntime as ort
import shap
import matplotlib.pyplot as plt
MODEL_PATH = "Assets/Models/Basic.onnx"
sess = ort.InferenceSession(MODEL_PATH)
action_names = [x.name for x in sess.get_outputs()]
continuous_actions_index = action_names.index("deterministic_continuous_actions")
discrete_actions_index = action_names.index("deterministic_discrete_actions")

background_data = np.loadtxt('sample.csv', delimiter=',')

results = sess.run(None, {
    "obs_0": background_data.astype(np.float32),
    "action_masks": np.ones((len(background_data), 10), dtype=np.float32)
})

print(results)

moves = ["Block", "Dash", "Horizontal Slash", "Idle", "Jump", "Super Jump", "Vertical Slash", "Walk Forward"]

with open("predicted.csv", "w", encoding="UTF-8") as f:
    f.write("pos_x,pos_y,vel_x,vel_y,e_pos_x,e_pos_y,e_vel_x,e_vel_y,flipped,move,kb_power,kb_direction,jump_power,jump_direction\n")
    for i in range(len(background_data)):
        discrete = results[discrete_actions_index][i]
        continuous = results[continuous_actions_index][i]
        f.write(f"{background_data[i][0]},{background_data[i][1]},{background_data[i][2]},{background_data[i][3]},{background_data[i][4]},{background_data[i][5]},{background_data[i][6]},{background_data[i][7]},{discrete[0]},{moves[int(discrete[1])]},{(float(continuous[0]) + 1.0) / 2.0},{(float(continuous[1]) + 1.0) * np.pi},{(float(continuous[2]) + 3.0) / 2.0},{(float(continuous[3]) + 1.0) * (np.pi / 2.0)}\n")