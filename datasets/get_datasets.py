import os
import shutil
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="box_stacking_normal_task_2500.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
os.rename('box_stacking_normal_task_2500.pkl', 'unity_stacking.pkl')
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="unity_stacking_actions.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
if not os.path.exists("../action_data/unity_stacking/"): 
	os.makedirs("../action_data/unity_stacking/") 
shutil.move("unity_stacking_actions.pkl", "../action_data/unity_stacking/unity_stacking_actions.pkl")
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="evaluation_unity_stacking_graph_classes.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)