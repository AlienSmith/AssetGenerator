import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nodes, folder_paths
cn = nodes.ControlNetLoader().load_controlnet("controlnet-union-sdxl.safetensors")[0]
print("ControlNet loaded:", type(cn).__name__)
st = __import__("comfy_extras.nodes_controlnet", fromlist=["SetUnionControlNetType"])
cn2 = st.SetUnionControlNetType().execute(cn, "canny/lineart/anime_lineart/mlsd")
print("union type set ok")
