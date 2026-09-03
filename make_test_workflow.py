"""Convert pony_sdxl_gguf_workflow.json to a runnable API-format prompt,
swapping the GGUF loader for the stock UNETLoader (native SDXL).
"""
import json, sys

def graph_to_prompt(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    prompt = {}
    for n in graph["nodes"]:
        pid = n["id"]
        inputs = {}
        for inp in n.get("inputs", []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            # find link
            for link in graph["links"]:
                if link[0] == link_id:
                    from_id, from_slot = link[1], link[2]
                    inputs[inp["name"]] = [str(from_id), from_slot]
                    break
        wv = n.get("widgets_values")
        if wv is not None:
            if n["type"] == "UNETLoader":
                # widgets: [unet_name, weight_dtype]
                inputs.setdefault("unet_name", wv[0])
                if len(wv) > 1:
                    inputs.setdefault("weight_dtype", wv[1])
            elif n["type"] == "DualCLIPLoader":
                inputs.setdefault("clip_name1", wv[0])
                inputs.setdefault("clip_name2", wv[1])
                inputs.setdefault("type", wv[2])
            elif n["type"] == "CLIPTextEncode":
                inputs.setdefault("text", wv[0])
            elif n["type"] == "EmptyLatentImage":
                inputs.setdefault("width", wv[0])
                inputs.setdefault("height", wv[1])
                inputs.setdefault("batch_size", wv[2])
            elif n["type"] == "KSampler":
                inputs.setdefault("seed", wv[0])
                inputs.setdefault("steps", wv[2])
                inputs.setdefault("cfg", wv[3])
                inputs.setdefault("sampler_name", wv[4])
                inputs.setdefault("scheduler", wv[5])
                inputs.setdefault("denoise", wv[6])
            elif n["type"] == "VAELoader":
                inputs.setdefault("vae_name", wv[0])
            elif n["type"] == "SaveImage":
                inputs.setdefault("filename_prefix", wv[0])
            elif n["type"] == "LoadImage":
                inputs.setdefault("image", wv[0])
            elif n["type"] == "IPAdapterModelLoader":
                inputs.setdefault("ipadapter_file", wv[0])
            elif n["type"] == "CLIPVisionLoader":
                inputs.setdefault("clip_name", wv[0])
            elif n["type"] == "IPAdapterAdvanced":
                inputs.setdefault("weight", wv[0])
                inputs.setdefault("weight_type", wv[1])
                inputs.setdefault("combine_embeds", wv[2])
                inputs.setdefault("start_at", wv[3])
                inputs.setdefault("end_at", wv[4])
                inputs.setdefault("embeds_scaling", wv[5])
        # handle special: KSampler seed control_after_generate
        prompt[str(pid)] = {"class_type": n["type"], "inputs": inputs}
    # add explicit UNETLoader node to replace UnetLoaderGGUF
    return prompt

def main():
    graph = json.load(open("pony_sdxl_gguf_workflow.json"))
    prompt = graph_to_prompt(graph)

    # Replace the UnetLoaderGGUF node (id 1) with stock UNETLoader
    # Node id 1 outputs MODEL into link 1 -> IPAdapterAdvanced (id 13) input 0
    # We keep node structure but change class_type and inputs.
    unet_node_id = "1"
    prompt[unet_node_id] = {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "autismmix_pony_sdxl_unet.safetensors",
            "weight_dtype": "default",
        },
    }
    # Swap the GGUF clip loader for the stock DualCLIPLoader (native safetensors)
    clip_node_id = "2"
    prompt[clip_node_id] = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "pony_clip_g.safetensors",
            "clip_name2": "pony_clip_l.safetensors",
            "type": "sdxl",
        },
    }
    out = json.dumps(prompt, indent=2)
    open("test_prompt_api.json", "w").write(out)
    print("Wrote test_prompt_api.json")
    for k, v in prompt.items():
        print(k, v["class_type"])

if __name__ == "__main__":
    main()