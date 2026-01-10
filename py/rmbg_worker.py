import argparse
import os
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--node-root", required=True, help="Path to the comfyui-rmbg node root")
    p.add_argument("--model", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--process-res", type=int, required=True)
    p.add_argument("--sensitivity", type=float, required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--outputs", nargs="+", required=True)
    return p.parse_args()


def main():
    args = _parse_args()
    node_root = Path(args.node_root).resolve()
    node_py = node_root / "py"
    sys.path.insert(0, str(node_root))
    sys.path.insert(0, str(node_py))

    from PIL import Image
    import numpy as np
    import torch

    from AILab_RMBG import RMBGModel

    if not os.path.isdir(args.cache_dir):
        raise RuntimeError(f"Cache dir does not exist: {args.cache_dir}")

    model = RMBGModel()
    model.load_model(args.model)

    images = []
    for path in args.inputs:
        img = Image.open(path).convert("RGB")
        t = torch.from_numpy(np.array(img).astype("float32") / 255.0)
        images.append(t)

    params = {"process_res": args.process_res, "sensitivity": args.sensitivity}
    masks = model.process_image(images, args.model, params)

    if len(masks) != len(args.outputs):
        raise RuntimeError(f"Expected {len(args.outputs)} masks, got {len(masks)}")

    for mask, out_path in zip(masks, args.outputs):
        if isinstance(mask, Image.Image):
            mask_img = mask.convert("L")
        else:
            mask_img = mask
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(out_path)


if __name__ == "__main__":
    main()

