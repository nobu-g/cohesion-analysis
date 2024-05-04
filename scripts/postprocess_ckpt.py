import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="path to input knp dir")
    parser.add_argument("OUTPUT", type=str, help="path to output dir")
    args = parser.parse_args()

    input_file = Path(args.INPUT)
    output_file = Path(args.OUTPUT)

    ckpt = torch.load(input_file, map_location="cpu")
    ckpt["hyper_parameters"]["trainer"]["strategy"] = "auto"
    ckpt["hyper_parameters"]["work_dir"] = ""
    ckpt["hyper_parameters"]["data_dir"] = ""
    ckpt["hyper_parameters"]["save_dir"] = ""
    ckpt["hyper_parameters"]["name"] = ""
    ckpt["hyper_parameters"]["exp_dir"] = ""
    ckpt["hyper_parameters"]["run_id"] = ""
    ckpt["hyper_parameters"]["run_dir"] = ""
    print(ckpt["hyper_parameters"])
    torch.save(ckpt, output_file)


if __name__ == "__main__":
    main()
