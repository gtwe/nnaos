import glob
import os
import yaml
import sys

sys.path.append("../../")

import aos.dataframe as sdataframe


def main():
    print("Stage 4: Compiling results into dataframe")
    results = []

    results_dir = "./results/"
    yaml_files = sorted(glob.glob(os.path.join(results_dir, "*.yaml")))

    for yaml_file in yaml_files:
        with open(yaml_file, "r") as file:
            results.append(yaml.load(file, Loader=yaml.UnsafeLoader))

    df = sdataframe.create_frame(results)

    df.to_pickle("./results/df.pkl")
    print("Dataframe saved to ./results/df.pkl")


if __name__ == "__main__":
    main()
