import pandas as pd
from os import listdir
from math import ceil
from pathlib import Path
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_frame_path",
        type=str,
        default="./data_frames",
        help="path to csv files that will be loaded as pandas dataframes"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./bar_graphs/control",
        help="path to store the bar graphs"
    )

    args = parser.parse_args()

    dir_names = listdir(args.data_frame_path)
    dir_names.sort()

    # trainer bar graphs
    trainer_metrics = {
        0: "backward",
        1: "batch",
        2: "forward_pass",
        3: "dense gradients",
        4: "sparse gradients"
    }

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    for index, metric in trainer_metrics.items():
        bar_map = {}
        max_val = 0
        index_labels = []
        for dir_name in dir_names:
            
            dir_path = args.data_frame_path + f"/{dir_name}"
            for file_name in listdir(dir_path):
                if "trainer" not in file_name:
                    continue
                bar_file_name = file_name.split(".")[0]
                if bar_file_name not in bar_map:
                    bar_map[bar_file_name] = []
                data_frame_path = f"{dir_path}/{file_name}"
                df = pd.read_csv(f"{dir_path}/{file_name}")
                bar_map[bar_file_name].append(df.iloc[index]["mean"])
                max_val = max(max_val, df.iloc[index]["mean"])
        y_limit = max_val * .2 + max_val
        ax = pd.DataFrame(bar_map, index=dir_names).plot.bar(figsize=(15,10),
            ylim=(0, y_limit), title=metric
        )
        ax.legend(loc='center left',bbox_to_anchor=(1.0, 1.0))
        ax.set_ylabel("Delay")
        ax.set_xlabel("Number of trainers and servers")
        ax.figure.savefig(f'{args.output_path}/{metric}.png', bbox_inches='tight')

    # TODO: add server bar graphs
