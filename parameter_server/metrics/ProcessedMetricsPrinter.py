import statistics

import pandas as pd
from tabulate import tabulate
from pathlib import Path


class ProcessedMetricsPrinter:

    def save_to_file(self, data_frame, file_name, nt, ns, cuda):
        if cuda:
            file_name += "_cuda"
        else:
            file_name += "_cpu"
        path = f"./data_frames/{nt}_{ns}"
        Path(path).mkdir(parents=True, exist_ok=True)
        file_path = f"{path}/{file_name}.csv"
        data_frame.to_csv(file_path, encoding='utf-8', index=False)

    def print_data_frame(self, name, nt, ns, nct, ncs, processed_metrics):
        print(f"metrics for {name}")
        data_frame = self.get_data_frame(processed_metrics)
        # only supports one type of rpc
        self.save_to_file(data_frame, name, max(nt, nct), max(ns, ncs), nct > 0)
        print(tabulate(data_frame, showindex=False, headers=data_frame.columns, tablefmt="grid"))

    def combine_processed_metrics(self, processed_metrics_list):
        r"""
        A method that merges the value arrays of the keys in the dictionary
        of processed metrics.

        Args:
            processed_metrics_list (list): a list containing dictionaries with
                recorded metrics as keys, and the values are lists of elapsed times.

        Returns::
            A merged dictionary that is created from the list of dictionaries passed
                into the method.

        Examples::
            >>> instance = ProcessedMetricsPrinter()
            >>> dict_1 = trainer1.get_processed_metrics()
            >>> dict_2 = trainer2.get_processed_metrics()
            >>> print(dict_1)
            {
                "forward_metric_type,forward_pass" : [.0429, .0888]
            }
            >>> print(dict_2)
            {
                "forward_metric_type,forward_pass" : [.0111, .0222]
            }
            >>> processed_metrics_list = [dict_1, dict_2]
            >>> result = instance.combine_processed_metrics(processed_metrics_list)
            >>> print(result)
            {
                "forward_metric_type,forward_pass" : [.0429, .0888, .0111, .0222]
            }
        """
        processed_metric_totals = {}
        for processed_metrics in processed_metrics_list:
            for metric_name, values in processed_metrics.items():
                if metric_name not in processed_metric_totals:
                    processed_metric_totals[metric_name] = []
                processed_metric_totals[metric_name] += values
        return processed_metric_totals

    def get_data_frame(self, processed_metrics):
        df = pd.DataFrame(
            columns=['name', 'min', 'max', 'mean', 'variance', 'stdev']
        )
        for metric_name in sorted(processed_metrics.keys()):
            values = processed_metrics[metric_name]
            row = {
                "name": metric_name,
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "variance": statistics.variance(values),
                "stdev": statistics.stdev(values)
            }
            df = df.append(row, ignore_index=True)
        return df

    def print_metrics(self, name, nt, ns, nct, ncs, rank_metrics_list):
        if rank_metrics_list:
            metrics_list = []
            for rank, metric in rank_metrics_list:
                metrics_list.append(metric)
            combined_metrics = self.combine_processed_metrics(metrics_list)
            self.print_data_frame(f"{name}", nt, ns, nct, ncs, combined_metrics)
