import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from multiprocessing import cpu_count
from itertools import product

from src.DRFL import DRFL, ParallelSearchDRFL, DRGS
from src.structures import HierarchyRoutine

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", type=str, default="data/activities-simulation.csv", help="Path to the data file")
argparser.add_argument("--dictionary_dir", type=str, default="data/dictionary_rooms.json",
                       help="Path to the dictionary file")
argparser.add_argument("--param_m", type=int, default=4, help="length of the subsequences")
argparser.add_argument("--param_R", type=int, default=10, help="least maximum distance between subsequences")
argparser.add_argument("--param_C", type=int, default=4, help="minimum number of matches of a routine")
argparser.add_argument("--param_G", type=int, default=60, help="minimum magnitude of a subsequence")
argparser.add_argument("--epsilon", type=float, default=0.5, help="minimum overlap percentage")
argparser.add_argument("--L", type=int, default=0, help="minimum number of subsequences in a routine")
argparser.add_argument("--fusion_distance", type=float, default=0.001,
                       help="minimum distance between clusters centroids to be fused")


def process_sequence(sequence: list):
    return np.where(np.array(sequence) == '', np.nan, np.array(sequence, dtype=int))


def load_data(data_dir: str):
    data = pd.read_csv(data_dir, skiprows=1, header=None)
    data.columns = ["Year", "Month", "Day"] + [f"Min_{i}" for i in range(data.shape[1] - 3)]
    data.set_index(["Year", "Month", "Day"], inplace=True)
    for col in data.columns:
        data[col] = process_sequence(data[col].tolist())
    return data


def obtain_correspondencies(json_dictionary_file: str):
    with open(json_dictionary_file, "r") as file:
        correspondences = json.load(file)
    return {int(v): k for k, v in correspondences.items()}


def group_by_hour(data: pd.DataFrame, correspondences: dict) -> pd.DataFrame:
    # List of all potential column names based on correspondences
    hour_cols = ["Year", "Month", "Day", "Hour"] + [f"N_{value}" for value in set(correspondences.values())]

    results = []
    for (year, month, day), df in data.groupby(level=[0, 1, 2]):
        df = df.melt(var_name='Minute', value_name='Room')
        df['Hour'] = df['Minute'].str.extract('(\d+)').astype(int) // 60
        df['Room'] = df['Room'].map(correspondences)

        # Count occurrences by hour
        hourly_data = df.pivot_table(index=["Hour"], columns="Room", aggfunc='size', fill_value=0)
        hourly_data = hourly_data.rename(columns=lambda x: f"N_{x}")
        hourly_data.reset_index(inplace=True)
        hourly_data['Year'] = year
        hourly_data['Month'] = month
        hourly_data['Day'] = day

        # Ensure all columns are present
        for col in hour_cols:
            if col not in hourly_data:
                hourly_data[col] = 0

        results.append(hourly_data[hour_cols])

    return pd.concat(results).reset_index(drop=True)


def get_time_series(path_to_feat_extraction: str, room: str) -> pd.Series:
    feat_extraction = pd.read_csv(path_to_feat_extraction)
    feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])
    feat_extraction.set_index("Date", inplace=True)
    room_time_series = feat_extraction[f"N_{room}"]
    return room_time_series


if __name__ == "__main__":
    # time_series = pd.Series([60, 60, 60, 60, 60, 60, 2, 2, 60, 60, 60, 60, 60, 60, 2, 2, 60, 60, 60, 60, 60, 60])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # json_dictionary_file = "data/dictionary_rooms.json"
    # data_dir = "data/activities-simulation-easy.csv"
    # correspondencies = obtain_correspondencies(json_dictionary_file)
    # df = load_data(data_dir)
    # feat_extraction = group_by_hour(df, correspondencies)
    # pd.set_option('display.max_columns', None, 'display.max_rows', None)
    # feat_extraction.to_csv("data/out_feat_extraction.csv", index=False)

    # json_dictionary_file = "data/dictionary_rooms.json"
    # data_dir = "data/activities-simulation-easy.csv"
    # correspondencies = obtain_correspondencies(json_dictionary_file)
    # df = load_data(data_dir)
    # feat_extraction = group_by_hour(df, correspondencies)
    # pd.set_option('display.max_columns', None, 'display.max_rows', None)
    # feat_extraction.to_csv("data/out_feat_extraction.csv", index=False)

    # room_time_series = get_time_series("data/out_feat_extraction.csv", "room")
    #
    # drgs = DRGS(length_range=(3, 100), R=5, C=300, G=35, epsilon=1, L=2, fusion_distance=0.001)
    # drgs.fit(room_time_series)
    # os.makedirs("results", exist_ok=True)
    # drgs.plot_separate_hierarchical_results(title_fontsize=35, labels_fontsize=30, yticks_fontsize=20,
    #                                         linewidth_bars=3, vline_width=3, xlim=(0, 300), figsize=(50, 25),
    #                                         show_xticks=False, save_dir="results")
    # routines = drgs.get_results()
    # routines.to_json("results/detected_routines.json")
    routines = HierarchyRoutine()
    routines.from_json("results/detected_routines.json")
    tree = routines.convert_to_cluster_tree()
    tree.plot_tree(title="Final node evolution")


    # tree.plot_tree(figsize=(45, 25), title="Dropping extra nodes")
    # print(dropped)
    # for node_drop in dropped:
    #     print(tree.get_node(node_drop))
    # Simple fit
    # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
    # target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]
    #
    # R = 1
    # C = 3
    # G = 5
    # epsilon = 1
    #
    # drfl = DRFL(3, R,  C,  G,  epsilon)
    # drfl.fit(time_series)
    #
    # detected_routines = drfl.get_results()
    #
    # drfl.show_results()
    # drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
    #                        yticks_fontsize=20, figsize=(45, 25),
    #                        linewidth_bars=2, xlim=(0, 50))
    #
    # DRGS = DRGS((3, 8), 2, 3, 5, 1)
    # DRGS.fit(time_series)
    # DRGS.show_results()
    # DRGS.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
    #                         yticks_fontsize=20, figsize=(45, 25),
    #                         linewidth_bars=2, xlim=(0, 50))

    # ts = [1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1]
    # for x in range(4):
    #     ts += ts
    #
    # print(ts)
    #
    # time_series = pd.Series(np.array(ts))
    # # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))
    # target_centroids = [[4 / 3, 3, 6], [3, 6, 4], [6, 4, 4 / 3]]
    #
    # _R = [x for x in range(1, 6, 1)]
    # _C = [x for x in range(1, 10, 1)]
    # _G = [x for x in range(1, 6, 1)]
    # _epsilon = [0.5, 1]
    # alpha = 0.5
    # sigma = 3
    #
    # parallel_search = ParallelSearchDRFL(n_jobs=10, alpha=alpha, sigma=sigma, param_grid={'_m': 3, '_R': _R, '_C': _C, '_G': _G, '_epsilon': _epsilon})
    # parallel_search.fit(time_series, target_centroids=target_centroids)
    # results = parallel_search.cv_results()
    # best_params = parallel_search.best_params()
    # print(results.head())
    # print(best_params)
    # best_drfl = DRFL(3, best_params["_R"], best_params["_C"], best_params["_G"], best_params["_epsilon"])
    # best_drfl.fit(time_series)
    #
    # detected_routines = best_drfl.get_results()
    #
    # best_drfl.show_results()
    # best_drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
    #                        yticks_fontsize=20, figsize=(45, 25),
    #                        linewidth_bars=2, xlim=(0,50))

    # PARALLEL SEARCH WITH COMPLICATED TIME SERIES

    # args = argparser.parse_args()
    # df = load_data(args.data_dir)
    # correspondencies = obtain_correspondencies(args.dictionary_dir)
    # feat_extraction = feature_extraction(df, correspondencies)
    # time_series = get_time_series(feat_extraction, "gym")
    #
    # R_params = [x for x in range(20, 70, 10)]
    # C_params = [x for x in range(5, 9)]
    # G_params = [x for x in range(20, 100, 10)]
    # # L_params = [x for x in range(20, 80, 10)]
    #
    # target_centroids = [[28, 0, 44, 0, 125, 11, 0], [79, 0, 47, 0, 66, 118, 0]]
    # params = list(product(R_params, C_params, G_params, [1], [0]))
    # alpha, sigma = 0.6, 4
    #
    # # Sequential search
    # st = time.time()
    # result = []
    # for R, C, G, epsilon, L in params:
    #     drfl = DRFL(7, R, C, G, epsilon, L)
    #     drfl.fit(time_series)
    #     mean_distance = drfl.estimate_distance(target_centroids, alpha, sigma)
    #     result.append({"R": R, "C": C, "G": G, "epsilon": epsilon, "L": L, "mean_distance": mean_distance})
    #
    # print(f"Elapsed sequential time: {time.time() - st}")
    #
    # param_grid = {'m': 7, 'R': R_params, 'C': C_params, 'G': G_params, 'epsilon': [1], 'L': [0]}
    #
    # # Parallel search: comparing time
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=1, alpha=alpha, sigma=sigma, param_grid=param_grid)
    # mDRFL.search_best(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers=1: {time.time() - st}")
    #
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=5, alpha=alpha, sigma=sigma, param_grid=param_grid)
    # mDRFL.search_best(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers=5: {time.time() - st}")
    #
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=10, alpha=alpha, sigma=sigma, param_grid=param_grid)
    # mDRFL.search_best(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers=10: {time.time() - st}")
    #
    # st = time.time()
    # mDRFL = ParallelSearchDRFL(n_jobs=cpu_count() - 2, alpha=alpha, sigma=sigma,  param_grid=param_grid)
    # mDRFL.search_best(time_series, target_centroids=target_centroids)
    # print(f"Elapsed time workers={cpu_count() - 2}: {time.time() - st}")
    #
    # # Results
    # results = mDRFL.cv_results()
    # best_params = mDRFL.best_params()
    # top = results.head()
    # print(top)
    #
    # best_drfl = DRFL(7, best_params["R"], best_params["C"], best_params["G"], best_params["epsilon"], best_params["L"])
    # best_drfl.fit(time_series)
    # best_drfl.show_results()
    # best_drfl.plot_results(title_fontsize=40, labels_fontsize=35, xticks_fontsize=18,
    #                        yticks_fontsize=20, figsize=(45, 20),
    #                        linewidth_bars=2)
