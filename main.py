import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
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


def group_by_quarter_hour(data: pd.DataFrame, correspondences: dict) -> pd.DataFrame:
    # List of all potential column names based on correspondences
    hour_cols = ["Year", "Month", "Day", "Hour", "Quarter"] + [f"N_{value}" for value in set(correspondences.values())]

    results = []
    for (year, month, day), df in data.groupby(level=[0, 1, 2]):
        df = df.melt(var_name='Minute', value_name='Room')
        df['Hour'] = df['Minute'].str.extract('(\d+)').astype(int) // 60
        df['Quarter'] = (df['Minute'].str.extract('(\d+)').astype(int) % 60) // 15
        df['Room'] = df['Room'].map(correspondences)

        # Count occurrences by hour
        hourly_data = df.pivot_table(index=["Hour", "Quarter"], columns="Room", aggfunc='size', fill_value=0)
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
    if "Quarter" not in feat_extraction.columns.tolist():
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])

    else:
        feat_extraction["Quarter"] = feat_extraction["Quarter"] * 15
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])
        for row in range(feat_extraction.shape[0]):
            feat_extraction.loc[row, "Date"] = feat_extraction.loc[row, "Date"] + pd.Timedelta(
                seconds=feat_extraction.loc[row, "Quarter"] * 60)
        # feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour", "Quarter"]])

    feat_extraction.set_index("Date", inplace=True)
    room_time_series = feat_extraction[f"N_{room}"]

    return room_time_series


def plot_groundtruth(time_series: pd.Series, room: str, top_days: int = 30, figsize=(30, 30), barcolors="blue",
                     linewidth=1.5, save_dir: str = None):
    date = time_series.index
    top_days = min(top_days, len(date) // (24 * 4))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(top_days, 1, figure=fig)
    for i in range(top_days):
        x_hour_minutes = [f"{hour:02}:{minute:02}" for hour in range(24) for minute in range(0, 60, 15)]
        ax = fig.add_subplot(gs[i, 0])
        ax.bar(np.arange(0, 24 * 4, 1), time_series[i * 24 * 4:(i + 1) * 24 * 4],
               color=barcolors, edgecolor="black", linewidth=linewidth)
        ax.set_title(f"N {room}; Date {date[i * 24 * 4].year} / {date[i * 24 * 4].month} / {date[i * 24 * 4].day}",
                     fontsize=20)
        ax.set_xlabel("Time", fontsize=15)
        ax.set_ylabel("N minutes", fontsize=15)
        ax.set_xticks(np.arange(0, 24 * 4, 2), labels=[x for idx, x in enumerate(x_hour_minutes) if idx % 2 == 0],
                      rotation=90)
        ax.grid(True)
        ax.set_ylim(0, 17)
        ax.set_xlim(-1, 24 * 4 + 1)

        # Annotate height of the bar
        for idx, value in enumerate(time_series[i * 24 * 4:(i + 1) * 24 * 4]):
            ax.text(idx, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)

    plt.show()


if __name__ == "__main__":
    # time_series = pd.Series([60, 60, 60, 60, 60, 60, 2, 2, 60, 60, 60, 60, 60, 60, 2, 2, 60, 60, 60, 60, 60, 60])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # time_series_2 = pd.Series([0, 0, 60, 60, 60, 60, 60, 0, 0, 20, 40, 60, 30, 0, 0, 0, 60, 60, 60, 60, 0, 0, 20, 40, 60, 30, 0, 0, 0, 0, 60, 60, 60, 60, 0, 0, 20, 40, 60, 30, 0, 0])
    # time_series_2.index = pd.date_range(start="2024-01-01", periods=len(time_series_2))

    # json_dictionary_file = "data/dictionary_rooms.json"
    # data_dir = "data/activities-simulation-easy.csv"
    # correspondencies = obtain_correspondencies(json_dictionary_file)
    # df = load_data(data_dir)
    # feat_extraction = group_by_quarter_hour(df, correspondencies)
    # print(feat_extraction.head(300))
    # feat_extraction.to_csv("data/out_feat_extraction_quarters.csv", index=False)

    # os.makedirs("groundtruth_figs", exist_ok=True)
    # correspondencies = obtain_correspondencies("data/dictionary_rooms.json")
    # base_colors = cm.rainbow(np.linspace(0, 1, len(correspondencies)))
    # for room, room_name in correspondencies.items():
    #     room_time_series = get_time_series("data/out_feat_extraction_quarters.csv", room_name)
    #     plot_groundtruth(time_series=room_time_series, room=room_name,
    #                      top_days=15, figsize=(30, 50),
    #                      barcolors=base_colors[room - 1],
    #                      save_dir=f"groundtruth_figs/{room_name}.png")

    # plot_groundtruth(room_time_series, top_days=15, figsize=(30, 50))

    # json_dictionary_file = "data/dictionary_rooms.json"
    # data_dir = "data/activities-simulation-easy.csv"
    # correspondencies = obtain_correspondencies(json_dictionary_file)
    # df = load_data(data_dir)
    # feat_extraction = group_by_hour(df, correspondencies)
    # pd.set_option('display.max_columns', None, 'display.max_rows', None)
    # feat_extraction.to_csv("data/out_feat_extraction.csv", index=False)

    # room_time_series = get_time_series("data/out_feat_extraction.csv", "room").loc["2024-01-01 00:00:00":"2024-03-01 00:00:00"]

    #     drgs = DRGS(length_range=(3, 100), R=5, C=60, G=30, epsilon=1, L=1, fusion_distance=0.0001)
    #     drgs.fit(room_time_series)
    #     os.makedirs("plot_hours_routines", exist_ok=True)
    #     drgs.results_per_hour_day(top_days=15, figsize=(30, 60), save_dir="plot_hours_routines",
    #                               bars_linewidth=2, show_background_annotations=True)

    # room_time_series = get_time_series("data/out_feat_extraction_quarters.csv", "room").loc[
    #                    "2024-02-01 00:00:00":"2024-02-20 00:00:00"]

    gym_hour = get_time_series("data/out_feat_extraction.csv", "gym").loc["2024-01-01 00:00:00":"2024-03-01 00:00:00"]
    gym_quarters = get_time_series("data/out_feat_extraction_quarters.csv", "gym").loc["2024-02-01 00:00:00":"2024-02-20 00:00:00"]

    drgs_hours = DRGS(length_range=(2, 100), R=5, C=4, G=20, epsilon=1, L=1, fusion_distance=0.0001)
    drgs_quarters = DRGS(length_range=(3, 100), R=3, C=10, G=5, epsilon=1, L=1, fusion_distance=0.0001)

    os.makedirs("plot_hours_routines", exist_ok=True)
    drgs_hours.fit(gym_hour)
    drgs_hours.results_per_hour_day(top_days=15, figsize=(30, 60), save_dir="plot_hours_routines",
                                    bars_linewidth=2, show_background_annotations=True)
    tree_hours = drgs_hours.convert_to_cluster_tree()
    tree_hours.plot_tree(title="Final node evolution", save_dir="plot_hours_routines/final_tree_hours.png",
                         figsize=(7, 7))

    os.makedirs("plot_quarters_routines", exist_ok=True)
    drgs_quarters.fit(gym_quarters)
    drgs_quarters.results_per_quarter_hour(top_days=15, figsize=(50, 60), save_dir="plot_quarters_routines",
                                           bars_linewidth=2, show_background_annotations=True)
    tree_quarters = drgs_quarters.convert_to_cluster_tree()
    tree_quarters.plot_tree(title="Final node evolution", save_dir="plot_quarters_routines/final_tree_quarters.png",
                            figsize=(14, 14))

    # drgs = DRGS(length_range=(5, 100), R=2, C=500, G=45, epsilon=1, L=1, fusion_distance=0.0001)
    # drgs.fit(room_time_series)
    # os.makedirs("resuldts", exist_ok=True)
    # drgs.plot_separate_hierarchical_results(title_fontsize=35, labels_fontsize=30, yticks_fontsize=20,
    #                                         linewidth_bars=3, vline_width=3, figsize=(50, 25),
    #                                         xlim=(0, 300), show_xticks=False, save_dir="results")
    # routines = drgs.get_results()
    # tree = routines.convert_to_cluster_tree()
    # tree.plot_tree(title="Final node evolution", save_dir="results/final_tree.png", figsize=(14,14))

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
