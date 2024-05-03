import json
import warnings

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


def extract_data_grouped_by_hour(path_to_data: str, path_to_dictionary: str) -> pd.DataFrame:
    data = pd.read_csv(path_to_data, skiprows=1, header=None)
    correspondencies = json.load(open(path_to_dictionary))
    data.columns = ["Year", "Month", "Day"] + [f"Sequence_{x}" for x in range(1439)]
    out_columns = ["Year", "Month", "Day", "Hour"] + [f"N_{x}" for x in correspondencies.keys()]
    new_df = pd.DataFrame(columns=out_columns)
    for row in range(data.shape[0]):
        for hour in range(24):
            sequence = data.iloc[row, 3 + hour * 60:3 + (hour + 1) * 60].replace(np.nan, 0).values
            frequency_rooms = [np.sum(sequence == i) for i in correspondencies.values()]
            year, month, day = data.iloc[row, 0], data.iloc[row, 1], data.iloc[row, 2]
            new_df.loc[len(new_df)] = [year, month, day, hour] + frequency_rooms

    return new_df


def extract_data_grouped_by_quarter_hour(path_to_data: str, path_to_dictionary: str) -> pd.DataFrame:
    data = pd.read_csv(path_to_data, skiprows=1, header=None)
    correspondencies = json.load(open(path_to_dictionary))
    data.columns = ["Year", "Month", "Day"] + [f"Sequence_{x}" for x in range(1439)]
    out_columns = ["Year", "Month", "Day", "Hour", "Quarter"] + [f"N_{x}" for x in correspondencies.keys()]
    new_df = pd.DataFrame(columns=out_columns)
    for row in range(data.shape[0]):
        for hour in range(24):
            for quarter in range(4):
                sequence = data.iloc[row, 3 + hour * 60 + quarter * 15:3 + hour * 60 + (quarter + 1) * 15].replace(
                    np.nan, 0).values
                frequency_rooms = [np.sum(sequence == i) for i in correspondencies.values()]
                year, month, day = data.iloc[row, 0], data.iloc[row, 1], data.iloc[row, 2]
                new_df.loc[len(new_df)] = [year, month, day, hour, quarter] + frequency_rooms

    return new_df


def get_time_series(path_to_feat_extraction: str, room: str) -> pd.Series:
    feat_extraction = pd.read_csv(path_to_feat_extraction)
    if "Quarter" not in feat_extraction.columns.tolist():
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])

    else:
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])
        for row in range(feat_extraction.shape[0]):
            feat_extraction.loc[row, "Date"] = feat_extraction.loc[row, "Date"] + pd.Timedelta(
                minutes=feat_extraction.loc[row, "Quarter"] * 15)

    feat_extraction.set_index("Date", inplace=True)
    room_time_series = feat_extraction[f"N_{room}"]

    return room_time_series


def plot_quarters_groundtruth(time_series: pd.Series, room: str, top_days: int = 30, figsize=(30, 30), barcolors="blue",
                              linewidth=1.5, show_plot: bool = True, save_dir: str = None):
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

    if show_plot:
        plt.show()


def plot_hours_groundtruth(time_series: pd.Series, room: str, top_days: int = 30, figsize=(30, 30), barcolors="blue",
                           linewidth=1.5, show_plot: bool = True, save_dir: str = None):
    date = time_series.index
    top_days = min(top_days, len(date) // 24)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(top_days, 1, figure=fig)
    for i in range(top_days):
        x_hour = [f"{hour:02}:00" for hour in range(24)]
        ax = fig.add_subplot(gs[i, 0])
        ax.bar(np.arange(0, 24, 1), time_series[i * 24:(i + 1) * 24],
               color=barcolors, edgecolor="black", linewidth=linewidth)
        ax.set_title(f"N {room}; Date {date[i * 24].year} / {date[i * 24].month} / {date[i * 24].day}",
                     fontsize=20)
        ax.set_xlabel("Time", fontsize=15)
        ax.set_ylabel("N minutes", fontsize=15)
        ax.set_xticks(np.arange(0, 24, 1), labels=x_hour, rotation=90)
        ax.grid(True)
        ax.set_ylim(0, 70)
        ax.set_xlim(-1, 25)

        # Annotate height of the bar
        for idx, value in enumerate(time_series[i * 24:(i + 1) * 24]):
            ax.text(idx, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)

    if show_plot:
        plt.show()


if __name__ == "__main__":
    # time_series = pd.Series([60, 60, 60, 60, 60, 60, 2, 2, 60, 60, 60, 60, 60, 60, 2, 2, 60, 60, 60, 60, 60, 60])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # time_series = pd.Series([1, 3, 6, 4, 2, 1, 2, 3, 6, 4, 1, 1, 3, 6, 4, 1])
    # time_series.index = pd.date_range(start="2024-01-01", periods=len(time_series))

    # time_series_2 = pd.Series([0, 0, 60, 60, 60, 60, 60, 0, 0, 20, 40, 60, 30, 0, 0, 0, 60, 60, 60, 60, 0, 0, 20, 40, 60, 30, 0, 0, 0, 0, 60, 60, 60, 60, 0, 0, 20, 40, 60, 30, 0, 0])
    # time_series_2.index = pd.date_range(start="2024-01-01", periods=len(time_series_2))

    for DIFICULTY in ["hard"]:

        ROOT_DATA = "data/data2"
        DICTIONARY_FILE = f"{ROOT_DATA}/metadata/dictionary_rooms.json"
        DIFICULTY_DATA = f"{ROOT_DATA}/{DIFICULTY}"
        DATA_FILE = f"{DIFICULTY_DATA}/activities-simulation-{DIFICULTY}.csv"

        HOUR_EXTRACTED = f"{DIFICULTY_DATA}/out_feat_extraction.csv"
        QUARTER_EXTRACTED = f"{DIFICULTY_DATA}/out_feat_extraction_quarters.csv"

        # Results figs
        RESULTS_FIG = "results/results-data2"
        RESULTS_FIG_DIFICULTY = f"{RESULTS_FIG}/{DIFICULTY}"
        HOUR_RESULTS = f"{RESULTS_FIG_DIFICULTY}/plot_hours_routines"
        QUARTER_RESULTS = f"{RESULTS_FIG_DIFICULTY}/plot_quarters_routines"

        # Create paths for results figs
        os.makedirs(RESULTS_FIG, exist_ok=True)
        os.makedirs(RESULTS_FIG_DIFICULTY, exist_ok=True)
        os.makedirs(HOUR_RESULTS, exist_ok=True)
        os.makedirs(QUARTER_RESULTS, exist_ok=True)

        # Groundtruth figs
        GROUNDTRUTH_ROOT = "figs/groundtruth_data2_figs"
        GROUNDTRUTH_DIFICULTY = f"{GROUNDTRUTH_ROOT}/{DIFICULTY}"
        HOURS_GROUNDTRUTH = f"{GROUNDTRUTH_DIFICULTY}/hours"
        QUARTERS_GROUNDTRUTH = f"{GROUNDTRUTH_DIFICULTY}/quarters"

        # Create paths for groundtruth figs
        os.makedirs(GROUNDTRUTH_ROOT, exist_ok=True)
        os.makedirs(GROUNDTRUTH_DIFICULTY, exist_ok=True)
        os.makedirs(HOURS_GROUNDTRUTH, exist_ok=True)
        os.makedirs(QUARTERS_GROUNDTRUTH, exist_ok=True)

        correspondencies = json.load(open(DICTIONARY_FILE))

        hour_data = extract_data_grouped_by_hour(DATA_FILE, DICTIONARY_FILE)
        quarter_data = extract_data_grouped_by_quarter_hour(DATA_FILE, DICTIONARY_FILE)

        hour_data.to_csv(HOUR_EXTRACTED, index=False)
        quarter_data.to_csv(QUARTER_EXTRACTED, index=False)

        all_rooms = list(correspondencies.keys())
        for room in tqdm(all_rooms):
            st = time.time()
            path_out_hour = f"{HOUR_RESULTS}/{room}"
            path_out_quarter = f"{QUARTER_RESULTS}/{room}"

            os.makedirs(path_out_hour, exist_ok=True)
            os.makedirs(path_out_quarter, exist_ok=True)

            hour_time_series = get_time_series(HOUR_EXTRACTED, room)
            quarter_time_series = get_time_series(QUARTER_EXTRACTED, room)

            # plot_hours_groundtruth(hour_time_series, room, top_days=15, figsize=(30, 60),
            #                        save_dir=f"{HOURS_GROUNDTRUTH}/{room}.png", show_plot=False)
            # plot_quarters_groundtruth(quarter_time_series, room, top_days=15, figsize=(50, 60),
            #                           save_dir=f"{QUARTERS_GROUNDTRUTH}/{room}.png", show_plot=False)

            R1, C1, G1 = 5, 5, 20
            R2, C2, G2 = 3, 15, 5

            if room == "room" and DIFICULTY != "hard":
                R1, C1, G1 = 3, 40, 20
                R2, C2, G2 = 1, 80, 5

            if DIFICULTY == "hard":
                R1, C1, G1 = 13, 5, 30
                R2, C2, G2 = 5, 10, 5

            drgs_hours = DRGS(length_range=(3, 100), R=R1, C=C1, G=G1, epsilon=0.5, L=1, fusion_distance=0.0001)
            drgs_quarters = DRGS(length_range=(3, 100), R=R2, C=C2, G=G2, epsilon=0.5, L=1, fusion_distance=0.0001)

            drgs_hours.fit(hour_time_series)
            drgs_hours.results_per_hour_day(top_days=30, figsize=(30, 120), save_dir=path_out_hour,
                                            bars_linewidth=2, show_background_annotations=True,
                                            show_plot=False)

            tree_hours = drgs_hours.convert_to_cluster_tree()

            if drgs_hours.get_results().is_empty():
                warnings.warn(f"Empty results for room {room}")

            else:
                if len(tree_hours.nodes) > 30:
                    tree_hours.plot_tree(title="Final node evolution",
                                         save_dir=f"{path_out_hour}/final_tree_hours.png",
                                         figsize=(27, 27))

                elif len(tree_hours.nodes) < 7:
                    tree_hours.plot_tree(title="Final node evolution",
                                         save_dir=f"{path_out_hour}/final_tree_hours.png",
                                         figsize=(7, 7))

                else:
                    tree_hours.plot_tree(title="Final node evolution",
                                         save_dir=f"{path_out_hour}/final_tree_hours.png",
                                         figsize=(14, 14))

            drgs_quarters.fit(quarter_time_series)
            drgs_quarters.results_per_quarter_hour(top_days=30, figsize=(50, 120), save_dir=path_out_quarter,
                                                   bars_linewidth=2, show_background_annotations=True,
                                                   show_plot=False)

            tree_quarters = drgs_quarters.convert_to_cluster_tree()

            if drgs_quarters.get_results().is_empty():
                warnings.warn(f"Empty results for room {room}")

            else:
                if len(tree_quarters.nodes) > 30:
                    tree_quarters.plot_tree(title="Final node evolution",
                                            save_dir=f"{path_out_quarter}/final_tree_quarters.png",
                                            figsize=(27, 27))
                elif len(tree_quarters.nodes) < 7:
                    tree_quarters.plot_tree(title="Final node evolution",
                                            save_dir=f"{path_out_quarter}/final_tree_quarters.png",
                                            figsize=(7, 7))

                else:
                    tree_quarters.plot_tree(title="Final node evolution",
                                            save_dir=f"{path_out_quarter}/final_tree_quarters.png",
                                            figsize=(14, 14))

            print(f"Elapsed time for room {room}: {time.time() - st}")

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
