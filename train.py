from typing import Optional

import pandas as pd
import numpy as np
import os

from src.DRFL import DRGS
from src.structures import HierarchyRoutine, Routines, Cluster


def get_time_series(path_to_feat_extraction: str, room: str, select_month: Optional[str] = None) -> pd.Series:
    feat_extraction = pd.read_csv(path_to_feat_extraction)
    if "Quarter" not in feat_extraction.columns.tolist():
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])

    else:
        feat_extraction["Date"] = pd.to_datetime(feat_extraction[["Year", "Month", "Day", "Hour"]])
        for row in range(feat_extraction.shape[0]):
            feat_extraction.loc[row, "Date"] = feat_extraction.loc[row, "Date"] + pd.Timedelta(
                minutes=feat_extraction.loc[row, "Quarter"] * 15)

    feat_extraction.set_index("Date", inplace=True)
    if select_month is not None:
        feat_extraction = feat_extraction[feat_extraction["Month"] == int(select_month)]

    room_time_series = feat_extraction[f"N_{room}"]

    return room_time_series


def get_path(dificulty: str):
    if not isinstance(dificulty, str):
        raise TypeError(f"Dificulty must be a string. Got {type(dificulty).__name__}")

    if dificulty not in ["easy", "medium", "hard"]:
        raise ValueError(f"Dificulty must be one of 'easy', 'medium' or 'hard'. Got {dificulty}")

    return f"data/Synthetic Activity Dataset/402E/{dificulty}/out_feat_extraction_quarters.csv"


def get_precision(target: Cluster, estimated: Cluster) -> float:
    P = target.get_sequences().get_starting_points(to_array=True)
    Q = estimated.get_sequences().get_starting_points(to_array=True)
    return len(set(P).intersection(set(Q))) / len(Q)


def get_recall(target: Cluster, estimated: Cluster) -> float:
    P = target.get_sequences().get_starting_points(to_array=True)
    Q = estimated.get_sequences().get_starting_points(to_array=True)
    return len(set(P).intersection(set(Q))) / len(P)


def get_f1_score(target: Cluster, estimated: Cluster) -> float:
    precision = get_precision(target, estimated)
    recall = get_recall(target, estimated)

    # Avoid division by zero
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def dist_exp_decay_1(target: Routines, estimated: Routines, sigma_1: float | int) -> float:
    P = target.get_centroids()
    Q = estimated.get_centroids()

    return np.exp(-(len(P) - len(Q)) ** 2 / sigma_1)


def dist_exp_decay_2(target: HierarchyRoutine, estimated: HierarchyRoutine, sigma_2: float | int) -> float:
    Dt = max(target.keys)
    De = max(estimated.keys)
    return np.exp(-(Dt - De) ** 2 / sigma_2)


def match_cluster(target: Cluster, estimated: Routines) -> Cluster:
    distances = []
    for id_cluster, cluster in enumerate(estimated):
        distance = np.max(np.abs(target.centroid - cluster.centroid))
        distances.append(distance)

    return estimated[int(np.argmin(distances))]


def media_f1_scores(target: Routines, estimated: Routines) -> float:
    f1_scores = []
    for target_cluster in target:
        estimated_cluster = match_cluster(target_cluster, estimated)
        f1_scores.append(get_f1_score(target_cluster, estimated_cluster))

    return np.mean(f1_scores)


def media_hierarchy_scores(target: HierarchyRoutine, estimated: HierarchyRoutine, alpha: float,
                           sigma_1: float) -> float:
    scores = []
    m_max = min(max(target.keys), max(estimated.keys))
    m_min = max(min(target.keys), min(estimated.keys))

    for hierarchy in range(m_min, m_max + 1):
        target_routine = target[hierarchy]
        estimated_routine = estimated[hierarchy]
        scores.append(hierarchy_score(target_routine, estimated_routine, alpha, sigma_1))

    return np.mean(scores)


def hierarchy_score(target: Routines, estimated: Routines, alpha: float, sigma_1: float) -> float:
    return alpha * dist_exp_decay_1(target, estimated, sigma_1) + (1 - alpha) * media_f1_scores(target, estimated)


def global_score(target: HierarchyRoutine, estimated: HierarchyRoutine, alpha: float, beta: float, sigma_1: float,
                 sigma_2: float) -> float:
    return beta * dist_exp_decay_2(target, estimated, sigma_2) + (1 - beta) * media_hierarchy_scores(target, estimated,
                                                                                                     alpha, sigma_1)


if __name__ == "__main__":
    ROOM = "Therapy Room"
    easy = get_time_series(path_to_feat_extraction=get_path("easy"), room=ROOM,
                           select_month="3")
    medium = get_time_series(path_to_feat_extraction=get_path("medium"), room=ROOM,
                             select_month="3")
    hard = get_time_series(path_to_feat_extraction=get_path("hard"), room=ROOM,
                           select_month="3")

    # get groundtruth
    drgs_easy = DRGS(length_range=(3, 100), R=3, C=10, G=8, epsilon=0.5, L=1, fusion_distance=0.001)
    drgs_easy.fit(easy, verbose=False)
    groundtruth = drgs_easy.get_results()

    # params
    params = {
        "Data": ["medium", "hard"],
        "R": [1, 3, 5, 7, 8],
        "C": [5, 10, 15, 20, 25],
        "G": [5, 8, 11, 14],
        "L": [0, 1, 2, 3]
    }

    # Load metrics if exists
    if os.path.exists("results/metrics.csv"):
        metrics = pd.read_csv("results/metrics.csv")

    else:
        metrics = pd.DataFrame(columns=["Data", "R", "C", "G", "L", "Score"])

    alpha, beta, sigma_1, sigma_2 = 0.5, 0.5, 15, 15

    for data in params["Data"]:
        for R in params["R"]:
            for C in params["C"]:
                for G in params["G"]:
                    for L in params["L"]:
                        if metrics[(metrics["Data"] == data) & (metrics["R"] == R) & (metrics["C"] == C) & (
                                metrics["G"] == G) & (metrics["L"] == L)].shape[0] == 0:

                            if data == "medium":
                                time_series = medium

                            else:
                                time_series = hard

                            drgs = DRGS(length_range=(3, 100), R=R, C=C, G=G, epsilon=0.5, L=L, fusion_distance=0.001)
                            drgs.fit(time_series, verbose=False)
                            estimated = drgs.get_results()

                            score = global_score(groundtruth, estimated, alpha, beta, sigma_1, sigma_2)
                            metrics.loc[len(metrics)] = [data, R, C, G, L, score]

                            print(f"Data: {data}, R: {R}, C: {C}, G: {G}, L: {L}, Score: {score}")

                            metrics.sort_values(by=["Data", "R", "C", "G", "L", "Score"]).to_csv("results/metrics.csv", index=False)
                        else:
                            print(f"Data: {data}, R: {R}, C: {C}, G: {G}, L: {L} already computed")
