import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_mean_and_std_dev(csv_paths, quantity):
    runs = []
    for csv_path in csv_paths:
        num_frames_mean = extract_num_frames_mean_from_csv(csv_path, quantity)
        num_frames_mean = np.array([float(i) for i in num_frames_mean])
        runs.append(num_frames_mean)
    min_length = min([len(i) for i in runs])
    runs = [run[:min_length] for run in runs]
    runs = np.array(runs)
    return np.mean(runs, axis=0), np.std(runs, axis=0)


def plot_mean_and_uncertainty(mean, std, label, num_of_points, multiply_factor):
    plt.plot(np.array(range(num_of_points)) * multiply_factor, mean, label=label)
    plt.fill_between(
        np.array(range(num_of_points)) * multiply_factor,
        mean - std,
        mean + std,
        alpha=0.2,
    )


def extract_num_frames_mean_from_csv(csv_path, quantity):
    df = pd.read_csv(csv_path)
    df = df.drop(df[df["update"] == "update"].index)
    num_frames_mean = df[quantity]
    return np.array(num_frames_mean)


def get_label_from_path(path_string):
    # format storage/noisy_tv_True_curiosity_True_uncertainty_True_random_seed_29_coefficient_0.0005
    label = path_string.split("/")[1].split("_random")[0].replace("_", " ")
    return label


def plot(title, path_strings, quantity):
    from matplotlib.pyplot import figure

    figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    ### formatting ###
    plt.rcParams["axes.formatter.limits"] = [-5, 5]
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize="medium")
    plt.rc("ytick")
    ### calculations ###
    for path_string_list in path_strings:
        path_string_list = [s + "/log.csv" for s in path_string_list]
        mean, std = get_mean_and_std_dev(path_string_list, quantity)
        frames = path_string_list[0].split("/")[1][0:9]
        label = get_label_from_path(path_string_list[0])
        plot_mean_and_uncertainty(mean, std, label, len(mean), 2.5e5 / len(mean))

    plt.xlabel("Total Frames Elapsed", fontsize=15)
    plt.ylabel(quantity.replace("_", " "), fontsize=15)
    plot_title = title + " " + quantity
    plt.title(plot_title.replace("_", " "))
    plt.legend(loc="best")
    plt.savefig(plot_title + "_" + frames + ".png")


def main():
    quantities_to_plot = ["intrinsic_rewards", "novel_states_visited", "uncertainties"]
    all_strings = glob.glob("storage/*")
    print(all_strings)
    for quantity in quantities_to_plot:
        Curious_True_Noisy_True_Uncertain_True = []
        Curious_False_Noisy_True_Uncertain_False = []
        Curious_False_Noisy_False_Uncertain_False = []
        Curious_True_Noisy_False_Uncertain_True = []
        Curious_True_Noisy_False_Uncertain_False = []
        Curious_True_Noisy_True_Uncertain_False = []

        # format: noisy_tv_True_curiosity_True_uncertainty_True_random_seed_29_coefficient_0.0005
        for string in all_strings:
            if "curiosity_True" in string:
                if "noisy_tv_True" in string:
                    if "uncertainty_True" in string:
                        Curious_True_Noisy_True_Uncertain_True.append(string)

            if "curiosity_False" in string:
                if "noisy_tv_True" in string:
                    if "uncertainty_False" in string:
                        Curious_False_Noisy_True_Uncertain_False.append(string)

            if "curiosity_False" in string:
                if "noisy_tv_False" in string:
                    if "uncertainty_False" in string:
                        Curious_False_Noisy_False_Uncertain_False.append(string)

            if "curiosity_True" in string:
                if "noisy_tv_False" in string:
                    if "uncertainty_True" in string:
                        Curious_True_Noisy_False_Uncertain_True.append(string)

            if "curiosity_True" in string:
                if "noisy_tv_False" in string:
                    if "uncertainty_False" in string:
                        Curious_True_Noisy_False_Uncertain_False.append(string)

            if "curiosity_True" in string:
                if "noisy_tv_True" in string:
                    if "uncertainty_False" in string:
                        Curious_True_Noisy_True_Uncertain_False.append(string)

        path_strings_noisy_tv = [
            Curious_True_Noisy_True_Uncertain_True,
            Curious_False_Noisy_True_Uncertain_False,
            Curious_True_Noisy_True_Uncertain_False,
        ]
        path_strings_no_noisy = [
            Curious_False_Noisy_False_Uncertain_False,
            Curious_True_Noisy_False_Uncertain_True,
            Curious_True_Noisy_False_Uncertain_False,
        ]
        print(path_strings_noisy_tv)
        plot(
            "With Noisy TV, Minigrid 6x6 Averaged Over 10 Random Seeds",
            path_strings_noisy_tv,
            quantity,
        )
        plot(
            "Without Noisy TV, Minigrid 6x6 Averaged Over 10 Random Seeds",
            path_strings_no_noisy,
            quantity,
        )


if __name__ == "__main__":
    main()