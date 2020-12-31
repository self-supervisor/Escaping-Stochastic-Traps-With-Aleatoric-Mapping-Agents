import pytest
import glob
import numpy as np

def test_get_all_files_of_same_type():
    from analyse_hyperparam import get_experiments, get_all_files_of_same_type
    
    all_csvs = glob.glob("*.csv")
    experiments, filtered_csvs = get_experiments()
    resultant_dict = get_all_files_of_same_type(experiments, filtered_csvs)
    for key, value in resultant_dict.items():
        run_types = []
        for a_value in value:
            run_types.append(a_value[:-6])
        assert len(set(run_types)) == 1

def test_get_experiments():
    from analyse_hyperparam import get_experiments
    import re
    
    def hasNumbers(inputString):
        """
        https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
        """
        return any(char.isdigit() for char in inputString)
        
    all_csvs = glob.glob("*.csv")
    experiments, filtered_csvs = get_experiments()
    
    assert all_csvs != filtered_csvs
    
    # check csvs are filtered to not include curiosity == False
    for i, value in enumerate(filtered_csvs):
        if "curiosity_False" in value:
            raise ValueError
    
    # check all seed numbers have been removed from the csv strigns
    for i, value in enumerate(filtered_csvs):
        assert hasNumbers(value[8:-6]) == False
    
    # check that all duplicate experiments have been removed
    assert len(filtered_csvs) == len(set(filtered_csvs))
    

def make_fake_csvs(first_csv_string, second_csv_string, first_csv_name="fake1.csv", second_csv_name="fake2.csv"):
    import csv
    import os
    
    try:
        os.remove(first_csv_name)
    except FileNotFoundError:
        pass
    try:
        os.remove(second_csv_name)
    except FileNotFoundError:
        pass
    
    assert len(first_csv_string) == len(second_csv_string)
    
    for i in range(len(first_csv_string)):
        with open(first_csv_name, "a") as fp:
            wr = csv.writer(fp)
            wr.writerow([float(first_csv_string[i][0]), float(first_csv_string[i][1]), float(first_csv_string[i][2])])
    
    for i in range(len(second_csv_string)):
        with open(second_csv_name, "a") as fp:
            wr = csv.writer(fp)
            wr.writerow([float(second_csv_string[i][0]), float(second_csv_string[i][1]), float(second_csv_string[i][2])])
        
def test_average_over_run_type():
    from analyse_hyperparam import average_over_run_type
    
    first_csv_string = [[0.01,0.1,13],[0.001,0.1,12],[0.0001,0.1,13], [0.01,1,17], [0.001,1,20], [0.0001,1,18]]
    second_csv_string = [[0.01,0.1,17], [0.001,0.1,16], [0.0001,0.1,16], [0.01,1,20], [0.001,1,20], [0.0001,1,21]]
    make_fake_csvs(first_csv_string, second_csv_string, first_csv_name="fake1.csv", second_csv_name="fake2.csv")
    result = average_over_run_type(["fake1.csv", "fake2.csv"])
    manually_computed_result = np.mean(np.array([np.array(first_csv_string), np.array(second_csv_string)]), axis=0)
    assert np.array_equal(result, manually_computed_result) == True    
    
    import os
    
    os.remove("fake1.csv")
    os.remove("fake2.csv")
    
    
def test_make_run_list():
    from analyse_hyperparam import make_run_list, average_over_run_type
    
    first_csv_string = [[0.01,0.1,13],[0.001,0.1,12],[0.0001,0.1,13], [0.01,1,17], [0.001,1,20], [0.0001,1,18]]
    second_csv_string = [[0.01,0.1,17], [0.001,0.1,16], [0.0001,0.1,16], [0.01,1,20], [0.001,1,20], [0.0001,1,21]]
    make_fake_csvs(first_csv_string, second_csv_string, first_csv_name="fake1.csv", second_csv_name="fake2.csv")
    result = average_over_run_type(["fake1.csv", "fake2.csv"])
    run_list = make_run_list(["fake1.csv", "fake2.csv"])
    manual_run_list = np.stack([np.array(first_csv_string, dtype=np.float64), np.array(second_csv_string, dtype=np.float64)], axis=0)
    run_list = np.array(run_list, dtype=np.float64)
    assert np.array_equal(run_list, manual_run_list) == True
    
    import os
    
    os.remove("fake1.csv")
    os.remove("fake2.csv")
    
    
def test_average_over_multiple_run_types():
    from analyse_hyperparam import average_over_multiple_run_types, average_over_run_type
    
    first_csv_string = [[0.01,0.1,13],[0.001,0.1,12],[0.0001,0.1,13], [0.01,1,17], [0.001,1,20], [0.0001,1,18]]
    second_csv_string = [[0.01,0.1,17], [0.001,0.1,16], [0.0001,0.1,16], [0.01,1,20], [0.001,1,20], [0.0001,1,21]]
    make_fake_csvs(first_csv_string, second_csv_string, first_csv_name="frames_unit_test_noisy_tv_False_random_2.csv", second_csv_name="frames_unit_test_noisy_tv_True_random_2.csv")
    files_of_same_type = {}
    files_of_same_type["frames_unit_test_noisy_tv_False"] = ["frames_unit_test_noisy_tv_False_random_2.csv"]
    files_of_same_type["frames_unit_test_noisy_tv_True"] = ["frames_unit_test_noisy_tv_True_random_2.csv"]
    average_performance_with_and_without_tv = average_over_multiple_run_types(files_of_same_type, ["frames_unit_test_noisy_tv_False", "frames_unit_test_noisy_tv_True"])
    manually_computed_result = np.mean(np.array([np.array(first_csv_string), np.array(second_csv_string)]), axis=0)
    assert np.array_equal(average_performance_with_and_without_tv, manually_computed_result) == True

    import os
    
    os.remove("frames_unit_test_noisy_tv_False_random_2.csv")
    os.remove("frames_unit_test_noisy_tv_True_random_2.csv")
    
def test_get_best_hyperparam():
    from analyse_hyperparam import get_best_hyperparam, average_over_multiple_run_types

    fake_average_performance_with_and_without_tv = np.zeros((3, 3))
    fake_average_performance_with_and_without_tv[1] = np.ones((3,))
    assert np.array_equal(get_best_hyperparam(fake_average_performance_with_and_without_tv), fake_average_performance_with_and_without_tv[1]) == True
