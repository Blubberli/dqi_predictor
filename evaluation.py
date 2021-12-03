import pandas as pd
import argparse
import os
import numpy as np
import statistics
from collections import defaultdict


# read in all classification reports and compute the average prec, recall, f1, f1 macro, accuracy, weighted avg with standard deviation


def read_classification_report(file):
    report_all = {}
    report_label = defaultdict(dict)
    for line in open(file).readlines():
        line = line.strip()
        parts = ' '.join(line.split()).split(" ")
        if parts[0] == "precision" or len(parts) <= 2:
            continue

        if len(parts) == 5:
            prec = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            label = parts[0]
            report_label[label]["precision"] = prec
            report_label[label]["recall"] = recall
            report_label[label]["f1"] = f1
        else:
            measure = parts[0]
            if measure == "accuracy":
                result = float(parts[1])
                report_all[measure] = result
            else:
                result = float(parts[2])
                report_all[measure] = result
    return report_all, report_label


def gather_all_report_files(output_dir):
    dev_reports = []
    test_reports = []
    for subdir in os.listdir(output_dir):
        if not "txt" in subdir:
            dev_report = "%s/%s/%s_dev_results.txt" % (output_dir, subdir, subdir)
            test_report = "%s/%s/%s_test_results.txt" % (output_dir, subdir, subdir)
            dev_reports.append(dev_report)
            test_reports.append(test_report)
    return dev_reports, test_reports


def average_all(reports):
    s = "metric\tmean\tdeviation\n"
    for k in reports[0].keys():
        all_vals = [report[k] for report in reports]
        s += "%s\t%.2f\t%.2f\n" % (k, statistics.geometric_mean(all_vals), statistics.stdev(all_vals))
    return s


def average_class(reports):
    s = "label\tmetric\tmean\tdeviation\n"
    for metric in reports[0]['0'].keys():
        for k in reports[0].keys():
            all_vals = [report[k][metric] for report in reports]
            s += "%s\t%s\t%.2f\t%.2f\n" % (metric,k, statistics.mean(all_vals), statistics.stdev(all_vals))
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    dev_reports, test_reports = gather_all_report_files(args.output_dir)
    dev_results_all = []
    test_results_all = []
    dev_results_label = []
    test_results_label = []
    for dev_file in dev_reports:
        all, label = read_classification_report(dev_file)
        dev_results_all.append(all)
        #dev_results_label.append(label)
    for test_file in test_reports:
        all, label = read_classification_report(test_file)
        test_results_all.append(all)
        #test_results_label.append(label)
    print("results on dev all")
    print(average_all(dev_results_all))
    #print(average_class(dev_results_label))
    print("\nresults on test all")
    print(average_all(test_results_all))
    #print(average_class(test_results_label))
