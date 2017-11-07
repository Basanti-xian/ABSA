import os, re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict

def PlotCatPolarityScatter(movingAvg_cat_polarity_grouped_merged_mat_list, timeline, cat_list, names):
    N, K = movingAvg_cat_polarity_grouped_merged_mat_list[0].shape  # N time slot, K categories
    ind = np.arange(N)  # the x-axis locations for the timeline
    width = 0.1  # the width of the bars
    plots = []
    color_map = ['red', 'blue', 'orange', 'purple', 'olive', 'cyan']
    marker_map = [".", "o", "^", "*", "h", "x"]
    for k in range(K):
        for name_ind, mat in enumerate(movingAvg_cat_polarity_grouped_merged_mat_list):
            color = color_map[name_ind]
            marker = marker_map[name_ind]
            p = plt.scatter(ind, mat[:, k], alpha=1, marker=marker, color=color)
            p = plt.plot(ind, mat[:, k], color=color, linewidth=1.5)
            plots.append(p)

        plt.ylabel('Polarity')
        plt.xticks(ind + width / 4, timeline, rotation=265)
        plt.yticks(np.arange(-0.1, 0.1, 0.05))
        plt.title('{} Polarity Distribution'.format(cat_list[k]))
        labels = ['{}'.format(x) for x in names]
        leg = plt.legend([p[0] for p in plots], labels, fancybox=True, prop={'size': 15})
        leg.get_frame().set_alpha(0.5)
        plt.show()
        # plt.savefig('{} Polarity Distribution'.format(cat_list[k]))


def GetMonthTimeline(date_list):
    month_timeline = []
    # Need the date from panda
    for f in date_list:
        time = str(f).split()[0][:-3]
        month_timeline.append(time)
    month_timeline = np.asarray(month_timeline)
    timeline = sorted(set(month_timeline), key=lambda x: datetime.datetime.strptime(x, '%Y-%m'))
    return month_timeline, timeline

def row_moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.hstack((a[0:n-1],(ret[n - 1:] / n)))

def movingAvg(cat_polarity_mat_grouped):
    m = cat_polarity_mat_grouped.T
    ma = np.zeros(shape= (m.shape))
    for ridx, row in enumerate(m):
        avg_row = row_moving_average(row)
        ma[ridx] = avg_row
    return ma.T


def GetGroupedCatPolarityMat(slot_timeline, no_cat, cat_polarity_mat):
    num_groups = len(set(slot_timeline))
    cat_polarity_mat_grouped = np.zeros((num_groups, no_cat))
    for i, slot in enumerate(sorted(set(slot_timeline))):
        cat_polarity_mat_grouped[i, :] = np.mean(cat_polarity_mat[slot_timeline == slot, :], axis=0)
    return cat_polarity_mat_grouped

def GetMergedCatPolarityMat(merged_cat_list, no_groups, cat_polarity_mat):
    num_cat = len(merged_cat_list)
    cat_polarity_mat_merged = np.zeros((no_groups, num_cat))
    for i, cat in enumerate(merged_cat_list):
        if 0==i:
            cat_polarity_mat_merged[:, 0] = np.mean(cat_polarity_mat[:, 0:2], axis=1)
        elif i==1:
            cat_polarity_mat_merged[:, 1] = np.mean(cat_polarity_mat[:, 2:4], axis=1)
        else:
            cat_polarity_mat_merged[:, i] = cat_polarity_mat[:, i+2]
    return cat_polarity_mat_merged

GroundTruth_flag = True #False #
def main():
    names = ['Cara', 'Alan', 'Tim', 'Other1', 'Other2']
    filenames = ['cara_v7.csv', 'alan_82_v4.csv', 'tim_52_v4.csv', 'other1v6.csv','Others2_v4.csv']
    pred = [float(l) for l in open('../Preds.csv').xreadlines()]

    cat_list = ['Finance', 'Statement', 'Energy', 'Business', 'Boss', 'Collaborators', 'Company', 'Health', 'Legal', 'Power scheduling task', 'Stock']
    merged_cat_list = ['Finance+Statement', 'Energy+Business', 'Boss', 'Collaborators', 'Company', 'Health', 'Legal', 'Power scheduling task', 'Stock']
    no_cat = len(cat_list)

    senlist_df_list = []
    for file in filenames:
        senlist_df = pd.read_csv(file)
        senlist_df_list.append(senlist_df)

    time_dict_list = []
    for senlist_df in senlist_df_list:
        time_dict = defaultdict(lambda: defaultdict(list))
        for ind, date in enumerate(senlist_df.Date):
            cat = str(senlist_df.Aspect_category[ind]).strip()
            if cat not in cat_list:
                print cat, ind
                raw_input()
            if GroundTruth_flag:
                ###for ground truth
                p = senlist_df.AspectCategorySentiment[ind]
            else:
                ###for SVM prediction
                p = pred[ind]
            if time_dict[date][cat]:
                time_dict[date][cat].append(p)
            else:
                cat_polarity_dict = {}
                cat_polarity_dict[cat] = [p]
                time_dict[date].update(cat_polarity_dict)
        time_dict_list.append(time_dict)

    Date_list_unsort = []
    for senlist_df in senlist_df_list:
        Date_list_unsort = Date_list_unsort + list(senlist_df.Date)
    Date_list_unsort = list(map(lambda x: str(x), set(Date_list_unsort)))
    Date_list = sorted(Date_list_unsort, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H.%M.%S'))

    sen_cat_polarity_mat_list = []
    ## sen: row---> a list of polarity for different cat in order
    for time_dict in time_dict_list:
        sen_cat_polarity_mat =[]
        for date in Date_list:
            cat_polarity_list = [0 for i in range(no_cat)]
            for cat in time_dict[date].keys():
                cat_polarity_list[cat_list.index(cat)] = sum(time_dict[date][cat])/len(time_dict[date][cat])
            sen_cat_polarity_mat.append(cat_polarity_list)
        sen_cat_polarity_mat = np.array(sen_cat_polarity_mat)
        sen_cat_polarity_mat_list.append(sen_cat_polarity_mat)


    month_timeline, timeline = GetMonthTimeline(Date_list)
    cat_polarity_grouped_mat_list = [GetGroupedCatPolarityMat(month_timeline, no_cat, mat) for mat in sen_cat_polarity_mat_list]

    no_group = len(timeline)
    cat_polarity_grouped_merged_mat_list = [GetMergedCatPolarityMat(merged_cat_list, no_group, mat)  for mat in cat_polarity_grouped_mat_list]

    movingAvg_cat_polarity_grouped_merged_mat_list = [movingAvg(mat) for mat in cat_polarity_grouped_merged_mat_list]

    PlotCatPolarityScatter(movingAvg_cat_polarity_grouped_merged_mat_list, timeline, merged_cat_list, names)


if __name__ == '__main__':
    main()













