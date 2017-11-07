import os, re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict

def PlotCatPolarityScatter(Cara_cat_polarity_mat_grouped, Alan_cat_polarity_mat_grouped, Tim_cat_polarity_mat_grouped, Other_cat_polarity_mat_grouped, timeline, cat_list, names):
    N, K = Cara_cat_polarity_mat_grouped.shape  # N time slot, K categories
    ind = np.arange(N)  # the x-axis locations for the timeline
    width = 0.1  # the width of the bars
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'gray', 'yellow', 'brown', 'olive', 'cyan']
    marker_map = [".", "o", "^", "8", "s", "*", "h", "H", "x", "D", "_"]
    for k in range(K):
        # color = color_map[k]
        # marker = marker_map[k]
        p = plt.scatter(ind, Cara_cat_polarity_mat_grouped[:, k], alpha=1, marker='.', color='red')
        p = plt.plot(ind, Cara_cat_polarity_mat_grouped[:, k], color='red', linewidth=1.5)
        pp = plt.scatter(ind, Alan_cat_polarity_mat_grouped[:, k], alpha=1, marker='*', color='cyan')
        pp = plt.plot(ind, Alan_cat_polarity_mat_grouped[:, k], color='cyan', linewidth=1.5)
        ppp = plt.scatter(ind, Tim_cat_polarity_mat_grouped[:, k], alpha=1, marker='*', color='purple')
        ppp = plt.plot(ind, Tim_cat_polarity_mat_grouped[:, k], color='purple', linewidth=1.5)
        pppp = plt.scatter(ind, Other_cat_polarity_mat_grouped[:, k], alpha=1, marker='*', color='pink')
        pppp = plt.plot(ind, Other_cat_polarity_mat_grouped[:, k], color='pink', linewidth=1.5)
        plots = [p, pp,ppp, pppp]

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
    names = ['Cara', 'Alan', 'Other1', 'Other2']
    pred = [float(l) for l in open('../Preds.csv').xreadlines()]
    cat_list = ['Finance', 'Statement', 'Energy', 'Business', 'Boss', 'Collaborators', 'Company', 'Health', 'Legal', 'Power scheduling task', 'Stock']
    merged_cat_list = ['Finance+Statement', 'Energy+Business', 'Boss', 'Collaborators', 'Company', 'Health', 'Legal', 'Power scheduling task', 'Stock']
    no_cat = len(cat_list)

    Cara_filename = 'cara_v7.csv'
    Cara_senlist_df = pd.read_csv(Cara_filename)
    Alan_filename = 'alan_82_v4.csv'
    Alan_senlist_df = pd.read_csv(Alan_filename)
    Tim_filename = 'Others2_v4.csv'
    Tim_senlist_df = pd.read_csv(Tim_filename)
    Other_filename = 'other1v6.csv'
    Other_senlist_df = pd.read_csv(Other_filename)

    # For Cara
    Cara_time_dict = defaultdict(lambda:defaultdict(list))
    for ind, date in enumerate(Cara_senlist_df.Date):
        cat = str(Cara_senlist_df.Aspect_category[ind]).strip()
        if cat not in cat_list:
            print cat, ind
            raw_input()
        if GroundTruth_flag:
            ###for ground truth
            p = Cara_senlist_df.AspectCategorySentiment[ind]
        else:
            ###for SVM prediction
            p =pred[ind]
        if Cara_time_dict[date][cat]:
            Cara_time_dict[date][cat].append(p)
        else:
            cat_polarity_dict = {}
            cat_polarity_dict[cat] = [p]
            Cara_time_dict[date].update(cat_polarity_dict)

    # For Alan
    Alan_time_dict = defaultdict(lambda: defaultdict(list))
    for ind, date in enumerate(Alan_senlist_df.Date):
        cat = str(Alan_senlist_df.Aspect_category[ind]).strip()
        if cat not in cat_list:
            print cat, ind
            raw_input()
        if GroundTruth_flag:
            ###for ground truth
            p = Alan_senlist_df.AspectCategorySentiment[ind]
        else:
            ###for SVM prediction
            p = pred[ind]
        if Alan_time_dict[date][cat]:
            Alan_time_dict[date][cat].append(p)
        else:
            cat_polarity_dict = {}
            cat_polarity_dict[cat] = [p]
            Alan_time_dict[date].update(cat_polarity_dict)

        # For Tim
        Tim_time_dict = defaultdict(lambda: defaultdict(list))
        for ind, date in enumerate(Tim_senlist_df.Date):
            cat = str(Tim_senlist_df.Aspect_category[ind]).strip()
            if cat not in cat_list:
                print cat, ind
                raw_input()
            if GroundTruth_flag:
                ###for ground truth
                p = Tim_senlist_df.AspectCategorySentiment[ind]
            else:
                ###for SVM prediction
                p = pred[ind]
            if Tim_time_dict[date][cat]:
                Tim_time_dict[date][cat].append(p)
            else:
                cat_polarity_dict = {}
                cat_polarity_dict[cat] = [p]
                Tim_time_dict[date].update(cat_polarity_dict)


    # For Other
    Other_time_dict = defaultdict(lambda: defaultdict(list))
    for ind, date in enumerate(Other_senlist_df.Date):
        cat = str(Other_senlist_df.Aspect_category[ind]).strip()
        if cat not in cat_list:
            print cat, ind
            raw_input()
        if GroundTruth_flag:
            ###for ground truth
            p = Other_senlist_df.AspectCategorySentiment[ind]
        else:
            ###for SVM prediction
            p = pred[ind]
        if Other_time_dict[date][cat]:
            Other_time_dict[date][cat].append(p)
        else:
            cat_polarity_dict = {}
            cat_polarity_dict[cat] = [p]
            Other_time_dict[date].update(cat_polarity_dict)

    Date_list_unsort = set(list(Cara_senlist_df.Date) + list(Alan_senlist_df.Date)+ list(Tim_senlist_df.Date)+ list(Other_senlist_df.Date))
    Date_list_unsort = list(map(lambda x: str(x), Date_list_unsort))
    Date_list = sorted(Date_list_unsort, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H.%M.%S'))


    ### For Cara
    ## sen: row---> a list of polarity for different cat in order
    Cara_sen_cat_polarity_mat =[]
    for date in Date_list:
        cat_polarity_list = [0 for i in range(no_cat)]
        for cat in Cara_time_dict[date].keys():
            cat_polarity_list[cat_list.index(cat)] = sum(Cara_time_dict[date][cat])/len(Cara_time_dict[date][cat])
        Cara_sen_cat_polarity_mat.append(cat_polarity_list)
    Cara_sen_cat_polarity_mat = np.array(Cara_sen_cat_polarity_mat)

    ### For Alan
    Alan_sen_cat_polarity_mat = []
    for date in Date_list:
        cat_polarity_list = [0 for i in range(no_cat)]
        for cat in Alan_time_dict[date].keys():
            cat_polarity_list[cat_list.index(cat)] = sum(Alan_time_dict[date][cat])/len(Alan_time_dict[date][cat])
        Alan_sen_cat_polarity_mat.append(cat_polarity_list)
    Alan_sen_cat_polarity_mat = np.array(Alan_sen_cat_polarity_mat)

    ### For Tim
    Tim_sen_cat_polarity_mat = []
    for date in Date_list:
        cat_polarity_list = [0 for i in range(no_cat)]
        for cat in Tim_time_dict[date].keys():
            cat_polarity_list[cat_list.index(cat)] = sum(Tim_time_dict[date][cat])/len(Tim_time_dict[date][cat])
        Tim_sen_cat_polarity_mat.append(cat_polarity_list)
    Tim_sen_cat_polarity_mat = np.array(Tim_sen_cat_polarity_mat)

    ### For Other
    Other_sen_cat_polarity_mat = []
    for date in Date_list:
        cat_polarity_list = [0 for i in range(no_cat)]
        for cat in Other_time_dict[date].keys():
            cat_polarity_list[cat_list.index(cat)] = sum(Other_time_dict[date][cat])/len(Other_time_dict[date][cat])
        Other_sen_cat_polarity_mat.append(cat_polarity_list)
    Other_sen_cat_polarity_mat = np.array(Other_sen_cat_polarity_mat)

    month_timeline, timeline = GetMonthTimeline(Date_list)
    Cara_cat_polarity_mat_grouped = GetGroupedCatPolarityMat(month_timeline, no_cat, Cara_sen_cat_polarity_mat)
    Alan_cat_polarity_mat_grouped = GetGroupedCatPolarityMat(month_timeline, no_cat, Alan_sen_cat_polarity_mat)
    Tim_cat_polarity_mat_grouped = GetGroupedCatPolarityMat(month_timeline, no_cat, Tim_sen_cat_polarity_mat)
    Other_cat_polarity_mat_grouped = GetGroupedCatPolarityMat(month_timeline, no_cat, Other_sen_cat_polarity_mat)

    no_group = len(timeline)
    Cara_cat_polarity_mat_grouped = GetMergedCatPolarityMat(merged_cat_list, no_group, Cara_cat_polarity_mat_grouped)
    Alan_cat_polarity_mat_grouped = GetMergedCatPolarityMat(merged_cat_list, no_group, Alan_cat_polarity_mat_grouped)
    Tim_cat_polarity_mat_grouped = GetMergedCatPolarityMat(merged_cat_list, no_group, Tim_cat_polarity_mat_grouped)
    Other_cat_polarity_mat_grouped = GetMergedCatPolarityMat(merged_cat_list, no_group, Other_cat_polarity_mat_grouped)


    Cara_cat_polarity_mat_grouped = movingAvg(Cara_cat_polarity_mat_grouped)
    Alan_cat_polarity_mat_grouped = movingAvg(Alan_cat_polarity_mat_grouped)
    Tim_cat_polarity_mat_grouped = movingAvg(Tim_cat_polarity_mat_grouped)
    Other_cat_polarity_mat_grouped = movingAvg(Other_cat_polarity_mat_grouped)

    PlotCatPolarityScatter(Cara_cat_polarity_mat_grouped, Alan_cat_polarity_mat_grouped, Tim_cat_polarity_mat_grouped, Other_cat_polarity_mat_grouped, timeline, merged_cat_list, names)


if __name__ == '__main__':
    main()













