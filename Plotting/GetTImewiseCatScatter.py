import os, re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict

def PlotCatPolarityScatter(cat_polarity_mat_grouped, timeline, cat_list, filename):
    N, K = cat_polarity_mat_grouped.shape  # N time slot, K categories
    ind = np.arange(N)  # the x-axis locations for the timeline
    width = 0.1  # the width of the bars
    plots = []
    height_cumulative = np.zeros(N)
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'gray', 'yellow', 'brown', 'olive', 'cyan']
    marker_map = [".", "o", "^", "8", "s", "*", "h", "H", "x", "D", "_"]
    for k in range(K):
        color = color_map[k]
        marker = marker_map[k]
        p = plt.scatter(ind, cat_polarity_mat_grouped[:, k], alpha=1, marker=marker, color=color)
        p = plt.plot(ind, cat_polarity_mat_grouped[:, k], color=color, linewidth=1.5)
        # p = plt.bar(ind, cat_polarity_mat_grouped[:, k], width, bottom=height_cumulative, color=color)
        height_cumulative += cat_polarity_mat_grouped[:, k]
        plots.append(p)

        plt.ylabel('Polarity')
        if GroundTruth_flag:
            plt.title('{} Groundtruth'.format(filename))
        else:
            plt.title('{} SVM Prediction Result'.format(filename))
        plt.xticks(ind + width / 4, timeline, rotation=265)
        plt.yticks(np.arange(0, 1, 1000))
        cat_labels = ['{}'.format(cat) for cat in cat_list]
        leg = plt.legend([p[0] for p in plots], cat_labels, fancybox=True, prop={'size': 9})
        leg.get_frame().set_alpha(0.5)
        plt.show()

def GetDayTimeline(list_df):
    day_timeline = []
    # Need the date from panda
    for f in list_df.Date[:500]:
        time = str(f).split()[0]
        day_timeline.append(time)
    day_timeline = np.asarray(day_timeline)
    timeline = sorted(set(day_timeline), key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    return day_timeline, timeline

def GetMonthTimeline(list_df):
    month_timeline = []
    # Need the date from panda
    for f in list_df.Date[:500]:
        time = str(f).split()[0][:-3]
        month_timeline.append(time)
    month_timeline = np.asarray(month_timeline)
    timeline = sorted(set(month_timeline), key=lambda x: datetime.datetime.strptime(x, '%Y-%m'))
    return month_timeline, timeline


def GetGroupedCatPolarityMat(day_timeline, no_cat,cat_polarity_mat):
    num_groups = len(set(day_timeline))
    cat_polarity_mat_grouped = np.zeros((num_groups, no_cat))

    for i, day in enumerate(sorted(set(day_timeline))):
        cat_polarity_mat_grouped[i, :] = np.mean(cat_polarity_mat[day_timeline == day, :], axis=0)

    return cat_polarity_mat_grouped


GroundTruth_flag = True #False #
def main():

    filename = 'Cara_labelledaspect_sentiments_v3_492_all.csv'
    senlist_df = pd.read_csv(filename)
    pred = [float(l) for l in open('../Preds.csv').xreadlines()]
    # cat_list = ['Buisness', 'Company', 'Data', 'Enron', 'Issue', 'Life', 'Market', 'Money', 'Name', 'Network', 'Place', 'Power', 'Product', 'Work']
    cat_list = ['Boss', 'Business', 'Collaborators', 'Company', 'Energy', 'Finance', 'Health', 'Legal', 'Power scheduling task', 'Statement', 'Stock']
    no_cat = len(cat_list)

    time_dict = defaultdict(lambda:defaultdict(list))
    for ind, date in enumerate(senlist_df.Date[:500]):
        cat = str(senlist_df.Aspect_category[ind]).strip()
        if cat not in cat_list:
            print cat, ind
            raw_input()
        if GroundTruth_flag:
            ###for ground truth
            p = senlist_df.AspectCategorySentiment[ind]
        else:
            ###for SVM prediction
            p =pred[ind]
        if time_dict[date][cat]:
            time_dict[date][cat].append(p)
        else:
            cat_polarity_dict = {}
            cat_polarity_dict[cat] = [p]
            time_dict[date].update(cat_polarity_dict)


    ### sen: row---> a list of polarity for different cat in order
    sen_cat_polarity_mat = []
    for date in senlist_df.Date[:500]:
        cat_polarity_list = [0 for i in range(no_cat)]
        for cat in time_dict[date].keys():
            cat_polarity_list[cat_list.index(cat)] = sum(time_dict[date][cat])/len(time_dict[date][cat])
        sen_cat_polarity_mat.append(cat_polarity_list)
    sen_cat_polarity_mat = np.array(sen_cat_polarity_mat)
    pprint(sen_cat_polarity_mat)
    # sen_cat_polarity_mat.dump('sen_cat_polarity.mat')
    # raw_input()
    # sen_cat_polarity_mat = np.load('sen_cat_polarity.mat')

    day_timeline, timeline = GetDayTimeline(senlist_df)
    cat_polarity_mat_grouped = GetGroupedCatPolarityMat(day_timeline, no_cat, sen_cat_polarity_mat)
    PlotCatPolarityScatter(cat_polarity_mat_grouped, timeline, cat_list, filename)


if __name__ == '__main__':
    main()













