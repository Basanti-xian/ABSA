import pandas as pd
import numpy, sys, os

reload(sys)
sys.setdefaultencoding('utf8')

def getCsvForNrc(excel_path, sheet_name=0, NumOfLine=-1, output_name='ForNRCFeats.csv'):
    # df = pd.read_excel(excel_path, sheetname=sheet_name, header=0)
    df = pd.read_csv(excel_path, header=0)

    red_df = df.loc[:NumOfLine, ['Sentence', 'Aspect_category', 'AspectCategorySentiment']]

    red_df[['AspectCategorySentiment']] = red_df[['AspectCategorySentiment']].astype(numpy.int32)

    red_df.to_csv(output_name, sep='^', header=False, index=False)


if __name__ == '__main__':
    # getCsvForNrc('./Plotting/Cara_labelledaspect_sentiments_v3_492_all.csv', NumOfLine=800, output_name='ForNRCFeats(492).csv')
    # getCsvForNrc('./Plotting/Cara_final_labeled_all_cleaned_nov2.csv', NumOfLine=800, output_name='ForNRCFeats.csv')
    f_folder = './Plotting/'
    filenames = ['cara_v7.csv', 'alan_82_v4.csv', 'tim_52_v4.csv', 'Others1_v6.csv', 'Others2_v4.csv']
    names = [f.split('_')[0] for f in filenames]
    fnames = [os.path.join(f_folder, str(file)) for file in filenames]

    Output_fnames = ['ForNRCFeats({})'.format(n) for n in names]

    for ind, f in enumerate(fnames):
        getCsvForNrc(f, NumOfLine=800, output_name=Output_fnames[ind])

