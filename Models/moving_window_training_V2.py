# from xgb_training import run
from xgb_trainingV2 import run
import os



if __name__ == '__main__':
    Years = list(range(2016, 2018))
    Seasons = list(range(1,7))
    result_predict_path = '/home/nyuser/jianghan/FeatureAlgorithm/model_results/prediction'
    for year in Years:
        for season in Seasons:
            if (year == 2017) and (season == 3):
                break
            print('%d_s%d start processing...' % (year, season))
            result_folder_path = '/home/nyuser/jianghan/FeatureAlgorithm/model_results/%d_s%d' % (year, season)
            if not os.path.exists(result_folder_path):
                os.makedirs(result_folder_path)
            run(year, season, result_folder_path, result_predict_path)

    print('complete!')