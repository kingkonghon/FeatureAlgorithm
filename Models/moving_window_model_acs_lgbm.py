# from xgb_training import run
from lgbm_model_accessment_remake import run
import os



if __name__ == '__main__':
    Years = list(range(2012, 2018))
    Seasons = list(range(1,7))
    result_predict_path = '/data2/jianghan/FeatureAlgorithm/lgbm_model_results/predict'
    if not os.path.exists(result_predict_path):
        os.makedirs(result_predict_path)
    for year in Years:
        for season in Seasons:
            # if (year == 2017) and (season == 3):
            if (year == 2014) and (season == 2):
                break
            print('%d_s%d start processing...' % (year, season))
            result_folder_path = '/data2/jianghan/FeatureAlgorithm/lgbm_model_results/%d_s%d' % (year, season)
            if not os.path.exists(result_folder_path):
                os.makedirs(result_folder_path)
            run(year, season, result_folder_path, result_predict_path)

    print('complete!')