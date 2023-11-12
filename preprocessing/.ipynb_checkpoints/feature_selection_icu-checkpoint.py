import os
import pickle
import glob
import importlib
import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *


if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")

def feature_icu(cohort_output, version_path):
        print("[EXTRACTING CHART EVENTS DATA]")
        chart=preproc_chart("./"+version_path+"/icu/chartevents.csv.gz",
                            './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None,
                            usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
        chart = drop_wrong_uom(chart, 0.95)
        chart[['stay_id', 'itemid','event_time_from_admit','valuenum']].to_csv(
            "./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

def preprocess_features_icu(cohort_output,clean_chart,
                            impute_outlier_chart,thresh,left_thresh): 
    if clean_chart:   
        print("[PROCESSING CHART EVENTS DATA]")
        chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
        chart = outlier_imputation(chart, 'itemid', 'valuenum', 
                                   thresh,left_thresh,impute_outlier_chart)
        print("Total number of rows",chart.shape[0])
        chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
        
        
        
def generate_summary_icu(chart_flag):
    print("[GENERATING FEATURE SUMMARY]")
        
    chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
    freq=chart.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
    freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

    missing=chart[chart['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
    total=chart.groupby('itemid').size().reset_index(name="total_count")
    summary=pd.merge(missing,total,on='itemid',how='right')
    summary=pd.merge(freq,summary,on='itemid',how='right')

    summary=summary.fillna(0)
    summary.to_csv('./data/summary/chart_summary.csv',index=False)
    summary['itemid'].to_csv('./data/summary/chart_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")
    
def features_selection_icu(cohort_output):
        
    print("[FEATURE SELECTION CHART EVENTS DATA]")
    chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',
                      header=0, index_col=None)
    features=pd.read_csv("./data/summary/chart_features.csv",header=0)
    chart=chart[chart['itemid'].isin(features['itemid'].unique())]
    print("Total number of rows",chart.shape[0])
    chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
    print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")