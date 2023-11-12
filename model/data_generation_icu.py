import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")
if not os.path.exists("./data/csv"):
    os.makedirs("./data/csv")
    
class Generator():
    def __init__(self,cohort_output,impute,include_time=24,bucket=1,predW=6):
        self.cohort_output=cohort_output
        self.impute=impute
        self.data = self.generate_adm()
        print("[ READ COHORT ]")
        
        self.generate_feat()
        print("[ READ ALL FEATURES ]")
        
        self.readmission_length(include_time)
        print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        
        self.smooth_meds(bucket)
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")
    
    def generate_feat(self):
        print("[ ======READING CHART EVENTS ]")
        self.generate_chart()

    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data['intime'] = pd.to_datetime(data['intime'])
        data['outtime'] = pd.to_datetime(data['outtime'])
        data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        data['Age']=data['Age'].astype(int)
        #print(data.head())
        #print(data.shape)
        return data    
        
    def generate_chart(self):
        chunksize = 5000000
        final=pd.DataFrame()
        for chart in tqdm(pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            chart=chart[chart['stay_id'].isin(self.data['stay_id'])]
            chart[['start_days', 'dummy','start_hours']] = chart['event_time_from_admit'].str.split(' ', -1, expand=True)
            chart[['start_hours','min','sec']] = chart['start_hours'].str.split(':', -1, expand=True)
            chart['start_time']=pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
            chart=chart.drop(columns=['start_days', 'dummy','start_hours','min','sec','event_time_from_admit'])
            chart=chart[chart['start_time']>=0]

            ###Remove where event time is after discharge time
            chart=pd.merge(chart,self.data[['stay_id','los']],on='stay_id',how='left')
            chart['sanity']=chart['los']-chart['start_time']
            chart=chart[chart['sanity']>0]
            del chart['sanity']
            del chart['los']
            
            if final.empty:
                final=chart
            else:
                final=final.append(chart, ignore_index=True)
        
        self.chart=final
    
    def mortality_length(self,include_time,predW):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time+predW)]
        self.hids=self.data['stay_id'].unique()
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
            
       ###CHART
        self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
        self.chart=self.chart[self.chart['start_time']<=include_time]
        
        #self.los=include_time
    def los_length(self,include_time):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
            
       ###CHART
        self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
        self.chart=self.chart[self.chart['start_time']<=include_time]
            
    def readmission_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed

            
       ###CHART
        self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
        self.chart=pd.merge(self.chart,self.data[['stay_id','select_time']],on='stay_id',how='left')
        self.chart['start_time']=self.chart['start_time']-self.chart['select_time']
        self.chart=self.chart[self.chart['start_time']>=0]
        
            
    def smooth_meds(self,bucket):
        final_chart=pd.DataFrame()
        
        self.chart=self.chart.sort_values(by=['start_time'])
        
        t=0
        for i in tqdm(range(0,self.los,bucket)): 
                          
            ###CHART
            sub_chart=self.chart[(self.chart['start_time']>=i) & (self.chart['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'valuenum':np.nanmean})
            sub_chart=sub_chart.reset_index()
            sub_chart['start_time']=t
            if final_chart.empty:
                final_chart=sub_chart
            else:    
                final_chart=final_chart.append(sub_chart)
            t=t+1
            
        print("bucket",bucket)
        los=int(self.los/bucket)
            
        ###chart
        f2_chart=final_chart.groupby(['stay_id','itemid']).size()
        self.chart_per_adm=f2_chart.groupby('stay_id').sum().reset_index()[0].max()             
        self.chartlength_per_adm=final_chart.groupby('stay_id').size().max()
        
        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")
        self.create_Dict(final_chart,los)
        
    
    def create_chartDict(self,chart,los):
        dataDic={}
        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Chart':{},'label':int(grp['label'])}
        for hid in tqdm(self.hids):
            ###CHART
            df2=chart[chart['stay_id']==hid]
            val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
            df2['val']=1
            df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
            #print(df2.shape)
            add_indices = pd.Index(range(los)).difference(df2.index)
            add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
            df2=pd.concat([df2, add_df])
            df2=df2.sort_index()
            df2=df2.fillna(0)
            
            val=pd.concat([val, add_df])
            val=val.sort_index()
            if self.impute=='Mean':
                val=val.ffill()
                val=val.bfill()
                val=val.fillna(val.mean())
            elif self.impute=='Median':
                val=val.ffill()
                val=val.bfill()
                val=val.fillna(val.median())
            val=val.fillna(0)
            
            
            df2[df2>0]=1
            df2[df2<0]=0
            #print(df2.head())
            dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
            dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")
            
            
                
        ######SAVE DICTIONARIES##############
        with open("./data/dict/metaDic", 'rb') as fp:
            metaDic=pickle.load(fp)
        
        with open("./data/dict/dataChartDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

      
        with open("./data/dict/chartVocab", 'wb') as fp:
            pickle.dump(list(chart['itemid'].unique()), fp)
        self.chart_vocab = chart['itemid'].nunique()
        metaDic['Chart']=self.chart_per_adm
        
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            
            
    def create_Dict(self,chart,los):
        dataDic={}
        print(los)
        labels_csv=pd.DataFrame(columns=['stay_id','label'])
        labels_csv['stay_id']=pd.Series(self.hids)
        labels_csv['label']=0

        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Chart':{},'ethnicity':grp['ethnicity'].iloc[0], 'age':int(grp['Age']),
                          'gender':grp['gender'].iloc[0],'label':int(grp['label'])}
            labels_csv.loc[labels_csv['stay_id']==hid,'label']=int(grp['label'])
            
            #print(static_csv.head())
        for hid in tqdm(self.hids):
            grp=self.data[self.data['stay_id']==hid]
            demo_csv=grp[['Age','gender','ethnicity','insurance']]
            if not os.path.exists("./data/csv/"+str(hid)):
                os.makedirs("./data/csv/"+str(hid))
            demo_csv.to_csv('./data/csv/'+str(hid)+'/demo.csv',index=False)
            
            dyn_csv=pd.DataFrame()         
                
            ###CHART
            feat=chart['itemid'].unique()
            df2=chart[chart['stay_id']==hid]
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                if self.impute=='Mean':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.mean())
                elif self.impute=='Median':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.median())
                val=val.fillna(0)


                df2[df2>0]=1
                df2[df2<0]=0
                #print(df2.head())
                dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")

                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
            
            #Save temporal data to csv
            dyn_csv.to_csv('./data/csv/'+str(hid)+'/dynamic.csv',index=False)
            grp.to_csv('./data/csv/'+str(hid)+'/static.csv',index=False)   
            labels_csv.to_csv('./data/csv/labels.csv',index=False)    
            
                
        ######SAVE DICTIONARIES##############
        metaDic={'Chart':{},'LOS':{}}
        metaDic['LOS']=los
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)
        
        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()
            
        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(list(self.data['Age'].unique()), fp)
            self.age_vocab = self.data['Age'].nunique()
            
        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()
            
        with open("./data/dict/chartVocab", 'wb') as fp:
            pickle.dump(list(chart['itemid'].unique()), fp)
        self.chart_vocab = chart['itemid'].nunique()
        metaDic['Chart']=self.chart_per_adm
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            
            
      


