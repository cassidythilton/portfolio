import ast
import base64
import pandas as pd
from datetime import datetime, timedelta, timezone, date
import requests
import itertools
import os
import json
import time
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import dotenv
import warnings
warnings.filterwarnings("ignore")
dotenv.load_dotenv(dotenv.find_dotenv())
import pandas as pd
from pyecharts.charts import Line, Bar
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot as driver
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE=NotebookType.JUPYTER_LAB
from bs4 import BeautifulSoup
import re
from helper import setDataBricksEnvConnects, setDataBricksRunId, getPredictionPolicies, getAllFeatPolicies, setOpenAItoken, getPolicies
import shutil
from bs4 import BeautifulSoup
import html
import openai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from IPython.display import display, HTML


def getModelMetricsMLFlow(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    return metrics

def getChChtrainingMetrics(cluster, token):
    print(f"...obtaining training metrics from Databricks")
    models = ["CHAMPION", "CHALLENGER"]
    trainingMetricsDict = {}

    for mod in models:
        print(f"   ...getting {mod.lower()} metrics")        
        if mod == "CHAMPION":
            isCC = True
        elif mod == "CHALLENGER":
            isCC = False

        setDataBricksEnvConnects(cluster, token)
        run_id = setDataBricksRunId(cluster, token, isCC)
        trainingMetrics = getModelMetricsMLFlow(run_id)

        modifiedMetrics = {mod.lower() + "_" + key: round(value, 4) for key, value in trainingMetrics.items()}    
        trainingMetricsDict.update(modifiedMetrics)
    print(f"...finished obtaining training metrics\n")
        
    return trainingMetricsDict

def remove_carriage_returns(raw_file):
    html_file=f"mrm/{raw_file}.html"
    with open(html_file, 'r') as file:
        html_content = file.read()

    cleaned_html = re.sub(r'\r', '', html_content)

    with open(html_file, 'w') as file:
        file.write(cleaned_html)
        
  
"""
move files from main directory to mrm folder
"""
rootDir = '.'
targetDir = './mrm'
# Ensure that target directory exists
os.makedirs(targetDir, exist_ok=True)

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname.endswith('.html'):
            source = os.path.join(dirName, fname)
            destination = os.path.join(targetDir, fname)
            shutil.move(source, destination)

def get_models(token=None, cluster=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/models/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"page":1,"page_size":10000,"order":[{"field":"name","direction":"ASC"}],"filters":{}}
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    return response.text

def get_model_info(modelname=None, version=1, token=None, cluster=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/model_deployment/getVersionedModel?model_name={modelname}&model_version={version}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {}
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    return response.text

def get_model_perf(modeluuid=None, token=None, cluster=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/model-performance?model_uuid={modeluuid}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {}
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    return response.text

def get_model_perf_dts(modeluuid=None, token=None, cluster=None, start=None, end=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/model-performance?startDateTime={start}&endDateTime={end}&model_uuid={modeluuid}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {}
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    return response.text

def get_segments(modeluuid=None, token=None, cluster=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/segments/search"

    payload = {
    "page": 1,
    "page_size": 10000,
    "filters": {},
    "model_uuids": [
    f"{modeluuid}"
    ],
    "include_policies": True
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    r=response.json()
    data=pd.json_normalize(r['items'],record_path='policies',meta=['id','name'],record_prefix='policy_')
    return data.rename(columns={'name':'segment_name'})

def get_drift_history(token,cluster,modelname, model_version, policy_name=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    payload = {}
    url = f"{WEBSERVICES_URL}/v1/driftdetections?model_name={modelname}&model_version={model_version}"
    if policy_name:
        url = f"{WEBSERVICES_URL}/v1/driftdetections?model_name={modelname}&model_version={model_version}&drift_policy_name={policy_name}"


    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    r=response.json()
    pr=pd.json_normalize(r)
    driftFrame=pr[[
    'results.segment_id',
    'drift_config.name',
    'results.drift_metrics.total_weighted_drift',
    'results.drift_metrics.total_weighted_drift_critical',
    'results.drift_metrics.total_weighted_drift_critical_flag',
    'results.drift_metrics.total_weighted_drift_warning',
    'results.drift_metrics.total_weighted_drift_warning_flag',
    'results.target_count',
    'processed_ts'
    ]].copy()

    driftFrame['processed_ts']=pd.to_datetime(driftFrame['processed_ts'], unit='ms').dt.date
    return driftFrame    


def get_drift_hist(modeluuid=None, token=None, cluster=None, policy_name=None):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    payload = {}
    url = f"{WEBSERVICES_URL}/v1/driftdetections?model_uuid={modeluuid}"
    if policy_name:
        url = f"{WEBSERVICES_URL}/v1/driftdetections?model_uuid={modeluuid}&drift_policy_name={policy_name}"  
        url = re.sub(r'prior policy', 'prior++policy', url)
        url = url.replace(' ', '+')  
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    pr=pd.json_normalize(response.json())
    driftHistory=pr[[ 'results.segment_id','drift_policy_name','results.drift_metrics.total_weighted_drift','results.drift_metrics.total_weighted_drift_critical','results.drift_metrics.total_weighted_drift_critical_flag','results.drift_metrics.total_weighted_drift_warning','results.drift_metrics.total_weighted_drift_warning_flag','results.target_count','processed_ts'
    ]].copy()
    driftHistory['processed_ts']=pd.to_datetime(driftHistory['processed_ts'], unit='ms').dt.date
    return driftHistory

def chat_with_model(messages, model='gpt-3.5-turbo', tokens=150, temperature=0.5):
    OPENAI_API_KEY=setOpenAItoken(cluster, token)
    openai.api_key = OPENAI_API_KEY

    message_objs = [{'role': 'system', 'content': 'You are talking to an AI model.'},
                    {'role': 'user', 'content': 'You are an expert in machine learning model monitoring and I need you to help me understand the performance of my models performance from the most recent ended quarter (Current) compared to the previous ended quarter (Prior). I will provide you with some metrics to give you context and I would like to you to explain to me in 350-500 words what I need to know, how I can address any negative outcomes, or maintain any positive outcomes, and you should note specifically that these details are to be inputted into my MRM documentation in financial services (see SR letter 11-7 from the board of governors of the federal reserve system office of the comptroller of the currency.'},
                    {'role': 'user', 'content': messages},                    
                    ] 

    response = openai.ChatCompletion.create(
        model=model,
        messages=message_objs,
        max_tokens=tokens,
        n=1,
        temperature=temperature
    )
    return response.choices[0].message['content']

def get_model_perf_history(cluster, token, modelname):
    models = json.loads(get_models(token, cluster))
    modeluuid = None
    for item in models['items']:
        if item["name"] == modelname :
            modeluuid = item["uuid"]
            #print(f"Model: {modeluuid}")
    rf=pd.json_normalize(json.loads(get_model_perf(modeluuid, token, cluster)))
    return rf, modeluuid

def get_model_perf_history_cust(cluster, token, modelname,additional_metrics):
    models = json.loads(get_models(token, cluster))
    modeluuid = None
    for item in models['items']:
        if item["name"] == modelname :
            modeluuid = item["uuid"]
            #print(f"Model: {modeluuid}")
    rf=pd.json_normalize(json.loads(get_model_perf(modeluuid, token, cluster)))
    return rf, modeluuid



def first_day_of_prior_quarters():
    today = date.today()
    # Calculate the current calendar quarter
    current_quarter = (today.month - 1) // 3 + 1

    # Calculate the first day of the prior quarter
    if current_quarter == 1:
        prior_quarter_first_day = date(today.year - 1, 10, 1)
    elif current_quarter == 2:
        prior_quarter_first_day = date(today.year, 1, 1)
    elif current_quarter == 3:
        prior_quarter_first_day = date(today.year, 4, 1)
    else:
        prior_quarter_first_day = date(today.year, 7, 1)

    # Calculate the first day of the prior prior quarter
    if current_quarter == 1:
        prior_prior_quarter_first_day = date(today.year - 1, 7, 1)
    elif current_quarter == 2:
        prior_prior_quarter_first_day = date(today.year - 1, 10, 1)
    elif current_quarter == 3:
        prior_prior_quarter_first_day = date(today.year - 1, 1, 1)
    else:
        prior_prior_quarter_first_day = date(today.year, 4, 1)

    return prior_quarter_first_day, prior_prior_quarter_first_day


def last_day_of_prior_quarters():
    today = date.today()
    # Calculate the current calendar quarter
    current_quarter = (today.month - 1) // 3 + 1

    # Calculate the last day of the prior quarter
    if current_quarter == 1:
        prior_quarter_last_day = date(today.year - 1, 12, 31)
    elif current_quarter == 2:
        prior_quarter_last_day = date(today.year, 3, 31)
    elif current_quarter == 3:
        prior_quarter_last_day = date(today.year, 6, 30)
    else:
        prior_quarter_last_day = date(today.year, 9, 30)

    # Calculate the last day of the prior prior quarter
    if current_quarter == 1:
        prior_prior_quarter_last_day = date(today.year - 1, 9, 30)
    elif current_quarter == 2:
        prior_prior_quarter_last_day = date(today.year - 1, 12, 31)
    elif current_quarter == 3:
        prior_prior_quarter_last_day = date(today.year, 3, 31)
    else:
        prior_prior_quarter_last_day = date(today.year, 6, 30)

    return prior_quarter_last_day, prior_prior_quarter_last_day

def load_vars(arg=None, isCC=False):
    base_dir = os.path.join(os.getcwd(), 'src')
    data = {}  # Set data as an empty dict to prevent errors in the next step

    # If a file path argument is provided, move the file to the src directory and load and merge it
    if arg is not None:
        try:
            # Determine new location of the file
            file_name = os.path.basename(arg)
            new_location = os.path.join(base_dir, file_name)
            # Move the file
            shutil.move(arg, new_location)
            # Load and merge the file
            with open(new_location, 'r') as f:
                user_data = json.load(f)
                data.update(user_data)  # Merge user_data into data, overwriting keys if there's any overlap
            print(f"...loaded variables from {arg}")
        except Exception as e:
            default_file = os.path.join(base_dir, "vars.json")
            with open(default_file, 'r') as f:
                data = json.load(f)
            print("...loaded variables from monitoring notebook")                
        except:
            print(f"Warning: encountered error {e} when trying to load user vars from file. No user vars loaded.")

    if isCC:
        with open('src/champion_challenger.json', 'r') as f:
            data = json.load(f)
    sorted_data = {k: data[k] for k in sorted(data)}
            
    return sorted_data


def getMLFlowDescription(cluster, token, run_id=None,isCC=None):
    modelDict={}
    if isCC is not None:
        print(f"...getting model descriptions from Databricks")
        models = ["CHAMPION", "CHALLENGER"]
        for mod in models:
            print(f"   ...obtaining {mod.lower()} model info")
            if mod == "CHAMPION":
                isCC = True
                file_name='description'
            elif mod == "CHALLENGER":
                isCC = False
                file_name='challenger_description'                

            run_id = setDataBricksRunId(cluster, token, isCC)
            lp=mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"description/{file_name}.txt")
            with open(lp, 'r') as f:
                data = f.read()

            ccDict={}
            ccDict['name'] = f'{mod.capitalize()} CCFD Classification Model Databricks'
            ccDict['last_updated_timestamp'] = f"{datetime.today()}"
            ccDict['latest_versions'] = '1'
            ccDict['group name'] = 'Risk Management Analysis'
            ccDict['description'] = data
            ccDict['model id'] = run_id
            modelDict[mod] = ccDict
        flattened_d4 = {}
        for key, value in modelDict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key.lower()}_{inner_key}"
                flattened_d4[new_key] = inner_value
        print(f"...finished assembling model info")

        return flattened_d4

    else:
        run_id = setDataBricksRunId(cluster, token, isCC)
        lp=mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"description/description.txt")
        with open(lp, 'r') as f:
            data = f.read()
        #modelDict=data
        modelDict['name'] = 'CCFD Classification Model Databricks'
        modelDict['last_updated_timestamp'] = f"{datetime.today()}"
        modelDict['latest_versions'] = '1'
        modelDict['group name'] = 'Risk Management Analysis'
        modelDict['description'] = data
        modelDict['model id'] = run_id
        
        return modelDict


def generateMRMmetrics(token, cluster, modelname, model_version,additional_metrics=None):
    rf, modeluuid=get_model_perf_history(cluster, token, modelname,additional_metrics)
    print("...getting current vs. prior quarter dates")
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters() #input 3

    #JOIN SEGMENTS TO POLICY PERFORMANCE DATA
    print("...getting segments")
    segmentData=get_segments(modeluuid, token, cluster).groupby(['id','segment_name']).count()
    segmentData.reset_index(inplace=True)

    #JOIN SEGMENTS TO DRIFT POLICY DATA
    print("...getting drift performance history")    
    dr=get_drift_history(token,cluster,modelname, model_version)
    drSegments=dr.merge(segmentData[['id','segment_name']],left_on='results.segment_id',right_on='id',how='left')
    drSegments=drSegments[['processed_ts','id','drift_config.name','segment_name','results.drift_metrics.total_weighted_drift','results.drift_metrics.total_weighted_drift_critical','results.drift_metrics.total_weighted_drift_critical_flag','results.drift_metrics.total_weighted_drift_warning','results.drift_metrics.total_weighted_drift_warning_flag']].rename(columns={'id':'segment_id','drift_config.name':'policy_name'})
    drSegments['segment_name']=drSegments['segment_name'].fillna('All Data')

    #Results for most recent quarter
    drCurrent=drSegments[#(drSegments['policy_name']=='All Data') &
    (drSegments['processed_ts'].astype(str)==str(prior_quarter_last_day))].groupby(['processed_ts','policy_name','segment_name']).max().sort_values(by=['policy_name','segment_name']).reset_index()

    #Results for prior quarter
    drPrevious=drSegments[#(drSegments['policy_name']=='All Data') &
    (drSegments['processed_ts'].astype(str)==str(prior_prior_quarter_last_day))].groupby(['processed_ts','policy_name','segment_name']).max().sort_values(by=['policy_name','segment_name']).reset_index()

    curPols = drCurrent['policy_name'].unique().tolist()
    priorPols = drPrevious['policy_name'].unique().tolist()
    policies=list(set(curPols) & set(priorPols))

    data2 = {}
    import math
    for policy in policies:
        dpCf = drCurrent[(drCurrent['policy_name']==policy)].dropna()
        dpPf = drPrevious[(drPrevious['policy_name']==policy)].dropna()
        segments = dpPf['segment_name'].unique().tolist()
        for n, segment in enumerate(segments):

            dsgCf = dpCf[(dpCf['segment_name']==segment)]
            current_val = round(dsgCf['results.drift_metrics.total_weighted_drift'].iloc[0],4)

            dsgPf = dpPf[(dpPf['segment_name']==segment)]
            prior_val = round(dsgPf['results.drift_metrics.total_weighted_drift'].iloc[0],4)

            if current_val <= 0 and prior_val<= 0:
                delta = 0
            else:
                delta = round(((current_val - prior_val) / prior_val) * 100, 4)

            delta = round(((current_val - prior_val) / prior_val) * 100, 4)
            data2[f"policy_seg_{n+1}_begin"] = f"{current_val}"
            data2[f"policy_seg_{n+1}_end"] = f"{prior_val}"
            data2[f"policy_seg_{n+1}_delta"] = f"{delta}"

    #Results for most recent quarter
    rfCurrent=rf[(rf['segment_id'].isna()) &
    (rf['formatted_predict_date']==str(prior_quarter_last_day))]

    #Results for prior quarter
    rfPrevious=rf[(rf['segment_id'].isna()) &
    (rf['formatted_predict_date']==str(prior_prior_quarter_last_day))]

    print("...blending and preparing final metrics")
    #Metrics Data - Current vs. Prior
    metrics_columns = ['metrics.False.accuracy','metrics.False.balanced_accuracy','metrics.False.f1','metrics.False.recall','metrics.False.specificity','metrics.False.precision']

    metrics_data = {}
    data = {}
    r=[]
    for column in metrics_columns:
        current_val = round(rfCurrent[column].iloc[0], 5)
        prior_val = round(rfPrevious[column].iloc[0], 5)
        delta = round(((current_val - prior_val) / prior_val) * 100, 4)

        metric_name = column.split('.')[-1]  # extract the metric name from the column name
        metrics_data[metric_name] = {
            'current_val': current_val,
            'prior_val': prior_val,
            'delta': delta
        }

        data[f"{metric_name}_begin"] = f"{current_val}"
        data[f"{metric_name}_end"] = f"{prior_val}"
        data[f"{metric_name}_delta"] = f"{delta}"
        detail=f"{metric_name.capitalize()}: Current = {current_val:.5f}, Prior = {prior_val:.5f}, Delta = {delta:.2f}%"
        r.append(detail)

    print("...getting model description and metadata from MLFlow in DataBricks")
    setDataBricksEnvConnects(cluster, token)
    modelDict = getMLFlowDescription(cluster, token)
    retry_count=0
    while retry_count < 3:
        try:
            modelDict = getMLFlowDescription(cluster, token)
            break  
        except Exception:
            print("An exception occurred. Retrying...")
            time.sleep(15)
            retry_count += 1
    else:
        print("Could not obtain MLFlow description after multiple retries.")

    data['image']='bar_chart.png'
    data['image2']='bar_chart_segs.png'    
    return data, data2, modelDict


def generateMRMPlots(token,cluster,model_name,model_version,policy_name,theme='light'):
    secondary_colors=['#ffffff','#dddddd','#4c4c4c']
    if theme == 'dark':
        secondary_colors=['#000000','#f9f9f9','#f9f9f9']
        
    payload = {}
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/driftdetections?model_name={model_name}&model_version={model_version}&drift_policy_name={policy_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    r=pd.json_normalize(response.json())
    driftResults=r[['results.target_count','results.target_window.end_date','results.drift_metrics.total_weighted_drift','results.critical_level','results.warning_level']].dropna()
    df=driftResults.copy()
    driftResults.groupby('results.target_window.end_date').max().reset_index()
    print(f"...generating plots")    
    #generate plot
    colors= ["#8419B2","#DA94FF","#085B70","#8F6B01","#A6181D"]
    df = driftResults.groupby('results.target_window.end_date').max().reset_index()

    bar_chart = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="600px", bg_color=secondary_colors[0]))
        .add_xaxis(df['results.target_window.end_date'].to_list())
        .add_yaxis("Predictions", df['results.target_count'].to_list(),
                   label_opts=opts.LabelOpts(is_show=False),
                   yaxis_index=1,  # Use the extended y-axis
                   itemstyle_opts=opts.ItemStyleOpts(color=secondary_colors[1]))  
        .extend_axis(
            yaxis=opts.AxisOpts(
                type_="value",
                name="Predictions",
                position="right",
                name_location="middle",
                axisline_opts=opts.AxisLineOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(formatter="{value}", color=secondary_colors[1])

            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Overall Feature Drift"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(save_as_image={})),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],

            legend_opts=opts.LegendOpts(pos_top="top",
                                        pos_left="center",
                                        orient="horizontal",
                                        item_width=10,
                                        item_height=8,
                                        textstyle_opts=opts.TextStyleOpts(color=secondary_colors[2]) 
                                            


                                        ),  
            yaxis_opts=opts.AxisOpts(
                name="PSI",
                name_location='middle',  
                name_gap=45,  
            axislabel_opts=opts.LabelOpts(formatter="{value}", color=secondary_colors[1])
                
            )              

        )
    )

    line_chart = (
        Line()
        .add_xaxis(df['results.target_window.end_date'].to_list())
        .add_yaxis("PSI", df['results.drift_metrics.total_weighted_drift'].to_list(),

                   linestyle_opts=opts.LineStyleOpts(color=colors[0], width=3),
                   itemstyle_opts=opts.ItemStyleOpts(color=colors[0]),
                   symbol="emptycircle",
                    symbol_size=8,
                   label_opts=opts.LabelOpts(is_show=False)
                   )
    )

    combined_chart = bar_chart.overlap(line_chart)
    combined_chart.options.get("series")[1].update(zlevel=100)  # Set z_level for line chart
    temp_dir = os.path.join(os.getcwd(), 'mrm') 
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
        
    make_snapshot(driver, combined_chart.render(), f"{temp_dir}/bar_chart.png", pixel_ratio=1)
    time.sleep(1)
    file_name = "render.html"
    if os.path.exists(file_name):
        shutil.move(f'{file_name}', f'mrm/{file_name}')
    
def insert_image_grid(soup, img_src_list, img_width="100px", img_height="auto"):
    grid_html = """
    <style>
    .grid-container {{
      display: grid;
      grid-template-columns: auto auto auto; 
      padding: 10px;
    }}

    .grid-item img {{
      width: {img_width};
      height: {img_height};
      object-fit: cover;
    }}
    </style>

    <div class="grid-container">
    """.format(img_width=img_width, img_height=img_height)

    for img_src in img_src_list:
        grid_html += f'<div class="grid-item"><img src="{img_src}" alt="Image"></div>'

    grid_html += '</div>'
    
    # Look for the {images} tag in the document and replace it with the grid
    images_tag = '{images_grid}'
    grid_bs = BeautifulSoup(grid_html, 'html.parser')
    
    for text in soup.find_all(text=re.compile(re.escape(images_tag))):
        new_content = re.sub(re.escape(images_tag), str(grid_bs), text)
        text.replace_with(BeautifulSoup(new_content, 'html.parser'))
        
def mrm_html_replace_and_highlight(soup, key, value, color):
    reg_expr = '{' + key + '}'
    for text in soup.find_all(text=re.compile(reg_expr)):
#        if key == 'image':
        if 'image' in key:            
            # If the key is 'image', replace with an img tag instead
            img_html = f'<img src="{value}" alt="Image">'
        else:
            highlighted_html = f'<span style="background-color: {color};">{value}</span>'
            img_html = highlighted_html
        new_content = re.sub(reg_expr, img_html, text)
        text.replace_with(BeautifulSoup(new_content, 'html.parser'))    

from bs4 import BeautifulSoup
import re

def process_html(soup, img_src_list=None, key=None, value=None, color=None, img_width="450px", img_height="auto"):
    # If img_src_list is provided, insert the image grid
    if img_src_list is not None:
        grid_html = """
        <style>
        .grid-container {{
          display: grid;
          grid-template-columns: auto auto auto; 
          grid-gap: 0;  # remove the space between grid items
        }}

        .grid-item {{
          margin: 0;  # set the margins of the grid items to zero
        }}

        .grid-item img {{
          width: {img_width};
          height: {img_height};
          object-fit: cover;
        }}
        </style>

        <div class="grid-container">
        """.format(img_width=img_width, img_height=img_height)

        for img_src in img_src_list:
            grid_html += f'<div class="grid-item"><img src="{img_src}" alt="Image"></div>'

        grid_html += '</div>'
        
        # Look for the {images} tag in the document and replace it with the grid
        images_tag = '{images_grid}'
        grid_bs = BeautifulSoup(grid_html, 'html.parser')
        
        for text in soup.find_all(text=re.compile(re.escape(images_tag))):
            new_content = re.sub(re.escape(images_tag), str(grid_bs), text)
            text.replace_with(BeautifulSoup(new_content, 'html.parser'))
        
    # If key and value are provided, replace and highlight
    if key is not None and value is not None:
        reg_expr = '{' + key + '}'
        for text in soup.find_all(text=re.compile(reg_expr)):
            if 'image' in key:            
                # If the key is 'image', replace with an img tag instead
                img_html = f'<img src="{value}" alt="Image">'
            else:
                highlighted_html = f'<span style="background-color: {color};">{value}</span>'
                img_html = highlighted_html
            new_content = re.sub(reg_expr, img_html, text)
            text.replace_with(BeautifulSoup(new_content, 'html.parser'))

            
def generateMRM(token,cluster,model_name,model_version,policy_name,data,data2,modelDict,trainingDict,files,isCC):
    generateMRMPlots(token
                ,cluster
                ,model_name
                ,model_version
                ,policy_name
                )    
    generateMRMPlotsAndSegments(token
                ,cluster
                ,model_name
                ,model_version
                ,policy_name
                )    
    if len(files)>1:
        repNum='reports'
    else:
        repNum='report'
    print(f"...generating {len(files)} {repNum}:")
    for n, file in enumerate(files):
        print(f"     {n+1}. {file}")
    print("")
    
    for i, file_name in enumerate(files):
        print(f"  ...loading template {i+1}: {file_name}.html")    
        temp_dir = os.path.join(os.getcwd(), 'mrm') 
        with open(f'{temp_dir}/{file_name}.html', 'r', encoding='ISO-8859-1') as f:
            html_doc = f.read()
        print("  ...automating metrics into template")
        soup = BeautifulSoup(html_doc, 'html.parser')
        print("  ...automating plots into template")
        for key, value in data.items(): #model performance 
            mrm_html_replace_and_highlight(soup, key, value, '#fdfe00')

        for key, value in data2.items(): #drift, segments, etc. 
            mrm_html_replace_and_highlight(soup, key, value, '#ff9a00')    

        for key, value in modelDict.items(): #databricks' mlflow
            mrm_html_replace_and_highlight(soup, key, value, '#00ff05')    
            
        for key, value in trainingDict.items(): #databricks training
            mrm_html_replace_and_highlight(soup, key, value, '#01c5ff')                
        print("  ...finalizing html")
        final_html = str(soup)

        with open(f'{temp_dir}/{file_name}_output.html', 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"  ...saving file {file_name}_output.html\n")

        soup = BeautifulSoup(final_html, 'html.parser')
        pretty_html = soup.prettify()
#        display(HTML(pretty_html))        


    
def displayAdditionalMetrics(displaymetrics=False):
    if displaymetrics:
            return {
     "metrics.False.accuracy" : "Accuracy Score - The proportion of correct predictions out of the total predictions made. This is a common evaluation metric for classification problems, but not always the best one, especially when the classes are imbalanced."
    ,"metrics.False.balanced_accuracy" : "Balanced Accuracy Score - The average of recall obtained on each class. This metric is useful for dealing with imbalanced datasets."
    ,"metrics.False.bm" : "Bookmaker Informedness - Also known as Youden's J statistic, it is a single statistic that captures the performance of a diagnostic test. Its value ranges from -1 to 1, with 0 denoting a test performs no better than chance, and 1 denoting a test with perfect discrimination."
    ,"metrics.False.dor" : "Diagnostic Odds Ratio - A measure of the effectiveness of a diagnostic test. It is defined as the ratio of the odds of the test being positive if the subject has a disease relative to the odds of the test being positive if the subject does not have the disease."
    ,"metrics.False.f1" : "F1 Score - The harmonic mean of precision and recall. It is particularly useful in scenarios where either false positives or false negatives are significantly more undesirable than the other."
    ,"metrics.False.fdr" : "False Discovery Rate - The expected proportion of false positives among all positives. It is a method of conceptualizing the rate of type I errors in null hypothesis testing when conducting multiple comparisons."
    ,"metrics.False.fm" : "Fowlkes–Mallows Index - The geometric mean of precision and recall. It is a measure of the effectiveness of a classification model in statistical analysis."
    ,"metrics.False.fn" : "False Negative - The number of instances that were actually positive but were predicted as negative. A lower value is typically desired for this metric."
    ,"metrics.False.fnr" : "False Negative Rate - The ratio of false negatives to the total number of actual positives. This is also known as 'miss rate'."
    ,"metrics.False.for" : "False Omission Rate - The ratio of false negatives to the total number of predicted negatives. This metric is particularly important when the cost of false negatives is high."
    ,"metrics.False.fp" : "False Positive - The number of instances that were actually negative but were predicted as positive. Minimizing this value is especially important when the cost of false alarms is high."
    ,"metrics.False.fpr" : "False Positive Rate - The ratio of false positives to the total number of actual negatives. This is also known as 'fall-out'."
    ,"metrics.False.mcc" : "Matthews Correlation Coefficient - A measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes."
    ,"metrics.False.mk" : "Markedness or deltaP - The difference between the true positive rate and the false positive rate. It provides a measure of how well a model can discriminate between positive and negative instances."
    ,"metrics.False.nlr" : "Negative Learn Rate - Refers to a setting where the learning rate for training is set to negative values. Interestingly, it has been observed that not only does the model performance increase by decreasing the learning rate to zero, but it can be improved even further by decreasing the learning rate into negative territory."
    ,"metrics.False.npv" : "Negative Predictive Value - The proportion of actual negatives among those predicted as negative. It's a measure of the reliability of a negative prediction."
    ,"metrics.False.plr" : "Positive Likelihood Ratio - The ratio of the true positive rate to the false positive rate. A high PLR is indicative of a good diagnostic test."
    ,"metrics.False.precision" : "Precision - Also known as the positive predictive value, it is the proportion of true positives among all positive predictions. It's a measure of the reliability of a positive prediction."
    ,"metrics.False.pt" : "Prevalence Threshold - The probability above which a condition or attribute is more likely to be present than absent, and vice versa. It is often used in decision making and diagnosis."
    ,"metrics.False.recall" : "Recall - Also known as sensitivity or true positive rate, it is the proportion of true positives among all actual positives. It's a measure of the model's ability to find all the positive examples."
    ,"metrics.False.rnp" : "Rate of Negative Predictions - The ratio of the number of negative predictions to the total number of instances. It gives an idea of how conservative the model is in predicting negatives."
    ,"metrics.False.specificity" : "Specificity - Also known as the true negative rate, it is the proportion of true negatives among all actual negatives. It's a measure of the model's ability to correctly identify negative examples."
    ,"metrics.False.tn" : "True Negative - The number of instances that were actually negative and were also predicted as negative. High true negatives are desired when the cost of false positives is high."
    ,"metrics.False.tp" : "True Positive - The number of instances that were actually positive and were also predicted as positive. High true positives are desired when the cost of false negatives is high."
    ,"metrics.False.ts" : "Threat Score - Also known as the critical success index, it measures the accuracy of a forecast when a particular event has occurred. It is the number of correct predictions of the event, divided by the total number of predictions of the event."
}

        
# change to 'mrm metrics calculation' because were going to add drift metrics as well
def generateMRMmetricsCustom(token, cluster, modelname, model_version, additional_metrics=None, current_date=None, previous_date=None, start_date=None, end_date=None):

    rf, modeluuid=get_model_perf_history_cust(cluster, token, modelname,additional_metrics)
    #print(modeluuid)
    print("...getting current vs. prior quarter dates")
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters() #input 3

    #JOIN SEGMENTS TO POLICY PERFORMANCE DATA
    print("...getting segments")
    segmentData=get_segments(modeluuid, token, cluster).groupby(['id','segment_name']).count()
    segmentData.reset_index(inplace=True)

    #JOIN SEGMENTS TO DRIFT POLICY DATA
    print("...getting drift performance history")    
    dr=get_drift_history(token,cluster,modelname, model_version)
    drSegments=dr.merge(segmentData[['id','segment_name']],left_on='results.segment_id',right_on='id',how='left')
    drSegments=drSegments[['processed_ts','id','drift_config.name','segment_name','results.drift_metrics.total_weighted_drift','results.drift_metrics.total_weighted_drift_critical','results.drift_metrics.total_weighted_drift_critical_flag','results.drift_metrics.total_weighted_drift_warning','results.drift_metrics.total_weighted_drift_warning_flag']].rename(columns={'id':'segment_id','drift_config.name':'policy_name'})
    drSegments['segment_name']=drSegments['segment_name'].fillna('All Data')
    
    #Results for most recent quarter
    drCurrent=drSegments[#(drSegments['policy_name']=='All Data') &
    (drSegments['processed_ts'].astype(str)==str(prior_quarter_last_day))].groupby(['processed_ts','policy_name','segment_name']).max().sort_values(by=['policy_name','segment_name']).reset_index()
    #Results for prior quarter
    drPrevious=drSegments[#(drSegments['policy_name']=='All Data') &
    (drSegments['processed_ts'].astype(str)==str(prior_prior_quarter_last_day))].groupby(['processed_ts','policy_name','segment_name']).max().sort_values(by=['policy_name','segment_name']).reset_index()
    
    #print(drPrevious['segment_name'].unique().tolist())
    curPols = drCurrent['policy_name'].unique().tolist()
    priorPols = drPrevious['policy_name'].unique().tolist()
    policies=list(set(curPols) & set(priorPols))
    data2 = {}
    import math
    for policy in policies:
        dpCf = drCurrent[(drCurrent['policy_name']==policy)]#.dropna()
        dpPf = drPrevious[(drPrevious['policy_name']==policy)]#.dropna()
        segments = dpCf['segment_name'].unique().tolist()

        for n, segment in enumerate(segments):
            #print(f"...generating {segment} data")
            dsgCf = dpCf[(dpCf['segment_name']==segment)]
            current_val = round(dsgCf['results.drift_metrics.total_weighted_drift'].iloc[0],4)

            dsgPf = dpPf[(dpPf['segment_name']==segment)]
            prior_val = round(dsgPf['results.drift_metrics.total_weighted_drift'].iloc[0],4)

            if current_val <= 0 and prior_val<= 0:
                delta = 0
            else:
                delta = round(((current_val - prior_val) / prior_val) * 100, 4)

            delta = round(((current_val - prior_val) / prior_val) * 100, 4)
            data2[f"policy_seg_{n}_begin"] = f"{current_val}"
            data2[f"policy_seg_{n}_end"] = f"{prior_val}"
            data2[f"policy_seg_{n}_delta"] = f"{delta}"

    #Accomodate Custom Dates
    if current_date is None:
        current_date = str(prior_quarter_last_day)
    if previous_date is None:
        previous_date = str(prior_prior_quarter_last_day)

    # Results for current date
    rfCurrent = rf[(rf['segment_id'].isna()) 
        & (rf['formatted_predict_date']==current_date)]
    # Results for previous date
    rfPrevious = rf[(rf['segment_id'].isna()) 
        & (rf['formatted_predict_date']==previous_date)]


    print("...blending and preparing final metrics")
    #Metrics Data - Current vs. Prior
    metrics_columns = ['metrics.False.accuracy','metrics.False.balanced_accuracy','metrics.False.f1','metrics.False.recall','metrics.False.specificity','metrics.False.precision']

    # Extend the existing metrics list with the additional metrics
    if additional_metrics is not None:
        for metric in additional_metrics:
            try:
                if metric in rfCurrent.columns:
                    metrics_columns.append(metric)
                    print(f"...added additional metric: \033[1m\033[93m{metric}\033[0m")
                else:
                    #metric = "performance"
                    print(f"\033[91mWARNING:\033[0m Metric {metric} is not in the data")
            except Exception as e:
                print(f"Error: {e}")

    
    metrics_data = {}
    data = {}
    data['current_date']=current_date
    data['previous_date']=previous_date    
    r=[]
    for column in metrics_columns:
        current_val = round(rfCurrent[column].iloc[0], 5)
        prior_val = round(rfPrevious[column].iloc[0], 5)
        delta = round(((current_val - prior_val) / prior_val) * 100, 4)

        metric_name = column.split('.')[-1]  # extract the metric name from the column name
        metrics_data[metric_name] = {
            'current_val': current_val,
            'prior_val': prior_val,
            'delta': delta
        }

        data[f"{metric_name}_begin"] = f"{current_val}"
        data[f"{metric_name}_end"] = f"{prior_val}"
        data[f"{metric_name}_delta"] = f"{delta}"
        detail=f"{metric_name.capitalize()}: Current = {current_val:.5f}, Prior = {prior_val:.5f}, Delta = {delta:.2f}%"
        r.append(detail)

    # Now, let's deal with the additional date range, if it was specified
    if start_date is not None and end_date is not None:
        data['start_date']=start_date
        data['end_date']=end_date
        print(f"...added custom date range: \033[1m\033[93m{start_date}\033[0m to \033[1m\033[93m{end_date}\033[0m")
       
        # Results for current date
        rfCurrent = rf[(rf['segment_id'].isna()) 
            & (rf['formatted_predict_date']==start_date)]
        # Results for previous date
        rfPrevious = rf[(rf['segment_id'].isna()) 
            & (rf['formatted_predict_date']==end_date)]
    
        for column in metrics_columns:
            current_val = round(rfCurrent[column].iloc[0], 5)
            prior_val = round(rfPrevious[column].iloc[0], 5)
            delta = round(((current_val - prior_val) / prior_val) * 100, 4)
            metric_name = column.split('.')[-1]  # extract the metric name from the column name
            #print(f"new metric name: {metric_name}")
            # Append the date range results to the dictionaries and list
            data[f"{metric_name}_begin_date_range"] = f"{current_val}"
            data[f"{metric_name}_end_date_range"] = f"{prior_val}"
            data[f"{metric_name}_delta_date_range"] = f"{delta}"
            
            detail=f"{metric_name.capitalize()}: Current = {current_val:.5f}, Prior = {prior_val:.5f}, Delta = {delta:.2f}%"
            r.append(detail)


    print("...getting model description and metadata from MLFlow in DataBricks")
    setDataBricksEnvConnects(cluster, token)
    run_id=setDataBricksRunId(cluster, token)
    trainingMetricsDict=getModelMetricsMLFlow(run_id)
#    modelDict = getDatabricksModelDescription(cluster,token)
    retry_count=0
    while retry_count < 3:
        try:
            modelDict = getDatabricksModelDescription(cluster,token)
            break  
        except Exception:
            print("An exception occurred. Retrying...")
            time.sleep(15)
            retry_count += 1
    else:
        print("Could not obtain MLFlow description after multiple retries.")
    print("...complete")

    data['image']='bar_chart.png' #plot image
    data['image2']='bar_chart_segs.png' #plot image

    return data, data2, modelDict, trainingMetricsDict



def getDatabricksModelDescription(cluster, token):
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/v_config/getValue?system=acme&module=databricks&name=connection_info"

    payload={}
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}

    response = requests.request("GET", url, headers=headers, data=payload)

    r=response.json()
    hostname=json.loads(r['value'])['hostname']
    dbtoken=json.loads(r['value'])['token']

    headers = {"Authorization": f"Bearer {dbtoken}"}

    # Create the path to your file on DBFS
    dbfs_path = "/FileStore/description.txt"

    # Use the Databricks REST API to read the file
    url = f"{hostname}/api/2.0/dbfs/read?path={dbfs_path}"
    response = requests.get(url, headers=headers)
    #print(response.text)
    # The file content is returned as base64, so we need to decode it
    file_content = base64.b64decode(response.json()['data']).decode('utf-8')
    # Now you can load the content as JSON
    modelDict = json.loads(file_content)
    
    run_id=setDataBricksRunId(cluster, token)
    client = MlflowClient()
    run = client.get_run(run_id)
    tags = run.data.tags

#    modelDict = {}
    modelDict['name'] = 'CCFD Classification Model Databricks'
    modelDict['last_updated_timestamp'] = f"{datetime.today()}"
    modelDict['latest_versions'] = '1'
    modelDict['group name'] = 'Risk Management Analysis'
    modelDict['model id'] = run_id

    description = modelDict["description"]
    return modelDict


def generateMRMPlotsAndSegments(token,cluster,model_name,model_version,policy_name,theme='light'):
    rf, modeluuid=get_model_perf_history(cluster, token, model_name)
    secondary_colors=['#ffffff','#dddddd','#4c4c4c']
    if theme == 'dark':
        secondary_colors=['#000000','#f9f9f9','#f9f9f9']
        
    #print(f"...getting historical metrics for {policy_name}")
    payload = {}
    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/driftdetections?model_name={model_name}&model_version={model_version}&drift_policy_name={policy_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
    rj=pd.json_normalize(response.json())
    driftResults=rj[['results.target_count','results.target_window.end_date','results.drift_metrics.total_weighted_drift','results.critical_level','results.warning_level','segment_id']]
    #df=driftResults.copy()
    #driftResults.groupby('results.target_window.end_date').max().reset_index()

    WEBSERVICES_URL = f"https://webservices.{cluster}"
    url = f"{WEBSERVICES_URL}/v1/segments/search"

    payload = {
    "page": 1,
    "page_size": 10000,
    "filters": {},
    "model_uuids": [
    f"{modeluuid}"
    ],
    "include_policies": True
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    r=response.json()
    segmentData=pd.json_normalize(r['items'],record_path='policies',meta=['id','name'],record_prefix='policy_')
    segmentData.rename(columns={'name':'segment_name'},inplace=True)
    drSegments=driftResults.merge(segmentData[['id','segment_name']],left_on='segment_id',right_on='id',how='left')   
    drSegments['segment_name']=drSegments['segment_name'].fillna('All Data')
    #generate plot
    colors= ["#8419B2","#DA94FF","#085B70","#8F6B01","#A6181D"]
    dfB=drSegments[(drSegments['segment_name']=='All Data')].reset_index(drop=True).groupby('results.target_window.end_date').max().reset_index()    
    datesList=dfB['results.target_window.end_date'].to_list()    
    segments=drSegments['segment_name'].unique().tolist()
    print("  ...generating bar plot for All Data")
    bar_chart = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="600px", bg_color=secondary_colors[0]))
        .add_xaxis(dfB['results.target_window.end_date'].to_list())
        .add_yaxis("Predictions", dfB['results.target_count'].to_list(),
                   label_opts=opts.LabelOpts(is_show=False),
                   yaxis_index=1,  # Use the extended y-axis
                   itemstyle_opts=opts.ItemStyleOpts(color=secondary_colors[1]))  
        .extend_axis(
            yaxis=opts.AxisOpts(
                type_="value",
                name="Predictions",
                position="right",
                name_location="middle",
                axisline_opts=opts.AxisLineOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(formatter="{value}", color=secondary_colors[1])

            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Overall Feature Drift by Segment"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(save_as_image={})),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],

            legend_opts=opts.LegendOpts(pos_top="top",
                                        pos_left="center",
                                        orient="horizontal",
                                        item_width=10,
                                        item_height=8,
                                        textstyle_opts=opts.TextStyleOpts(color=secondary_colors[2]) 
                                            


                                        ),  
            yaxis_opts=opts.AxisOpts(
                name="PSI",
                name_location='middle',  
                name_gap=45,  
            axislabel_opts=opts.LabelOpts(formatter="{value}", color=secondary_colors[1])
                
            )              

        )
    )
    
    line_chart = Line().add_xaxis(datesList)
    for i, seg in enumerate(segments):
        print(f"  ...generating line plot for {seg}")
        dSeg = drSegments[(drSegments['segment_name']==seg)].reset_index(drop=True).groupby('results.target_window.end_date').max().reset_index()
        line_chart.add_yaxis(f"{seg}", dSeg['results.drift_metrics.total_weighted_drift'].to_list(),
                   linestyle_opts=opts.LineStyleOpts(color=colors[i], width=2),
                   itemstyle_opts=opts.ItemStyleOpts(color=colors[i]),
                   symbol="emptycircle",
                    symbol_size=8,
                   label_opts=opts.LabelOpts(is_show=False)
                   )

    combined_chart = bar_chart.overlap(line_chart)
    for i, seg in enumerate(segments):
        combined_chart.options.get("series")[i+1].update(zlevel=(i+1000))  # Set z_level for line chart(s)
    temp_dir = os.path.join(os.getcwd(), 'mrm') 
        
    make_snapshot(driver, combined_chart.render(), f"{temp_dir}/bar_chart_segs.png", pixel_ratio=1)
    time.sleep(1)
    file_name = "render.html"
    if os.path.exists(file_name):
        os.rename(file_name, f"{file_name.replace('.html', '')}_segs.html")
        shutil.move(f"{file_name.replace('.html', '')}_segs.html", f"img/{file_name.replace('.html', '')}_segs.html")
    print("...complete\n")

    
def generateMRMperfComparisonPlots(cluster,token,vars_dict):
    model_names=["CHAMPION","CHALLENGER"]

    dataAll = pd.DataFrame()

    prior_quarter_first_day, prior_prior_quarter_first_day = first_day_of_prior_quarters()
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters()
    
    temp_dir = os.path.join(os.getcwd(), 'mrm')            
    img_dir = os.path.join(os.getcwd(), 'img') 
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
        
    for model in model_names:
        model_name=vars_dict[model]['model_name']
        data,modeluuid=get_model_perf_history(cluster, token, model_name)
        data['RowType']=model
        dataAll = pd.concat([dataAll, data]).reset_index(drop=True)

    columns=["metrics.False.accuracy","metrics.False.balanced_accuracy","metrics.False.f1","metrics.False.recall","metrics.False.specificity","metrics.False.precision"]
    col_names=["Accuracy","Balanced Accuracy","F1","Recall","Specificity","Precision"]
    bar_colors=['#75408C', '#C979EC', '#3E6389', '#77B1EC']
    colors= ["#79EC7D","#ECC979","#45C1AF","#831BB4"]
    l=[]
    for i, col in enumerate(columns):
        l.append(col)
        print(f"...generating {col_names[i]} plot                 ",end='\r',flush=True)
        cNresultsDf=dataAll[['formatted_predict_date','num_ground_truths','num_inferences',"metrics.False.accuracy","metrics.False.balanced_accuracy","metrics.False.f1","metrics.False.recall","metrics.False.specificity","metrics.False.precision","RowType"]][(dataAll['formatted_predict_date']>= str(prior_prior_quarter_first_day))&(dataAll['formatted_predict_date']<= str(prior_quarter_last_day))]

        """
        champion
        """
        dfCh=cNresultsDf[(cNresultsDf['RowType']=='CHAMPION')].groupby('formatted_predict_date').min().round(4).reset_index()
        dfCh.drop(columns=['RowType'],inplace=True)
        """
        challenger
        """
        dfCl=cNresultsDf[(cNresultsDf['RowType']=='CHALLENGER')].groupby('formatted_predict_date').min().round(4).reset_index()
        dfCl.drop(columns=['RowType'],inplace=True)


        bar_chart = (
            Bar(init_opts=opts.InitOpts(width="1600px", height="600px"))
            .add_xaxis(dfCh['formatted_predict_date'].to_list())
            .add_yaxis("Champion Inferences", dfCh['num_inferences'].to_list(),
                       label_opts=opts.LabelOpts(is_show=False),
                       yaxis_index=1,  # Use the extended y-axis
                       itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[0]),
                       gap="0%"
                       )
            .add_yaxis("Champion Ground Truths", dfCh['num_ground_truths'].to_list(),
                       label_opts=opts.LabelOpts(is_show=False),
                       yaxis_index=1,  # Use the extended y-axis
                       itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[1]),
                       gap="0%"
                       )
            .add_yaxis("Challenger Inferences", dfCl['num_inferences'].to_list(),
                       label_opts=opts.LabelOpts(is_show=False),
                       yaxis_index=1,  # Use the extended y-axis
                       itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[2]),
                       gap="0%"
                       )
            .add_yaxis("Challenger Ground Truths", dfCl['num_ground_truths'].to_list(),
                       label_opts=opts.LabelOpts(is_show=False),
                       yaxis_index=1,  # Use the extended y-axis
                       itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[3]),
                       gap="0%"
                       )    
            .extend_axis(
                yaxis=opts.AxisOpts(
                    type_="value",
                    name="Predictions & Ground Truths",
                    position="right",
                    name_location="middle",
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(formatter="{value}")

                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{col_names[i]}Performance Drift"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(save_as_image={})),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],

                legend_opts=opts.LegendOpts(pos_top="top",
                                            pos_left="center",
                                            orient="horizontal",
                                            item_width=10,
                                            item_height=8,
                                            textstyle_opts=opts.TextStyleOpts(color=bar_colors[2]) 



                                            ),  
                yaxis_opts=opts.AxisOpts(
                    name="Metrics",
                    name_location='middle',  
                    name_gap=45,  
                    axislabel_opts=opts.LabelOpts(formatter="{value}")

                )              

            )
        )

        line_chart = (
            Line()
            .add_xaxis(dfCh['formatted_predict_date'].to_list())
            .add_yaxis(f"Champion {col_names[i]}", dfCh[col].to_list(),

                       linestyle_opts=opts.LineStyleOpts(color=colors[0], width=3),
                           itemstyle_opts=opts.ItemStyleOpts(color=colors[0]),

                       symbol="emptycircle",
                        symbol_size=8,
                       label_opts=opts.LabelOpts(is_show=False)
                       )

            .add_yaxis(f"Challenger {col_names[i]}", dfCl[col].to_list(),

                       linestyle_opts=opts.LineStyleOpts(color=colors[1], width=3),
                        itemstyle_opts=opts.ItemStyleOpts(color=colors[1]),

                       symbol="emptycircle",
                        symbol_size=8,
                       label_opts=opts.LabelOpts(is_show=False)
                       )    
        )

        combined_chart = bar_chart.overlap(line_chart)
        for z in range(4,6):
            combined_chart.options.get("series")[z].update(zlevel=1001+z)  # Set z_level for line chart

        file_name = f"{col_names[i].replace(' ','_').lower()}"
        make_snapshot(driver, combined_chart.render(f"{temp_dir}/render_{file_name}.html"), f"{temp_dir}/bar_chart_perf_{file_name}.png", pixel_ratio=1)
        time.sleep(1)
    print(f"...finished generating {i+1} plots")        
        
def generatePlots(token,cluster,model_name,model_version,policy_name,theme='light'):
    generateMRMPlots(token
                ,cluster
                ,model_name
                ,model_version
                ,policy_name
                )    
    generateMRMPlotsAndSegments(token
                ,cluster
                ,model_name
                ,model_version
                ,policy_name
                )                     
    
def mV():
    shutil.move('src/variables.json', 'variables.json')
    
def pR(d):
    print(json.dumps(d,indent=7))
        
        
def generatePredictionDriftMetrics(token, cluster, vars_dict, isCC=False):
    prior_quarter_first_day, prior_prior_quarter_first_day = first_day_of_prior_quarters()
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters()  
    model_version='1'
    if isCC:
        models = ['CHAMPION', 'CHALLENGER']
        all_deltas = {}  # Empty dictionary to store the deltas for all models
        print(f"...generating prediction metrics")
        for model in models:
            modelname = vars_dict[model]['model_name']
            policy=getPredictionPolicies(modelname)
            data = get_drift_history(token, cluster, modelname, model_version, policy)
            data = data.sort_values('processed_ts')
            df = data[(data['processed_ts'] >= prior_prior_quarter_first_day) & (data['processed_ts'] <= prior_quarter_last_day)]
            deltas = {}
            print(f"   ...calculating deltas for {model.lower()} model")
            # Calculate the deltas for Day 1 to Day 2, Day 2 to Day 3, etc.
            for i in range(1, 6):
                start_day = df['processed_ts'].iloc[i - 1]
                end_day = df['processed_ts'].iloc[i]
                start_value = df.loc[df['processed_ts'] == start_day, 'results.drift_metrics.total_weighted_drift'].values[0]
                end_value = df.loc[df['processed_ts'] == end_day, 'results.drift_metrics.total_weighted_drift'].values[0]
                delta = round(((start_value - end_value) / end_value) * 100, 4)
                deltas[f"{model.lower()}_prediction_drift_day_{i}_to_day_{i + 1}"] = delta

            # Calculate the deltas for Week 1 to Week 2, Week 2 to Week 3, etc.
            for i in range(1, 6):
                start_week = df['processed_ts'].iloc[7 * i - 7]
                end_week = df['processed_ts'].iloc[7 * i]
                start_value = df.loc[df['processed_ts'] == start_week, 'results.drift_metrics.total_weighted_drift'].values[0]
                end_value = df.loc[df['processed_ts'] == end_week, 'results.drift_metrics.total_weighted_drift'].values[0]
                delta = round(((start_value - end_value) / end_value) * 100, 4)
                deltas[f"{model.lower()}_prediction_drift_week_{i}_to_week_{i + 1}"] = delta

            # Calculate the deltas for Month 1 to Month 2, Month 2 to Month 3, etc.
            for i in range(1, 6):
                start_index = 30 * i - 30 if 30 * i - 30 < len(df) else -1
                start_month = df['processed_ts'].iloc[start_index]
                end_index = 30 * i if 30 * i < len(df) else -1  # Use -1 to refer to the last index
                end_month = df['processed_ts'].iloc[end_index]
                start_value = df.loc[
                    df['processed_ts'] == start_month, 'results.drift_metrics.total_weighted_drift'].values[0]
                end_value = df.loc[df['processed_ts'] == end_month, 'results.drift_metrics.total_weighted_drift'].values[0]
                delta = round(((start_value - end_value) / end_value) * 100, 4)
                deltas[f"{model.lower()}_prediction_drift_month_{i}_to_month_{i + 1}"] = delta

            # Calculate the delta for Quarter 1 to Quarter 2
            start_quarter = prior_prior_quarter_first_day
            end_quarter = prior_quarter_last_day
            try:
                start_value = df.loc[
                    df['processed_ts'] == start_quarter, 'results.drift_metrics.total_weighted_drift'].values[0]
            except:                
                start_quarter_idx = (df['processed_ts'] - start_quarter).abs().idxmin()
                start_value = df.loc[start_quarter_idx, 'results.drift_metrics.total_weighted_drift']
            try:
                end_value = df.loc[df['processed_ts'] == end_quarter, 'results.drift_metrics.total_weighted_drift'].values[0]                
            except:
                end_quarter_idx = (df['processed_ts'] - end_quarter).abs().idxmin()
                end_value = df.loc[end_quarter_idx, 'results.drift_metrics.total_weighted_drift']

            delta = round(((start_value - end_value) / end_value) * 100, 4)
            deltas[f"{model.lower()}_prediction_drift_quarter_1_to_quarter_2"] = delta

            # Update the all_deltas dictionary with deltas for the current model
            all_deltas.update(deltas)
            all_deltas['prediction_drift_image']='bar_chart_perf_champion_challenger_prediction_feature_drift_image.png'
        print(f"...finished generating prediction metrics\n")

    else:

        modelname=vars_dict['model_name']
        policy=getPredictionPolicies(modelname)
        data=get_drift_history(token,cluster,modelname, model_version, policy_name=policy)
        data = data.sort_values('processed_ts')
        df=data[(data['processed_ts']>=prior_prior_quarter_first_day)
               &(data['processed_ts']<=prior_quarter_last_day)
               ]

        all_deltas = {}
        # Calculate the all_deltas for Day 1 to Day 2, Day 2 to Day 3, etc.
        for i in range(1, 6):
            start_day = df['processed_ts'].iloc[i-1]
            end_day = df['processed_ts'].iloc[i]
            start_value = df.loc[df['processed_ts'] == start_day, 'results.drift_metrics.total_weighted_drift'].values[0]
            end_value = df.loc[df['processed_ts'] == end_day, 'results.drift_metrics.total_weighted_drift'].values[0]
            delta = round(((start_value - end_value) / end_value) * 100, 4)
            all_deltas[f"prediction_drift_day_{i}_to_day_{i+1}"] = delta

        # Calculate the all_deltas for Week 1 to Week 2, Week 2 to Week 3, etc.
        for i in range(1, 6):
            start_week = df['processed_ts'].iloc[7*i-7]
            end_week = df['processed_ts'].iloc[7*i]
            start_value = df.loc[df['processed_ts'] == start_week, 'results.drift_metrics.total_weighted_drift'].values[0]
            end_value = df.loc[df['processed_ts'] == end_week, 'results.drift_metrics.total_weighted_drift'].values[0]
            delta = round(((start_value - end_value) / end_value) * 100, 4)
            all_deltas[f"prediction_drift_week_{i}_to_week_{i+1}"] = delta
        # Calculate the all_deltas for Month 1 to Month 2, Month 2 to Month 3, etc.
        for i in range(1, 6):
            start_month = df['processed_ts'].iloc[30*i-30]
            end_month = df['processed_ts'].iloc[30*i]
            start_value = df.loc[df['processed_ts'] == start_month, 'results.drift_metrics.total_weighted_drift'].values[0]
            end_value = df.loc[df['processed_ts'] == end_month, 'results.drift_metrics.total_weighted_drift'].values[0]
            delta = round(((start_value - end_value) / end_value) * 100, 4)
            all_deltas[f"prediction_drift_month_{i}_to_month_{i+1}"] = delta
        # Calculate the delta for Quarter 1 to Quarter 2
        start_quarter = prior_prior_quarter_first_day
        end_quarter = prior_quarter_last_day
        start_value = df.loc[df['processed_ts'] == start_quarter, 'results.drift_metrics.total_weighted_drift'].values[0]
        end_value = df.loc[df['processed_ts'] == end_quarter, 'results.drift_metrics.total_weighted_drift'].values[0]
        delta = round(((start_value - end_value) / end_value) * 100, 4)
        all_deltas[f"prediction_drift_quarter_1_to_quarter_2"] = delta

        # Print the dictionary of all_deltas
    return all_deltas   


def calculate_deltas(df, time_periods, period_counts, date_col, model, columns, col_names, prior_prior_quarter_first_day, prior_quarter_last_day):
    all_deltas = {}
    all_values = {}  # New dictionary to store the actual values at the end of each period

    for period in time_periods:
        period_count = period_counts.get(period, 6)  # if period not specified in period_counts, fallback to 6
        for i in range(1, period_count):
            if period == 'Day':
                start_date = df[date_col].iloc[i - 1]
                end_date = df[date_col].iloc[i]
            elif period == 'Week':
                start_date = df[date_col].iloc[7 * i - 7]
                end_date = df[date_col].iloc[7 * i]
            elif period == 'Month':
                start_date = df[date_col].iloc[30 * i - 30]
                end_date = df[date_col].iloc[30 * i]
            elif period == 'Quarter':
                start_date = str(prior_prior_quarter_first_day)
                end_date = str(prior_quarter_last_day)

            col_name = f"{model.lower()}_{period.lower()}_{i}_to_{i + 1}"

            try:
                for f, column in enumerate(columns):
                    delta_column_name = f"{col_name}_{col_names[f].lower().replace(' ', '_')}"
                    start_value = df.loc[df[date_col] == start_date, column].values[0]
                    end_value = df.loc[df[date_col] == end_date, column].values[0]
                    delta = round(((start_value - end_value) / end_value) * 100, 4)
                    all_deltas[delta_column_name] = delta

                    # Store the actual value at the end of each period
                    all_values[delta_column_name + "_val"] = round(end_value,4)
            except:
                print("...not all dates available, remapping to nearest dates..")
                for f, column in enumerate(columns):
                    delta_column_name = f"{col_name}_{col_names[f].lower().replace(' ', '_')}"
                    
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    
                    start_value_index = df[date_col].sub(start_date).abs().idxmin()
                    start_value = df.loc[start_value_index, column]
                    #print(f"..remapped start value: {start_value}")
                    end_value_index = df[date_col].sub(end_date).abs().idxmin()
                    end_value = df.loc[end_value_index, column]
                    #print(f"..remapped end value: {end_value}")
                    delta = round(((start_value - end_value) / end_value) * 100, 4)
                    all_deltas[delta_column_name] = delta

                    # Store the actual value at the end of each period
                    all_values[delta_column_name + "_val"] = round(end_value, 4)
                
    return all_deltas, all_values


def generateChChPerformanceMetrics(token, cluster, vars_dict):
    models = ['CHAMPION', 'CHALLENGER']
    all_deltas = {}  # Empty dictionary to store the deltas for all models
    all_values = {}  # New dictionary to store the actual values at the end of each period for all models
    print(f"...generating performance metrics")
    for model in models:
        print(f"   ...assembling {model.lower()} model performance metrics")
        modelname = vars_dict[model]['model_name']
        model_names = list(vars_dict.keys())

        date_col = 'formatted_predict_date'
        prior_quarter_first_day, prior_prior_quarter_first_day = first_day_of_prior_quarters()
        prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters()

        temp_dir = os.path.join(os.getcwd(), 'mrm')            
        img_dir = os.path.join(os.getcwd(), 'img') 
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        data, modeluuid = get_model_perf_history(cluster, token, modelname)
        data[date_col] = pd.to_datetime(data[date_col])
        df = data[['formatted_predict_date', 'num_ground_truths', 'num_inferences',
                   "metrics.False.accuracy", "metrics.False.balanced_accuracy",
                   "metrics.False.f1", "metrics.False.recall",
                   "metrics.False.specificity", "metrics.False.precision"]]\
            [(data['formatted_predict_date'] >= str(prior_prior_quarter_first_day)) &
             (data[date_col] <= str(prior_quarter_last_day))]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)


        columns = ['metrics.False.accuracy', 'metrics.False.balanced_accuracy', 'metrics.False.f1',
                   'metrics.False.recall', 'metrics.False.specificity', 'metrics.False.precision','num_inferences','num_ground_truths']
        col_names = ["Accuracy", "Balanced Accuracy", "F1", "Recall", "Specificity", "Precision", "Predictions", "Actuals"]
        time_periods = ['Day', 'Week', 'Month', 'Quarter']

        period_counts = {'Day': 8, 'Week': 9, 'Month': 6, 'Quarter': 2}


        deltas, values = calculate_deltas(df, ['Day', 'Week', 'Month', 'Quarter'], period_counts, date_col, model, columns, col_names, prior_prior_quarter_first_day, prior_quarter_last_day)
        all_deltas.update(deltas)
        all_values.update(values)  # Update the all_values dictionary with the values dictionary

    myKeys = list(all_deltas.keys())
    myKeys.sort()
    sorted_dict = {i: all_deltas[i] for i in myKeys}

    # Create a new dictionary to include the sorted values
    sorted_values = {i: all_values[i] for i in sorted(all_values.keys())}
    combined_dict = {}
    combined_dict.update(sorted_dict)
    combined_dict.update(sorted_values)
    print(f"...finished generating performance metrics\n")

    return combined_dict

def generatePredictionDriftPlots(token, cluster, vars_dict):
    prior_quarter_first_day, prior_prior_quarter_first_day = first_day_of_prior_quarters()
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters()  
    model_version='1'
    isCC=True
    dataAll=pd.DataFrame()

    models = ['CHAMPION', 'CHALLENGER']
    all_deltas = {}  # Empty dictionary to store the deltas for all models

    for model in models:
        print(f"...pulling data for {model.lower()} model")
        modelname = vars_dict[model]['model_name']
        policy=getPredictionPolicies(modelname)
        data = get_drift_history(token, cluster, modelname, model_version, policy)
        data = data.sort_values('processed_ts')
        df = data[(data['processed_ts'] >= prior_prior_quarter_first_day) & (data['processed_ts'] <= prior_quarter_last_day)]
        df['rowtype']=model
        dataAll=pd.concat([dataAll,df]).reset_index(drop=True)
    print(f"...combining and preparing data")
    columns=["drift_config.name","results.drift_metrics.total_weighted_drift","results.target_count","processed_ts","rowtype"]
    cNresultsDf=dataAll[columns]

    bar_colors=['#A8A3AE', '#C1BDC5', '#3E6389', '#77B1EC']
    colors= ["#827C8B","#A8A3AE","#45C1AF","#db93fe"]


    """
    champion
    """
    dfCh=cNresultsDf[(cNresultsDf['rowtype']=='CHAMPION')].groupby('processed_ts').min().round(4).reset_index()
    dfCh.drop(columns=['rowtype'],inplace=True)
    """
    challenger
    """
    dfCl=cNresultsDf[(cNresultsDf['rowtype']=='CHALLENGER')].groupby('processed_ts').min().round(4).reset_index()
    dfCl.drop(columns=['rowtype'],inplace=True)

    bar_chart = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="600px"))
        .add_xaxis(dfCh['processed_ts'].to_list())
        .add_yaxis("Champion Inferences", dfCh['results.target_count'].to_list(),
                   label_opts=opts.LabelOpts(is_show=False),
                   yaxis_index=1,  # Use the extended y-axis
                   itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[0]),
                   gap="0%"
                   )
        .add_yaxis("Challenger Inferences", dfCl['results.target_count'].to_list(),
                   label_opts=opts.LabelOpts(is_show=False),
                   yaxis_index=1,  # Use the extended y-axis
                   itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[1]),
                   gap="0%"
                   )

        .extend_axis(
            yaxis=opts.AxisOpts(
                type_="value",
                name="Inference Totals",
                position="right",
                name_location="middle",
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(formatter="{value}")

            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Champion & Challenger Prediction Feature Drift"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(save_as_image={})),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],

            legend_opts=opts.LegendOpts(pos_top="top",
                                        pos_left="center",
                                        orient="horizontal",
                                        item_width=10,
                                        item_height=8,
                                        textstyle_opts=opts.TextStyleOpts(color=bar_colors[2]) 



                                        ),  
            yaxis_opts=opts.AxisOpts(
                name="Metrics",
                name_location='middle',  
                name_gap=45,  
                axislabel_opts=opts.LabelOpts(formatter="{value}")

            )              

        )
    )    


    line_chart = (
        Line()
        .add_xaxis(dfCh['processed_ts'].to_list())
        .add_yaxis(f"Champion Prediction Drift", dfCh["results.drift_metrics.total_weighted_drift"].to_list(),

                   linestyle_opts=opts.LineStyleOpts(color=colors[2], width=3),
                       itemstyle_opts=opts.ItemStyleOpts(color=colors[2]),

                   symbol="emptycircle",
                    symbol_size=8,
                   label_opts=opts.LabelOpts(is_show=False)
                   )

        .add_yaxis(f"Challenger Prediction Drift", dfCl["results.drift_metrics.total_weighted_drift"].to_list(),

                   linestyle_opts=opts.LineStyleOpts(color=colors[3], width=3),
                    itemstyle_opts=opts.ItemStyleOpts(color=colors[3]),

                   symbol="emptycircle",
                    symbol_size=8,
                   label_opts=opts.LabelOpts(is_show=False)
                   )    
    )

    combined_chart = bar_chart.overlap(line_chart)
    
    for z in range(2,4):
        combined_chart.options.get("series")[z].update(zlevel=1001+z)  # Set z_level for line chart

    temp_dir = os.path.join(os.getcwd(), 'mrm')            
    file_name = f"champion_challenger_prediction_feature_drift_image"
    make_snapshot(driver, combined_chart.render(f"{temp_dir}/render_{file_name}.html"), f"{temp_dir}/bar_chart_perf_{file_name}.png", pixel_ratio=1)
    print("...plots generated successfully")     
    
def generateMRMmetricsCustomC(token, cluster, vars_dict, additional_metrics=None, current_date=None, previous_date=None, date_config=None, isCC=None):
    if isCC is None:
        model_name=vars_dict['model_name']
        model_version=vars_dict['model_version']
        additional_metrics=additional_metrics['additional_metrics']
        start_date=date_config['custom_start_date']
        end_date=date_config['custom_end_date']

        return generateMRMmetricsCustom(token,cluster,model_name,model_version
            ,additional_metrics=additional_metrics,start_date=start_date,end_date=end_date
                                              )
    elif isCC:
        prediction_drift_data=generatePredictionDriftMetrics(token, cluster, vars_dict, isCC=True)
        performance_data=generateChChPerformanceMetrics(token, cluster, vars_dict)        
        training_data=getChChtrainingMetrics(cluster, token)
        description_data=getMLFlowDescription(cluster, token, isCC=True)
        return prediction_drift_data,performance_data,description_data,training_data    
    
def generateMRMc(token,cluster,vars_dict,data,data2,modelDict,trainingDict,files,isCC=None):

    if isCC is None:
        model_name=vars_dict['model_name']
        model_version=vars_dict['model_version']
        policy_name=vars_dict['policy_name']
        
        generateMRMPlots(token
                    ,cluster
                    ,model_name
                    ,model_version
                    ,policy_name
                    )    
        generateMRMPlotsAndSegments(token
                    ,cluster
                    ,model_name
                    ,model_version
                    ,policy_name
                    )    
        
        if len(files)>1:
            repNum='reports'
        else:
            repNum='report'
        print(f"\n...generating {len(files)} {repNum}:")

        for n, file in enumerate(files):
            print(f"     {n+1}. {file}")
        print("")
        
        for i, file_name in enumerate(files):
            print(f"  ...loading template {i+1}: {file_name}.html")    
            temp_dir = os.path.join(os.getcwd(), 'mrm') 
            with open(f'{temp_dir}/{file_name}.html', 'r', encoding='ISO-8859-1') as f:
                html_doc = f.read()
            print("  ...automating metrics into template")
            soup = BeautifulSoup(html_doc, 'html.parser')
            print("  ...automating plots into template")
            for key, value in data.items(): #model performance 
                mrm_html_replace_and_highlight(soup, key, value, '#fdfe00')

            for key, value in data2.items(): #drift, segments, etc. 
                mrm_html_replace_and_highlight(soup, key, value, '#ff9a00')    

            for key, value in modelDict.items(): #databricks' mlflow
                mrm_html_replace_and_highlight(soup, key, value, '#00ff05')    
                
            for key, value in trainingDict.items(): #databricks training
                mrm_html_replace_and_highlight(soup, key, value, '#01c5ff')                
            print("  ...finalizing html")
            final_html = str(soup)

            with open(f'{temp_dir}/{file_name}_output.html', 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"  ...saving file {file_name}_output.html\n")

            soup = BeautifulSoup(final_html, 'html.parser')
            pretty_html = soup.prettify()

    elif isCC is not None:

        
        generateMRMperfComparisonPlots(cluster
                    ,token
                    ,vars_dict
                    )

        generatePredictionDriftPlots(token
                    ,cluster
                    ,vars_dict
                    )

        img_src_list= ['bar_chart_perf_specificity.png',
                     'bar_chart_perf_precision.png',
                     'bar_chart_perf_recall.png',
                     'bar_chart_perf_balanced_accuracy.png',
                     'bar_chart_perf_f1.png',
                     'bar_chart_perf_accuracy.png']

        if len(files)>1:
            repNum='reports'
        else:
            repNum='report'
        print(f"\n...generating {len(files)} {repNum}:")
        
        for n, file in enumerate(files):
            print(f"     {n+1}. {file}")
        print("")
        
        for i, file_name in enumerate(files):
            try:
                remove_carriage_returns(file_name)
            except:
                pass
            print(f"  ...loading template {i+1}: {file_name}.html")    
            temp_dir = os.path.join(os.getcwd(), 'mrm') 
            with open(f'{temp_dir}/{file_name}.html', 'r', encoding='ISO-8859-1') as f:
                html_doc = f.read()
            print("  ...automating metrics into template")
            soup = BeautifulSoup(html_doc, 'html.parser')
            print("  ...automating plots into template")
            for key, value in data.items(): #model performance 
                mrm_html_replace_and_highlight(soup, key, value, '#fdfe00')
                process_html(soup, img_src_list, key, value, color='#fdfe00')
            for key, value in data2.items(): #drift, segments, etc. 
                mrm_html_replace_and_highlight(soup, key, value, '#fdfe00')
                process_html(soup, img_src_list, key, value, color='#fdfe00')
            for key, value in modelDict.items(): #databricks' mlflow
                mrm_html_replace_and_highlight(soup, key, value, '#fdfe00')
                process_html(soup, img_src_list, key, value, color='#fdfe00')        
            for key, value in trainingDict.items(): #databricks training
                mrm_html_replace_and_highlight(soup, key, value, '#fdfe00')
                process_html(soup, img_src_list, key, value, color='#fdfe00')
            print("  ...finalizing html")
            final_html = str(soup)

            with open(f'{temp_dir}/{file_name}_output.html', 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"  ...saving file {file_name}_output.html")
            insert_image_grid(soup, img_src_list)
            soup = BeautifulSoup(final_html, 'html.parser')

            pretty_html = soup.prettify()
            
    remove_checkpoint_files('mrm')
    
    
    
def remove_checkpoint_files(directory):
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith("-checkpoint.html"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"  ...cleaning up {i}",end='\r',flush=True)    
            
def chat_with_model_tasks(analysisRequest, i, GPT4=True, tokens=700, temperature=1, max_retries=3):
    retry_count = 0
    while retry_count <= max_retries:
        try:
            #print(f"...analyzing request: {analysisRequest}")
            print(f"\n   ...analyzing request {i+1}")    
            print("   ...inquiring about appropriate task path")
            OPENAI_API_KEY = setOpenAItoken(cluster, token)
            openai.api_key = OPENAI_API_KEY

            sysTasks = """
            The tasks that my operating system can do are: 1. Retrieve model performance of last "n" quarters compared to prior "n" quarters, 
            2. Retrieve model performance of last "n" months vs. prior "n" months, 3. Retrieve quarter to quarter feature drift, 
            4. Retrieve month to month feature drift, 5. Retrive model performance for specific date range comparisons, 
            6. Retrive feature drift for specific date range comparisons
            """

            textRequest=f"""
            You are a data analyst, and you receive a request to "{analysisRequest}". Which task should be
            performed from the task list above? Only return the most appropriate one along with it's original number. If a request contains the word "performance" drift analysis is not appropriate. also keywords like month and feature drift or prediction drift should be applied specifically to the appropriate tasks.
            """

            specificDates=f"""
            Return the date range with actual dates (a total of four dates) for the task returned in only this format: "Current Period: 4/2/2023-4/8/2023 vs. Prior Period: 4/9/2023-4/15/2023"
            """
            message_objs = [{'role': 'system', 'content': 'You are talking to an AI model.'},
                            {'role': 'user', 'content': sysTasks},
                            {'role': 'user', 'content': textRequest},          
                            {'role': 'user', 'content': specificDates},                              
                            ] 

            response = openai.ChatCompletion.create(
                model="gpt-4-0613" if GPT4 else "gpt-3.5-turbo-0613",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature
            )
            rS = response.choices[0].message['content']
            time.sleep(5)
            #print(f"...chatGPT says: {rS}\n")
            return rS
        except Exception as e:
            print(f"...an error occurred: {e}")
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries})")
            time.sleep(2)  # Wait for a short time before retrying
    print("...max retry attempts reached, unable to complete the request.")
    return None

def chat_with_model_tasks_opAiFunc(analysisRequest, cluster, token, i, OPENAI_API_KEY=None, GPT4=True, tokens=1000, temperature=0.05, max_retries=3):
    openai.api_key = setOpenAItoken(cluster, token)
    retry_count = 0
    prior_quarter_first_day, prior_prior_quarter_first_day = first_day_of_prior_quarters()
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters()
    today = datetime.today().strftime('%Y-%m-%d') 
    
    while retry_count <= max_retries:
        try:
            print(f"\n   ...analyzing request {i+1}")    
            print("   ...inquiring about appropriate task path")

            sysTasks = """
            The tasks that my operating system can do are: 
            1. Retrieve model performance of last "n" quarters compared to prior "n" quarters, 
            2. Retrieve model performance of last "n" months vs. prior "n" months, 
            3. Retrieve quarter to quarter feature drift, 
            4. Retrieve month to month feature drift, 
            5. Retrive model performance for specific date range comparisons, 
            6. Retrive feature drift for specific date range comparisons
            """

            textRequest=f"""
            You are a data analyst, and you receive a request to "{analysisRequest}". Which task should be
            performed from the task list above? Only return the most appropriate one along with it's original number. If a request contains the word "performance" drift analysis is not appropriate. also keywords like month and feature drift or prediction drift should be applied specifically to the appropriate tasks.
            """

            message_objs = [
                {'role': 'system', 'content': 'You are talking to an AI model.'},
                {'role': 'user', 'content': sysTasks},
                {'role': 'user', 'content': textRequest},
            ] 

            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature
            )

            task_response = response.choices[0].message['content']

            specificDates=f"""
            Please call the 'chat_w_model_get_task_info' function with the required parameters. 
            Note: Today's date is {today}. The most recent quarter ranges from {prior_quarter_first_day} to {prior_quarter_last_day} and the quarter prior to that ranges from {prior_prior_quarter_first_day} to {prior_prior_quarter_last_day}

            """

            function_descriptions = [
                {
                    "name": "chat_w_model_get_task_info",
                    "description": "Get information related to a specific task, including the task number and dates for the current and prior periods",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_num": {"type": "integer",
                                         "description": "The unique number identifying the task, e.g. 1234"},
                            "current_period_start_date": {"type": "string", 
                                                          "description": "The start date of the current period for the task, e.g. '1900-01-01'"},
                            "current_period_end_date": {"type": "string", 
                                                        "description": "The end date of the current period for the task, e.g. '1900-03-31'"},
                            "prior_period_start_date": {"type": "string", 
                                                        "description": "The start date of the prior period for the task, e.g. '1900-10-01'"},
                            "prior_period_end_date": {"type": "string", 
                                                      "description": "The end date of the prior period for the task, e.g. '1900-12-31'"},
                            "query_type": {"type": "string", 
                                           "description": "The type of request e.g. 'performance' or 'drift' "},                
                        },
                        "required": ["task_num", "current_period_start_date", "current_period_end_date", "prior_period_start_date", "prior_period_end_date", "query_type"],
                    },
                }
            ]

            message_objs.append({'role': 'user', 'content': specificDates})

            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature,
                functions=function_descriptions,
            )

            function_output = response.choices[0].message.get('function_call')
            
            if function_output:
                task_info = function_output.get('arguments')
                return task_info
            
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"...an error occurred: {e}")
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries})")
            time.sleep(2)

    print("...max retry attempts reached, unable to complete the request.")
    return None



def chat_with_model_tasks_opAiFuncN(analysisRequest, cluster, token, i, OPENAI_API_KEY=None, GPT4=True, tokens=1000, temperature=0.05, max_retries=3):
    openai.api_key = setOpenAItoken(cluster, token)
    retry_count = 0

    prior_quarter_first_day, prior_prior_quarter_first_day = first_day_of_prior_quarters()
    prior_quarter_last_day, prior_prior_quarter_last_day = last_day_of_prior_quarters()
    today = datetime.today().strftime('%Y-%m-%d')  # get today's date

    while retry_count <= max_retries:
        try:
            print(f"\n   ...analyzing request {i+1}")    
            print("   ...inquiring about appropriate task path")

            sysTasks = f"""
            The tasks that my operating system can do are: 1. Retrieve model performance of last "n" quarters compared to prior "n" quarters, 
            2. Retrieve model performance of last "n" months vs. prior "n" months, 
            3. Retrieve quarter to quarter feature drift, 
            4. Retrieve month to month feature drift, 
            5. Retrive model performance for specific date range comparisons, 
            6. Retrive feature drift for specific date range comparisons.             
            """
            textRequest=f"""
            You are a data analyst, and you receive a request to "{analysisRequest}". Which task should be
            performed from the task list above? Only return the most appropriate one along with it's original number. If a request contains the word "performance" drift analysis is not appropriate. also keywords like month and feature drift or prediction drift should be applied specifically to the appropriate tasks.
            """
            message_objs = [
                {'role': 'system', 'content': 'You are a data analyst.'},
                {'role': 'user', 'content': sysTasks},
                {'role': 'user', 'content': textRequest},
            ] 
            
            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature
            )

            task_response = response.choices[0].message['content']
            task_number = int(task_response.split('.')[0])  # get the task number from the response

            print(f"Identified task: {task_response}")

            specificDates=f"""
            Please call the 'chat_w_model_get_task_info' function with the required parameters. 
            Note: Today's date is {today}. The most recent quarter ranges from {prior_quarter_first_day} to {prior_quarter_last_day} and the quarter prior to that ranges from {prior_prior_quarter_first_day} to {prior_prior_quarter_last_day}

            """

            function_descriptions = [
                {
                    "name": "chat_w_model_get_task_info",
                    "description": "Get information related to a specific task, including the task number and dates for the current and prior periods",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_num": {"type": "integer",
                                         "description": "The unique number identifying the task, e.g. 1234"},
                            "current_period_start_date": {"type": "string", 
                                                          "description": "The start date of the current period for the task, e.g. '1900-01-01'"},
                            "current_period_end_date": {"type": "string", 
                                                        "description": "The end date of the current period for the task, e.g. '1900-03-31'"},
                            "prior_period_start_date": {"type": "string", 
                                                        "description": "The start date of the prior period for the task, e.g. '1900-10-01'"},
                            "prior_period_end_date": {"type": "string", 
                                                      "description": "The end date of the prior period for the task, e.g. '1900-12-31'"},
                            "query_type": {"type": "string", 
                                           "description": "The type of request e.g. 'performance' or 'drift' "},                
                        },
                        "required": ["task_num", "current_period_start_date", "current_period_end_date", "prior_period_start_date", "prior_period_end_date", "query_type"],
                    },
                }
            ]

            message_objs.append({'role': 'user', 'content': specificDates})

            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature,
                functions=function_descriptions,
            )

            function_output = response.choices[0].message.get('function_call')
            
            if function_output:
                task_info = function_output.get('arguments')
                return task_info
            
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"...an error occurred: {e}")
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries})")
            time.sleep(2)

    print("...max retry attempts reached, unable to complete the request.")
    return None


def chat_with_model_summary(messages, cluster, token, GPT4=False, tokens=700, temperature=1, max_retries=3):
    retry_count = 0
    while retry_count <= max_retries:
        try:
            openai.api_key = setOpenAItoken(cluster, token)

            message_objs = [
                {'role': 'system', 'content': 'You are talking to an AI model.'},
                {'role': 'user', 'content': 'You are an expert in machine learning model monitoring, and I need you to help me understand the performance of my models performance from the most recent ended period (e.g. month, quarter, etc.) (Current) compared to the previous ended period (e.g. month, quarter, etc.) (Prior). I will provide you with some metrics to give you context and I would like to you to explain to me in 350-500 words what I need to know, how I can address any negative outcomes, or maintain any positive outcomes, and you should note specifically that these details are to be inputted into my MRM documentation in financial services (see SR letter 11-7 from the board of governors of the federal reserve system office of the comptroller of the currency.'},
                {'role': 'user', 'content': str(messages)},                    
            ] 

            response = openai.ChatCompletion.create(
                model="gpt-4-0613" if GPT4 else "gpt-3.5-turbo-0613",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature
            )

            rS = response.choices[0].message['content']  
            return rS  
        except Exception as e:
            print(f"...an error occurred: {e}")
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries})")
            time.sleep(2)  # Wait for a short time before retrying
    print("...max retry attempts reached, unable to complete the request.")
    return None

def chat_with_model_metrics(analysisRequest, GPT4=False, tokens=2000, temperature=0.1):

    openai.api_key = setOpenAItoken(cluster, token)

    sysTasks = """
    My system can generate the following metrics (some in the form of acronyms): [accuracy,balanced_accuracy,f1,recall,specificity,precision,tn,fp,fn,tp,npv,fnr,fpr,fdr,for,plr,nlr,pt,ts,mcc,fm,bm,mk,dor,rnp], where the first six are standard. Return non-standard metrics in their original form if specifically referenced in my request.
    """
    req1="""
    'Your response should be: "non-standard metrics: [observed_metric_from_original_request] unsupported metrics: [unsupported metrics]". If no metrics are observed, your response should be: "non-standard metrics: [] unsupported metrics: []".'
    """
    req2="""
    your response should be in the following format: "non-standard metrics: [observed_metric_from_original_request] unsupported metrics: [unsupported metrics]"
    """
    req3="""
    if no metrics are observed in my request, your response should be in the following format: "non-standard metrics: [] unsupported metrics: []"
    """    
    allMetrics="""
    all metrics=[accuracy,balanced_accuracy,f1,recall,specificity,precision,tn,fp,fn,tp,npv,fnr,fpr,fdr,for,plr,nlr,pt,ts,mcc,fm,bm,mk,dor,rnp]
    """
    nonStandardMetrics="""non-standard metrics=[tn,fp,fn,tp,npv,fnr,fpr,fdr,for,plr,nlr,pt,ts,mcc,fm,bm,mk,dor,rnp]
    """
    message_objs = [{'role': 'system', 'content': 'You are talking to an AI model.'},
                    {'role': 'user', 'content': sysTasks},
                    {'role': 'user', 'content': f"my request is as follows: {analysisRequest}. Only return metrics explicitly mentioned in my request. Do not return the following metrics: accuracy, balanced_accuracy, f1, recall, specificity, precision."},         
                    {'role': 'user', 'content': req1},                              
                    ] 
    
    response = openai.ChatCompletion.create(
        model="gpt-4-0613" if GPT4 else "gpt-3.5-turbo-0613",
        messages=message_objs,
        max_tokens=tokens,
        n=1,
        temperature=temperature
    )

    rS=response.choices[0].message['content']
    time.sleep(5)    
    return rS

def chat_with_model_metrics_opAiFunc(analysisRequest, cluster, token, i,  OPENAI_API_KEY=None, GPT4=True, tokens=1000, temperature=0.1, max_retries=3, custom_metric_function=None):
    openai.api_key = setOpenAItoken(cluster, token)
    retry_count = 0
    while retry_count <= max_retries:
        try:
            non_standard_metrics=["tn","fp","fn","tp","npv","fnr","fpr","fdr","for","plr","nlr","pt","ts","mcc","fm","bm","mk","dor","rnp"]
            already_supported_metrics=["accuracy","balanced_accuracy","f1","recall","specificity","precision"]
            sysTasks = """
            My system can generate the following metrics (some in the form of acronyms): [accuracy,balanced_accuracy,f1,recall,specificity,precision,tn,fp,fn,tp,npv,fnr,fpr,fdr,for,plr,nlr,pt,ts,mcc,fm,bm,mk,dor,rnp], where the first six are standard. Return non-standard metrics in their original form if specifically referenced in my request.
            """
            req1="""
            'Your response should be: "non-standard metrics: [observed_metric_from_original_request] unsupported metrics: [unsupported metrics]". If no metrics are observed, your response should be: "non-standard metrics: [] unsupported metrics: []".'
            """
            allMetrics="""
            all metrics=[accuracy,balanced_accuracy,f1,recall,specificity,precision,tn,fp,fn,tp,npv,fnr,fpr,fdr,for,plr,nlr,pt,ts,mcc,fm,bm,mk,dor,rnp]
            """
            nonStandardMetrics=f"""non-standard metrics={non_standard_metrics}
            """
            message_objs = [{'role': 'system', 'content': 'You are talking to an AI model.'},
                            {'role': 'user', 'content': sysTasks},
                            {'role': 'user', 'content': f"my request is as follows: {analysisRequest}. Only return metrics explicitly mentioned in my request. Do not return the following metrics: accuracy, balanced_accuracy, f1, recall, specificity, precision."},         
                            {'role': 'user', 'content': req1},                              
                            ] 
            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo-0613",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature
            )
            task_response=response.choices[0].message['content']

            specificRequest=f"""
            Please call the 'chat_w_model_get_metrics' function with the required parameters. The non_standard_metrics parameter should be filled with any metrics ({non_standard_metrics}) observed in prior request. The already_supported_metrics should be filled with any metrics ({already_supported_metrics}) observed. Any metrics not in {non_standard_metrics+already_supported_metrics} should be included in unsupported_metrics parameter. And any observed python functions, like this example: 
                     "def cost_based_fscore_custom(precision, recall, alpha=0.5, beta=1.0):
                        cost_based_fscore = (1 + (alpha*beta)**2) * ((precision * recall) / ((alpha*beta)**2 * precision + recall))
                        return cost_based_fscore"
                        should be included in the custom_function parameter as a string.
            """
            function_descriptions = [
                {
                    "name": "chat_w_model_get_metrics",
                    "description": "Get information about specific metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "non_standard_metrics": {"type": "string",
                                                    "description": f"metrics included in {non_standard_metrics}"},
                            "unsupported_metrics": {"type": "string",
                                                   "description": f"metrics not included in {non_standard_metrics+already_supported_metrics}"},
                            "already_supported_metrics": {"type": "string",
                                                         "description": f"metrics included in {already_supported_metrics}"},
                            "custom_function": {"type": "string",
                                               "description": f"any observed python functions"},
                        },
                        "required": ["non_standard_metrics","unsupported_metrics","already_supported_metrics","custom_function"],
                    },
                }
            ]
            message_objs.append({'role': 'user', 'content': specificRequest})

            response = openai.ChatCompletion.create(
                model="gpt-4" if GPT4 else "gpt-3.5-turbo",
                messages=message_objs,
                max_tokens=tokens,
                n=1,
                temperature=temperature,
                functions=function_descriptions,
            )

            function_output = response.choices[0].message.get('function_call')

            if function_output:
                task_info = function_output.get('arguments')
                return task_info

            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries+1})")
            time.sleep(2)
        except Exception as e:
            print(f"...an error occurred: {e}")
            retry_count += 1
            print(f"...retrying: (attempt {retry_count} of {max_retries+1})")
            time.sleep(2)

    print("...max retry attempts reached, unable to complete the request.")
    return None

def extract_dates(request):
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    dates_str = re.findall(date_pattern, request)

    if dates_str:  # If explicit dates are found in the string
        dates = [datetime.strptime(date, '%m/%d/%Y') for date in dates_str]
        # If there are more than four dates, take only the last four.
        dates.sort()
        if len(dates) > 4:
            dates = dates[-4:]
        # If there are only two dates, consider them as end dates and generate start dates
        elif len(dates) == 2:
            start_dates = [date - pd.DateOffset(days=1) for date in dates]
            dates = start_dates + dates
        return dates
    else:  # If no explicit dates are found, calculate based on period type
        current_date = datetime.now()

        if 'quarter' in request:
            period = 'quarter'
            end_date = pd.to_datetime(current_date) - pd.offsets.QuarterEnd(n=1)
            start_date = end_date - pd.DateOffset(months=3) + pd.DateOffset(days=1)
        elif 'month' in request:
            period = 'month'
            end_date = pd.to_datetime(current_date) - pd.offsets.MonthEnd(n=1)
            start_date = end_date - pd.DateOffset(months=1) + pd.DateOffset(days=1)
        elif 'week' in request:
            period = 'week'
            end_date = pd.to_datetime(current_date) - pd.offsets.Week(weekday=6)  # set to the end of the week (Sunday)
            start_date = end_date - pd.DateOffset(days=7) + pd.DateOffset(days=1)
        elif 'day' in request:
            period = 'day'
            end_date = pd.to_datetime(current_date) - pd.DateOffset(days=1)
            start_date = end_date - pd.DateOffset(days=1)

        # Return calculated dates for current period and prior period
        current_period = [start_date, end_date]
        prior_period = [start_date - pd.DateOffset(months=1), end_date - pd.DateOffset(months=1)] if period == 'month' else \
                        [start_date - pd.DateOffset(months=3), end_date - pd.DateOffset(months=3)] if period == 'quarter' else \
                        [start_date - pd.DateOffset(weeks=1), end_date - pd.DateOffset(weeks=1)] if period == 'week' else \
                        [start_date - pd.DateOffset(days=1), end_date - pd.DateOffset(days=1)]  # for 'day'
        current_period_start, current_period_end=current_period[0], current_period[1]
        previous_period_start, previous_period_end=prior_period[0], prior_period[1]        
        
        return current_period_start, current_period_end, previous_period_start, previous_period_end

def check_sequential_dates(start_date1, end_date1, start_date2, end_date2):
    # Convert the date strings to datetime objects
    date1_start = datetime.strptime(start_date1, '%Y-%m-%d')
    date1_end = datetime.strptime(end_date1, '%Y-%m-%d')
    date2_start = datetime.strptime(start_date2, '%Y-%m-%d')
    date2_end = datetime.strptime(end_date2, '%Y-%m-%d')

    # Add 1 day to the end date of the first range
    date1_end_plus_one = date1_end + timedelta(days=1)

    # Check if the start date of the second range is the day after the end date of the first range
    if date1_end_plus_one == date2_start:
        return True

    return False

def normalize_dates(df, date_col):
    min_date = df[date_col].min()
    df['day'] = (df[date_col] - min_date).dt.days + 1
    return df

def generate_metrics(token,cluster,modeluuid,policyName,request,i,additional_metrics_raw=None,unsupported_metrics=None):
    metricsDict={'tn': 'metrics.False.tn', 'fp': 'metrics.False.fp', 'fn': 'metrics.False.fn', 'tp': 'metrics.False.tp', 'accuracy': 'metrics.False.accuracy', 'balanced_accuracy': 'metrics.False.balanced_accuracy', 'f1': 'metrics.False.f1', 'recall': 'metrics.False.recall', 'specificity': 'metrics.False.specificity', 'precision': 'metrics.False.precision', 'npv': 'metrics.False.npv', 'fnr': 'metrics.False.fnr', 'fpr': 'metrics.False.fpr', 'fdr': 'metrics.False.fdr', 'for': 'metrics.False.for', 'plr': 'metrics.False.plr', 'nlr': 'metrics.False.nlr', 'pt': 'metrics.False.pt', 'ts': 'metrics.False.ts', 'mcc': 'metrics.False.mcc', 'fm': 'metrics.False.fm', 'bm': 'metrics.False.bm', 'mk': 'metrics.False.mk', 'dor': 'metrics.False.dor', 'rnp': 'metrics.False.rnp'}
    start_date, end_date, prev_start_date, prev_end_date=extract_dates(request)

    start_date = pd.to_datetime(start_date)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    end_date = pd.to_datetime(end_date)
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    prev_start_date = pd.to_datetime(prev_start_date)
    prev_start_date_str = prev_start_date.strftime('%Y-%m-%d')
    
    prev_end_date = pd.to_datetime(prev_end_date)
    prev_end_date_str = prev_end_date.strftime('%Y-%m-%d')    
    #print(start_date_str, end_date_str, prev_start_date_str, prev_end_date_str)

    
    if 'model performance' in request:
        df_type = 'perfHist'
        pH = pd.json_normalize(json.loads(get_model_perf(modeluuid, token, cluster))).reset_index(drop=True)
        
        # Extend the existing metrics list with the additional metrics
        metrics_columns=['formatted_predict_date','num_inferences','metrics.False.accuracy','metrics.False.balanced_accuracy','metrics.False.f1','metrics.False.recall','metrics.False.specificity','metrics.False.precision']

        if additional_metrics_raw is not None:
            additional_metrics=[]
            for m in additional_metrics_raw:
                additional_metrics.append(metricsDict[m])
                
        if additional_metrics is not None:
            for metric in additional_metrics:
                try:
                    if metric in pH.columns:
                        metrics_columns.append(metric)
                        print(f"   ...added metric: \033[0m\033[93m{metric}\033[0m")
                    else:
                        #metric = "performance"
                        print(f"\033[91mWARNING:\033[0m Metric {metric} is not in the data")
                except Exception as e:
                    print(f"Error: {e}")    
                    
        if additional_metrics is not None:
            for u in unsupported_metrics:
                print(f"\033[0m    ...\033[37m{u}\033[0mnot supported")
                 
        df = pH[metrics_columns]
        datetime_column = 'formatted_predict_date'
    elif 'feature drift' in request:
        print(f"   ...querying ACME drift detection endpoint")        
        df_type = 'driftHist'
        df = get_drift_hist(modeluuid, token, cluster, policyName)
        display(df.head())
        datetime_column = 'processed_ts'
    
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df_selected = df[(df[datetime_column] >= start_date) & (df[datetime_column] <= end_date)]
    df_prev_selected = df[(df[datetime_column] >= prev_start_date) & (df[datetime_column] <= prev_end_date)]

    """
    generate plots
    """
    df_all = pd.concat([df_selected,df_prev_selected])
    print(df_all)
    df_all['date'] = df_all[datetime_column].dt.date  # Ensure we're working with dates (not datetimes)
    df_all = df_all.groupby('date').mean()  # Aggregate by day
    df_all.reset_index(inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors= ["#8419B2","#DA94FF","#085B70","#8F6B01","#A6181D"]
    color_iter = itertools.cycle(colors)  # This will cycle through the colors if there are more lines than colors

    # Add line plots for all other metric values
    for column in df_all.columns:
        if column not in ['date', 'num_inferences', 'results.target_count','results.drift_metrics.total_weighted_drift_critical','results.drift_metrics.total_weighted_drift_critical_flag','results.drift_metrics.total_weighted_drift_warning','results.drift_metrics.total_weighted_drift_warning_flag','metrics.False.recall','metrics.False.specificity','metrics.False.precision']:
            fig.add_trace(go.Scatter(x=df_all['date'], y=df_all[column], mode='lines+markers', name=column, line_color=next(color_iter)), secondary_y=True
)

    # Add bar plot for 'num_inferences' or 'results.target_count'
    bar_color = '#D3D3D3'  # Light gray
    if df_type == 'perfHist':
        fig.add_trace(go.Bar(x=df_all['date'], y=df_all['num_inferences'], name='num_inferences', marker_color=bar_color), secondary_y=False)
    elif df_type == 'driftHist':
        fig.add_trace(go.Bar(x=df_all['date'], y=df_all['results.target_count'], name='target_count', marker_color=bar_color), secondary_y=False)

                    
    fig.update_layout(
        autosize=False,
        width=1920,  # Fit for a wide screen
        height=500,
        hoverlabel=dict(namelength=-1),
        legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Anchor legend at bottom
        y=1.02,  # Put a little bit above the top of the plot
        xanchor="center",  # Anchor legend at right
        x=0.5  # Put all the way to the right
    )
    )

    fig.write_image(f"mrm/plot_{i}.png")  # Save the plot as a PNG
    print(f"   ...generated image {i+1}")
    #fig.show()
    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###
    
    display(df_selected.head())
    metrics = {}
    for column in df_selected.columns:
        if df_type == 'perfHist' and 'metrics' in column:
            metric_name = column.split('.')[-1]
        elif df_type == 'driftHist' and 'results.drift_metrics' in column:
            metric_name = column.split('.')[-1]
        else:
            continue
        curr_metric = df_selected[column].mean()
        prev_metric = df_prev_selected[column].mean()
        delta = curr_metric - prev_metric
        metrics[metric_name + '_' + 'pp' + '_current'] = round(curr_metric, 4)
        metrics[metric_name + '_' + 'pp' + '_previous'] = round(prev_metric, 4)
        metrics[metric_name + '_' + 'pp' + '_delta'] = round(delta, 4)

    #print(f"metrics: {metrics}")
    metrics_df = pd.DataFrame.from_records([metrics])
    metrics_df = metrics_df.melt(var_name='full_metric')
    metrics_df[['metric', 'period', 'measure']] = metrics_df['full_metric'].str.rsplit('_', n=2, expand=True)
    #print("..metrics df:")
    #display(metrics_df.head())
    suffix=metrics_df.period.unique()[0]
    metrics_df = metrics_df.pivot_table(index=['metric', ], columns='measure', values='value').reset_index()
    metrics_df.columns.name = None
    metrics_df.columns = ['metric', f"current_{suffix}", 'delta', f"previous_{suffix}"]
    display(metrics_df.columns)
    metrics_df=metrics_df[['metric', f"current_{suffix}", f"previous_{suffix}",'delta']]
    #print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
    return metrics, start_date_str, end_date_str, prev_start_date_str, prev_end_date_str


def generate_apply_code(func_str, valid_args):
    # Parse the function string using ast
    exec(func_str)
    func_def = ast.parse(func_str)
    func_name = func_def.body[0].name
    #print(astor.to_source(func_def))
    # Extract the argument names from the function definition
    arg_names_orig = [arg.arg.lower() for arg in func_def.body[0].args.args]
    arg_names=[]
    for a in arg_names_orig:
        try:
            print(valid_args[a])
            arg_names.append(valid_args[a])
        except:
            pass
    display(arg_names)
    # Generate the line of code that applies the function to a DataFrame
    args_str = ', '.join([f"row['{arg_name}']" for arg_name in arg_names])
    apply_code = f"df['{func_name}'] = df.apply(lambda row: {func_name}({args_str}), axis=1)"
    print(f"   ...custom function def {func_name}(...): used to add metric {func_name}")
    return apply_code


def generate_metrics_opAiFunc(token,cluster,modeluuid,policyName,opai_func_dict_raw,opai_metr_dict_raw,i):
    metricsDict={'tn': 'metrics.False.tn', 'fp': 'metrics.False.fp', 'fn': 'metrics.False.fn', 'tp': 'metrics.False.tp', 'accuracy': 'metrics.False.accuracy', 'balanced_accuracy': 'metrics.False.balanced_accuracy', 'f1': 'metrics.False.f1', 'recall': 'metrics.False.recall', 'specificity': 'metrics.False.specificity', 'precision': 'metrics.False.precision', 'npv': 'metrics.False.npv', 'fnr': 'metrics.False.fnr', 'fpr': 'metrics.False.fpr', 'fdr': 'metrics.False.fdr', 'for': 'metrics.False.for', 'plr': 'metrics.False.plr', 'nlr': 'metrics.False.nlr', 'pt': 'metrics.False.pt', 'ts': 'metrics.False.ts', 'mcc': 'metrics.False.mcc', 'fm': 'metrics.False.fm', 'bm': 'metrics.False.bm', 'mk': 'metrics.False.mk', 'dor': 'metrics.False.dor', 'rnp': 'metrics.False.rnp'}
    #start_date, end_date, prev_start_date, prev_end_date=extract_dates(request)
    opai_metr_dict=json.loads(opai_metr_dict_raw)
    opai_func_dict=json.loads(opai_func_dict_raw)
    #display(opai_func_dict)
    #display(opai_metr_dict)
    
    start_date, end_date, prev_start_date, prev_end_date=opai_func_dict['current_period_start_date'],opai_func_dict['current_period_end_date'],opai_func_dict['prior_period_start_date'],opai_func_dict['prior_period_end_date']
    start_date = pd.to_datetime(start_date)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    end_date = pd.to_datetime(end_date)
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    prev_start_date = pd.to_datetime(prev_start_date)
    prev_start_date_str = prev_start_date.strftime('%Y-%m-%d')
    
    prev_end_date = pd.to_datetime(prev_end_date)
    prev_end_date_str = prev_end_date.strftime('%Y-%m-%d')    
    #print(start_date_str, end_date_str, prev_start_date_str, prev_end_date_str)

    
    if 'performance' in opai_func_dict['query_type']:
        print(f"   ...querying ACME model performance endpoint")
        df_type = 'perfHist'
        pH = pd.json_normalize(json.loads(get_model_perf(modeluuid, token, cluster))).reset_index(drop=True)
        
        # Extend the existing metrics list with the additional metrics
        metrics_columns=['formatted_predict_date','num_inferences','metrics.False.accuracy','metrics.False.balanced_accuracy','metrics.False.f1','metrics.False.recall','metrics.False.specificity','metrics.False.precision']
        """
        if additional_metrics_raw is not None:
            additional_metrics=[]
            for m in additional_metrics_raw:
                additional_metrics.append(metricsDict[m])
                
        if additional_metrics is not None:
            for metric in additional_metrics:
                try:
                    if metric in pH.columns:
                        metrics_columns.append(metric)
                        print(f"   ...added metric: \033[0m\033[93m{metric}\033[0m")
                    else:
                        #metric = "performance"
                        print(f"\033[91mWARNING:\033[0m Metric {metric} is not in the data")
                except Exception as e:
                    print(f"Error: {e}")    
        """
        additional_metrics=[]        
        if len(opai_metr_dict['non_standard_metrics']) >1:
            for m in opai_metr_dict['non_standard_metrics'].split(","):
                try:
                    additional_metrics.append(metricsDict[m])
                except:
                    #print("   ...custom metric not found")
                    pass

        if len(additional_metrics) > 1:
            for metric in additional_metrics:
                try:
                    if metric in pH.columns:
                        metrics_columns.append(metric)
                        print(f"   ...added metric: \033[0m\033[93m{metric}\033[0m")
                    else:
                        #metric = "performance"
                        print(f"\033[91mWARNING:\033[0m Metric {metric} is not in the data")
                except Exception as e:
                    print(f"Error: {e}")  
        if len(opai_metr_dict['unsupported_metrics']) >5:
            for u in opai_metr_dict['unsupported_metrics'].split(","):
                try:
                    print(f"\033[0m   ...\033[37m{u}\033[0m not supported")
                except IndexError:
                    continue            
            
        #print(metrics_columns)            
        df = pH[metrics_columns]

        if len(opai_metr_dict['custom_function']) > 5:
            print(f"   ...custom function found:\n         {opai_metr_dict['custom_function']}")
            
            max_retries = 1
            retry_count = 0
            success = False
            while not success and retry_count < max_retries:
                try:
                    df = pH[metrics_columns]

                    code_string=opai_metr_dict['custom_function']
                    def_str_parts = code_string.split(":")
                    if len(def_str_parts) < 2:
                        print(f"Invalid custom function format for: {code_string}")
                        break

                    #func_str = def_str_parts[0] + ":" + def_str_parts[1].replace("\n ", "\n    ")
                    func_str = def_str_parts[0].strip() + ":" + def_str_parts[1].replace("\n ", "\n    ")         #added strip() to part 0 to alleviate leading spaces before "def"            
                    func_def = ast.parse(func_str)
                    func_name = func_def.body[0].name
                    arg_names_orig = [arg.arg.lower() for arg in func_def.body[0].args.args]

                    arg_names=[]
                    for a in arg_names_orig:
                        try:
                            #print(metricsDict[a])
                            arg_names.append(metricsDict[a])
                        except:
                            pass
                    #print(f"arg names: {arg_names}")
                    # Generate the line of code that applies the function to a DataFrame
                    args_str = ', '.join([f"row['{arg_name}']" for arg_name in arg_names])
                    apply_code = f"df['custmetrics.{func_name}'] = df.apply(lambda row: {func_name}({args_str}), axis=1)"
                    print(f"   ...custom function def {func_name}() used to add metric {func_name}")

                    #print(opai_metr_dict['custom_function'])
                    #exec(opai_metr_dict['custom_function'])
                    #exec(func_str)
                    exec(func_str, globals())
                    namespace = dict(globals(), **locals())
                    exec(apply_code, namespace)
                    print(f"   ...successfully added custom metric")
                    success = True  # Add this line to indicate the operation was successful.

                except IndexError as ie:
                    print("An error occurred: if custom metric was specified it was missed by model")   
                    break

                except Exception as e:
                    retry_count += 1
                    print(f"An error occurred: {str(e)}")
                    #print(f"Retrying... (Attempt {retry_count}/{max_retries})")
                        
            if not success:
                #print("Failed to execute the custom function after multiple retries.")
                raise MaxRetriesExceededError("Failed to execute the custom function after multiple retries.")
                
        datetime_column = 'formatted_predict_date'
        
    elif 'drift' in opai_func_dict['query_type']:
        print(f"   ...querying ACME drift detection endpoint")        
        df_type = 'driftHist'
        #print(f"policy: {policyName}")
        df = get_drift_hist(modeluuid, token, cluster, policyName)
        datetime_column = 'processed_ts'
    
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df_selected = df[(df[datetime_column] >= start_date) & (df[datetime_column] <= end_date)]
    df_prev_selected = df[(df[datetime_column] >= prev_start_date) & (df[datetime_column] <= prev_end_date)]

    """
    generate plots
    """
    df_all = pd.concat([df_selected,df_prev_selected])
    df_all['date'] = df_all[datetime_column].dt.date  # Ensure we're working with dates (not datetimes)
    cols_to_include = df_all.select_dtypes(include=[float, int]).columns.tolist()
    df_all = df_all.groupby('date')[cols_to_include].mean()    
    #df_all = df_all.groupby('date').mean()  # Aggregate by day
    df_all.reset_index(inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors= ["#8419B2","#DA94FF","#085B70","#8F6B01","#A6181D"]
    color_iter = itertools.cycle(colors)  # This will cycle through the colors if there are more lines than colors

    # Add line plots for all other metric values

    for column in df_all.columns:
        if column not in ['date', 'num_inferences', 'results.target_count','results.drift_metrics.total_weighted_drift_critical','results.drift_metrics.total_weighted_drift_critical_flag','results.drift_metrics.total_weighted_drift_warning','results.drift_metrics.total_weighted_drift_warning_flag','metrics.False.recall','metrics.False.specificity','metrics.False.precision']:
            fig.add_trace(go.Scatter(x=df_all['date'], y=df_all[column], mode='lines+markers', name=column, line_color=next(color_iter)), secondary_y=True
)

    # Add bar plot for 'num_inferences' or 'results.target_count'
    bar_color = '#D3D3D3'  # Light gray
    if df_type == 'perfHist':
        fig.add_trace(go.Bar(x=df_all['date'], y=df_all['num_inferences'], name='num_inferences', marker_color=bar_color), secondary_y=False)
    elif df_type == 'driftHist':
        fig.add_trace(go.Bar(x=df_all['date'], y=df_all['results.target_count'], name='target_count', marker_color=bar_color), secondary_y=False)

                    
    fig.update_layout(
        autosize=False,
        width=1920,  # Fit for a wide screen
        height=500,
        hoverlabel=dict(namelength=-1),
        legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Anchor legend at bottom
        y=1.02,  # Put a little bit above the top of the plot
        xanchor="center",  # Anchor legend at right
        x=0.5  # Put all the way to the right
    )
    )

    fig.write_image(f"mrm/plot_{i}.png")  # Save the plot as a PNG
    print(f"   ...generated image {i+1}")
    #fig.show()
    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###
    
    metrics = {}
    for column in df_selected.columns:
        if df_type == 'perfHist' and 'metrics' in column:
            metric_name = column.split('.')[-1]
        elif df_type == 'driftHist' and 'results.drift_metrics' in column:
            metric_name = column.split('.')[-1]
        else:
            continue
        curr_metric = df_selected[column].mean()
        prev_metric = df_prev_selected[column].mean()
        delta = curr_metric - prev_metric
        metrics[metric_name + '_' + 'pp' + '_current'] = round(curr_metric, 4)
        metrics[metric_name + '_' + 'pp' + '_previous'] = round(prev_metric, 4)
        metrics[metric_name + '_' + 'pp' + '_delta'] = round(delta, 4)

    """
    metrics_df = pd.DataFrame.from_records([metrics])
    metrics_df = metrics_df.melt(var_name='full_metric')
    metrics_df[['metric', 'period', 'measure']] = metrics_df['full_metric'].str.rsplit('_', n=2, expand=True)
    suffix=metrics_df.period.unique()[0]
    metrics_df = metrics_df.pivot_table(index=['metric', ], columns='measure', values='value').reset_index()
    metrics_df.columns.name = None
    metrics_df.columns = ['metric', f"current_{suffix}", 'delta', f"previous_{suffix}"]
    metrics_df=metrics_df[['metric', f"current_{suffix}", f"previous_{suffix}",'delta']]
    """
    #print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
    remove_checkpoint_files('mrm')
    return metrics, start_date_str, end_date_str, prev_start_date_str, prev_end_date_str


def extract_parts(key):
    # Extract the base parameter, optional flag, timeframe, and metric from the key
#    match = re.match(r'^(.*?)(?:_(critical|warning))?_(quarter|month)_(current|previous|delta)$', key)
    match = re.match(r'^(.*?)(?:_(critical|warning))?_(vo|month|quarter|day|pp)_(current|previous|delta)$', key)
    if match:
        return match.groups()
    return None

#parse template for requests

#file_name = file_config['files'][0]
#modeluuid="bdb83435-8c91-45ef-a399-e03244264849"
#policyName='All Feat Base 68B Day'

def extract_metrics(string):
    non_standard_metrics = re.findall(r'non-standard metrics: \[([\w\s,]+)\]', string)
    unsupported_metrics = re.findall(r'unsupported metrics: \[([\w\s,]+)\]', string)
    
    non_standard_metrics = [metric.strip() for metric in non_standard_metrics[0].split(',')] if non_standard_metrics else []
    unsupported_metrics = [metric.strip() for metric in unsupported_metrics[0].split(',')] if unsupported_metrics else []    
    return non_standard_metrics, unsupported_metrics


class MaxRetriesExceededError(Exception):
    pass


def generateMRMeval(token, cluster, vars_dict, file_config, abstract=None):
    file_name=file_config['files'][0]    
    model_name=vars_dict['model_name']

    policyName=getAllFeatPolicies(vars_dict,is_primary=None)
    
    #print(policyName)
    rf, modeluuid=get_model_perf_history(cluster, token, model_name)
    #print(rf, modeluuid)
    extR=parseHTMLtemplate(file_name)  
    fileNO=f'mrm/{file_name}_{abstract}_out.html'
    fileN=f'mrm/{file_name}.html'
    with open(fileN, 'r') as file:
        html_doc = file.read()

    soup = BeautifulSoup(html_doc, 'html.parser')    
    max_loop_retries = 5  # Define the maximum number of retries for each loop iteration

    for i, r in enumerate(extR):
        loop_retry_count = 0  # Initialize the retry count for the current loop iteration
        
        while loop_retry_count < max_loop_retries:
            try:
                analysisRequestP = r
                rrS = chat_with_model_tasks_opAiFunc(analysisRequestP, cluster, token, i)
                mtR = chat_with_model_metrics_opAiFunc(analysisRequestP, cluster, token, i)
                data, start_date_str, end_date_str, prev_start_date_str, prev_end_date_str = generate_metrics_opAiFunc(token, cluster, modeluuid, policyName, rrS, mtR, i)

                if abstract == 'actual':
                    print("   ...using actual data for summarization")
                    chat_summary=chat_with_model_summary(data,cluster,token)
                elif abstract == 'scaling':
                    scaler = StandardScaler()
                    scaler.fit(np.array(list(data.values())).reshape(-1, 1))
                    scaled_data = scaler.transform(np.array(list(data.values())).reshape(-1, 1))
                    scaled_data_dict = dict(zip(data.keys(), scaled_data.flatten()))
                    print("   ...using scaling data abstraction for summarization")
                    chat_summary = chat_with_model_summary(scaled_data_dict,cluster,token)
                elif abstract == 'binning':
                    n_bins = 5  
                    binarizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
                    binned_data = binarizer.fit_transform(np.array(list(data.values())).reshape(-1, 1))
                    binned_data_dict = dict(zip(data.keys(), binned_data.flatten()))            
                    print("   ...using binned data abstraction for summarization")
                    chat_summary = chat_with_model_summary(binned_data_dict,cluster,token)
                elif abstract == 'noise':
                    noise = np.random.normal(0, 0.1, len(data))
                    data_with_noise = np.array(list(data.values())) + noise
                    data_with_noise_dict = dict(zip(data.keys(), data_with_noise.flatten()))
                    print("   ...using data abstraction with noise for summarization")
                    chat_summary = chat_with_model_summary(data_with_noise_dict,cluster,token)
                elif abstract == 'permutation':
                    permuted_data = np.random.permutation(list(data.values()))
                    permuted_data_dict = dict(zip(data.keys(), permuted_data.flatten()))
                    print("   ...using data permutation abstraction for summarization")
                    chat_summary = chat_with_model_summary(permuted_data_dict,cluster,token)
                elif abstract == 'language':
                    abstracted_data = abstract_with_language(data)
                    print("   ...using natural language abstraction for summarization")
                    chat_summary = chat_with_model_summary(abstracted_data,cluster,token)
                else:
                    print("   ...no summarization specified")
                    chat_summary = ""           

                placeholder_text="{{"+analysisRequestP+"}}"
                parts = [extract_parts(key) for key in data.keys()]
                parameters = set(part[0] for part in parts if part)
                flags = set(part[1] for part in parts if part and part[1])
                flags.add(None)
                timeframes = set(part[2] for part in parts if part)
                for timeframe in timeframes:
                    new_table = soup.new_tag('table')
                    header_row = soup.new_tag('tr')
                    for header_text in ['Metric', f'Period Ended {end_date_str}', f'Prior Period Ended {prev_end_date_str}', 'Delta']:
                        header_cell = soup.new_tag('th')
                        header_cell.string = header_text
                        header_row.append(header_cell)
                    new_table.append(header_row)
                    
                    for parameter in parameters:
                        for flag in flags:
                            base_key = parameter
                            if flag:
                                base_key += f'_{flag}'
                            base_key += f'_{timeframe}'

                            if not any(k.startswith(base_key) for k in data.keys()):
                                continue

                            row = soup.new_tag('tr')
                            cell1 = soup.new_tag('td')
                            cell1.string = base_key.replace('_', ' ').title()
                            row.append(cell1)
                            cell2 = soup.new_tag('td')
                            cell2.string = str(data.get(f'{base_key}_current', ''))
                            row.append(cell2)
                            cell3 = soup.new_tag('td')
                            cell3.string = str(data.get(f'{base_key}_previous', ''))
                            row.append(cell3)
                            cell4 = soup.new_tag('td')
                            cell4.string = str(data.get(f'{base_key}_delta', ''))
                            row.append(cell4)
                            new_table.append(row)

                    style_tag = soup.new_tag('style')
                    style_tag.string = """
                    table {
                        border-collapse: collapse;
                        width: 100%;
                    }
                    th, td {
                        border: 1px solid black;
                        padding: 8px;
                        text-align: left;
                    }
                    """
                    soup.head.append(style_tag)

                    for string in soup.stripped_strings:
                        if placeholder_text in string.replace('\n', ' ').replace('\r', ''):
                            placeholder = soup.find(text=string)
                            break
                    else:
                        placeholder = None

                    if placeholder:
                        placeholder.replace_with(new_table)
                    else:
                        print("Placeholder text not found in the HTML document.")
                    
                    chat_summary_paragraph = soup.new_tag('p')
                    chat_summary_paragraph.string = chat_summary
                    new_table.insert_after(chat_summary_paragraph)        
                    image_tag = soup.new_tag('img',style="width: 100%")
                    image_tag['src'] = f'plot_{i}.png'
                    chat_summary_paragraph.insert_after(image_tag)            
                
                break  # If the function executes successfully, break the retry loop and proceed to the next iteration

            except MaxRetriesExceededError:
                # If an error occurs, increment the retry count and let the loop repeat
                loop_retry_count += 1
                print(f"Retrying loop iteration... (Attempt {loop_retry_count}/{max_loop_retries})")

        # If the maximum retries were reached for the current loop iteration without success,
        # print an error message and skip to the next iteration
        if loop_retry_count == max_loop_retries:
            print(f"Failed to execute loop iteration after {max_loop_retries} retries. Skipping to the next loop iteration...")
            continue  # Skip to the next iteration

        font_family = "Nunito Sans"

        for element in soup.find_all():
            element['style'] = f"font-family: {font_family}; {element.get('style', '')}"

    pretty_html = soup.prettify()
    with open(fileNO, 'w') as fileW:
        fileW.write(str(pretty_html))
    print("...report generated successfully\n")
    
    
    
def parseHTMLtemplate(file_name):
    file=f'mrm/{file_name}.html'
    print(f"...parsing {file}")
    with open(file, 'r') as file:
        html_doc = file.read()

    soup = BeautifulSoup(html_doc, 'html.parser')

    decoded_html = html.unescape(str(soup))

    result = []
    for string in soup.stripped_strings:
        if "{{" in string and "}}" in string:
            result.append(string[string.index("{{")+2:string.index("}}")].replace('\n', ' ').replace('\r', ''))
    print(f"...extracted {len(result):,} requests")
    return result        


from bs4 import BeautifulSoup, NavigableString
from html import unescape

def parseReplaceHTML(file_name):
    file_raw = file_name['files'][0]
    file = f'mrm/{file_raw}.html'
    with open(file, 'r') as file:
        html_doc = file.read()

    soup = BeautifulSoup(html_doc, 'html.parser')

    # Your original method for finding strings between {{...}}
    result = []
    matches = []
    for string in soup.stripped_strings:
        if "{{" in string and "}}" in string:
            str_ = string[string.index("{{")+2:string.index("}}")].replace('\n', ' ').replace('\r', '')
            result.append(str_)
            matches.append(string)

    print(f"...extracted {len(result):,} requests")

    # Print extracted strings and ask user to select one
    for i, string in enumerate(result):
        print(f"{i}: {string}")
    print(f"{len(result)}: Add new string")
    print(f"{len(result)+1}: Remove a string")
    selected_string_index = int(input("Enter the index of the string you want to replace, add, or remove: "))
    
    if selected_string_index == len(result):
        # Add new string at the end
        new_string = input("Enter the new string: ")
        # Create 'p' tag
        new_p_tag = soup.new_tag("p", **{"class": "MsoNormal", "style": "margin-top:12.0pt"})
        # Create 'span' tag and nest it inside 'p'
        new_span_tag = soup.new_tag("span", **{"style": 'font-size:10.5pt; font-family:"Nunito Sans";color:#5fbc04'})
        new_span_tag.string = f"{{{{{new_string}}}}}"
        new_p_tag.insert(0, new_span_tag)
        # Create 'o:p' tag and nest it inside 'span'
        new_o_p_tag = soup.new_tag("o:p")
        new_span_tag.insert(1, new_o_p_tag)
        # Add new 'p' tag to soup body
        soup.body.append(new_p_tag)
        print(soup.body)

    elif selected_string_index == len(result) + 1:
        # Remove a string
        for i, string in enumerate(result):
            print(f"{i}: {string}")
        remove_string_index = int(input("Enter the index of the string you want to remove: "))

        # Find the tag containing the original string
        for tag in soup.find_all(text=lambda t: matches[remove_string_index] in t):
            # Remove the string within the tag
            tag.string.replace_with(tag.string.replace(matches[remove_string_index], ''))
    else:
        # Print the original string before asking for the new string
        print(f"Original string: {result[selected_string_index]}")
        new_string = input("Enter the new string: ")

        # Find the tag containing the original string
        for tag in soup.find_all(text=lambda t: matches[selected_string_index] in t):
            # Replace the string within the tag
            tag.string.replace_with(tag.string.replace(matches[selected_string_index], f'{{{{{new_string}}}}}'))

    with open(f'mrm/{file_raw}.html', 'w') as file:
        file.write(soup.prettify())
    print(f"...updated HTML document")

    

def abstract_with_language(data):
    abstracted_data = {}
    for key, value in data.items():
        if "delta" in key:
            base_metric = key.split('_')[0]
            if value > 0:
                change = "an increase"
            elif value < 0:
                change = "a decrease"
            else:
                change = "no change"

            if abs(value) < 0.001:
                magnitude = "a negligible amount"
            elif abs(value) < 0.01:
                magnitude = "a small amount"
            elif abs(value) < 0.1:
                magnitude = "a moderate amount"
            else:
                magnitude = "a significant amount"

            abstracted_data[base_metric] = f"There was {change} of {magnitude} in {base_metric} compared to the previous period."
    
    return abstracted_data