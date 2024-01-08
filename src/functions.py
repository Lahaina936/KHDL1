import numpy as np
import pandas as pd

import xgboost as xgb

import lightgbm as lgb
from lightgbm import (
    early_stopping,
    log_evaluation,
)

from sklearn.model_selection import (
    StratifiedKFold, 
    TimeSeriesSplit,
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

def remove_playoff_games(df):
    
    df = df[df["PLAYOFF"] == 0]
    
    df = df.drop("PLAYOFF", axis=1)
    
    return df

def fix_datatypes(df):
    df['GAME_DATE_EST'] = df['GAME_DATE_EST'].apply(lambda x: x[:10])
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])

    long_integer_fields = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']

    #convert long integer fields to int32 from int64
    for field in long_integer_fields:
        df[field] = df[field].astype('int32')
    
    #convert the remaining int64s to int8
    for field in df.select_dtypes(include=['int64']).columns.tolist():
        df[field] = df[field].astype('int8')
        
    #convert float64s to float16s
    for field in df.select_dtypes(include=['float64']).columns.tolist():
        df[field] = df[field].astype('float16')
        
    return df

def add_date_features(df):
    #convert game date to month to limit cardinality

    df['MONTH'] = df['GAME_DATE_EST'].dt.month
    
    return df

def process_x_minus_league_avg(df, feature_list, team_feature):
    
    # create feature list for temp dataframe to hold league averages
    temp_feature_list = feature_list.copy()
    temp_feature_list.append(team_feature)
    temp_feature_list.append("GAME_DATE_EST")
   
    df_temp = df[temp_feature_list]

    # populate the dataframe with all days played and forward fill previous value if a particular team did not play that day
    # https://stackoverflow.com/questions/70362869
    df_temp = (df_temp.set_index('GAME_DATE_EST')
            .groupby([team_feature])[feature_list]
            .apply(lambda x: x.asfreq('d', method = "ffill"))
            .reset_index()
            [temp_feature_list]
            )
    
    # find the average across all teams for each day
    df_temp = df_temp.groupby(['GAME_DATE_EST'])[feature_list].mean().reset_index()
    
    # rename features for merging
    df_temp = df_temp.add_suffix('_LEAGUE_AVG')
    temp_features = df_temp.columns
    
    # merge all-team averages with each record so that they can be subtracted
    df = df.sort_values(by = 'GAME_DATE_EST', axis=0, ascending= True, ignore_index=True)   
    df = pd.merge(df, df_temp, left_on='GAME_DATE_EST', right_on='GAME_DATE_EST_LEAGUE_AVG', how="left",)
    for feature in feature_list:
        df[feature + "_MINUS_LEAGUE_AVG"] = df[feature] - df[feature + "_LEAGUE_AVG"]

    # drop temp features that were only used for subtraction
    df = df.drop(temp_features, axis = 1)
    
    return df

def add_rolling_home_visitor(df, location, roll_list): 
    
    location_id = location + "_TEAM_ID"

    # sort games by the order in which they were played for each home or visitor team
    df = df.sort_values(by = [location_id, 'GAME_DATE_EST'], axis=0, ascending=[True, True,], ignore_index=True)
    
    # Win streak, negative if a losing streak
    df[location + '_TEAM_WIN_STREAK'] = df['HOME_TEAM_WINS'].groupby((df['HOME_TEAM_WINS'].shift() != df.groupby([location_id])['HOME_TEAM_WINS'].shift(2)).cumsum()).cumcount() + 1
    # if home team lost the last game of the streak, then the streak must be a losing streak. make it negative
    df[location + '_TEAM_WIN_STREAK'].loc[df['HOME_TEAM_WINS'].shift() == 0] =  -1 * df[location + '_TEAM_WIN_STREAK']

    # If visitor, the streak has opposite meaning (3 wins in a row for home team is 3 losses in a row for visitor)
    if location == 'VISITOR':
        df[location + '_TEAM_WIN_STREAK'] = - df[location + '_TEAM_WIN_STREAK']  


    # rolling means
    feature_list = ['HOME_TEAM_WINS', 'PTS_home',
       'FGM_home', 'FGA_home', 'FG_PCT_home', '3PM_home', '3PA_home',
       'FG3_PCT_home', 'FTM_home', 'FTA_home', 'FT_PCT_home', 'OREB_home',
       'DREB_home', 'REB_home', 'AST_home', 'STL_home', 'BLK_home', 'TOV_home',
       'PF_home', '+/-_home']
    
    if location == 'VISITOR':
        feature_list = ['HOME_TEAM_WINS', 'PTS_away',
            'FGM_away', 'FGA_away', 'FG_PCT_away', '3PM_away', '3PA_away',
            'FG3_PCT_away', 'FTM_away', 'FTA_away', 'FT_PCT_away', 'OREB_away',
            'DREB_away', 'REB_away', 'AST_away', 'STL_away', 'BLK_away', 'TOV_away',
            'PF_away', '+/-_away']
    
      
    roll_feature_list = []
    for feature in feature_list:
        for roll in roll_list:
            roll_feature_name = location + '_' + feature + '_AVG_LAST_' + str(roll) + '_' + location
            if feature == 'HOME_TEAM_WINS': #remove the "HOME_" for better readability
                roll_feature_name = location + '_' + feature[5:] + '_AVG_LAST_' + str(roll) + '_' + location
            roll_feature_list.append(roll_feature_name)
            df[roll_feature_name] = df.groupby(['HOME_TEAM_ID'])[feature].rolling(roll, closed= "left").mean().values
            
    
    roll_feature_list = [x for x in roll_feature_list if not x.startswith('HOME_TEAM_WINS')]
    
    df = process_x_minus_league_avg(df, roll_feature_list, location_id)
    
 
    return df

def process_games_consecutively(df_data):
    df_home = pd.DataFrame()
    df_home['GAME_DATE_EST'] = df_data['GAME_DATE_EST']
    df_home['GAME_ID'] = df_data['GAME_ID']
    df_home['TEAM1'] = df_data['HOME_TEAM_ID']
    df_home['TEAM1_home'] = 1
    df_home['TEAM1_win'] = df_data['HOME_TEAM_WINS']
    df_home['TEAM2'] = df_data['VISITOR_TEAM_ID']
    df_home['SEASON'] = df_data['SEASON']

    df_home['PTS'] = df_data['PTS_home']
    df_home['FGM'] = df_data['FGM_home']
    df_home['FGA'] = df_data['FGA_home']
    df_home['FG_PCT'] = df_data['FG_PCT_home']
    df_home['FT_PCT'] = df_data['FT_PCT_home']
    df_home['FG3_PCT'] = df_data['FG3_PCT_home']
    df_home['AST'] = df_data['AST_home']
    df_home['REB'] = df_data['REB_home']
    df_home['3PM'] = df_data['3PM_home']
    df_home['3PA'] = df_data['3PA_home']
    df_home['FTM'] = df_data['FTM_home']
    df_home['FTA'] = df_data['FTA_home']
    df_home['OREB'] = df_data['OREB_home']
    df_home['DREB'] = df_data['DREB_home']
    df_home['STL'] = df_data['STL_home']
    df_home['BLK'] = df_data['BLK_home']
    df_home['PF'] = df_data['PF_home']
    df_home['TOV'] = df_data['TOV_home']
    df_home['+/-'] = df_data['+/-_home']


    
    # now for visitor teams  

    df_visitor = pd.DataFrame()
    df_visitor['GAME_DATE_EST'] = df_data['GAME_DATE_EST']
    df_visitor['GAME_ID'] = df_data['GAME_ID']
    df_visitor['TEAM1'] = df_data['VISITOR_TEAM_ID'] 
    df_visitor['TEAM1_home'] = 0
    df_visitor['TEAM1_win'] = df_data['HOME_TEAM_WINS'].apply(lambda x: 1 if x == 0 else 0)
    df_visitor['TEAM2'] = df_data['HOME_TEAM_ID']
    df_visitor['SEASON'] = df_data['SEASON']
    
    df_visitor['FGA'] = df_data['FGA_away']
    df_visitor['FGM'] = df_data['FGM_away']
    df_visitor['FG_PCT'] = df_data['FG_PCT_away']
    df_visitor['FT_PCT'] = df_data['FT_PCT_away']
    df_visitor['FG3_PCT'] = df_data['FG3_PCT_away']
    df_visitor['AST'] = df_data['AST_away']
    df_visitor['REB'] = df_data['REB_away']
    df_visitor['3PM'] = df_data['3PM_away']
    df_visitor['3PA'] = df_data['3PA_away']
    df_visitor['FTM'] = df_data['FTM_away']
    df_visitor['FTA'] = df_data['FTA_away']
    df_visitor['OREB'] = df_data['OREB_away']
    df_visitor['DREB'] = df_data['DREB_away']
    df_visitor['STL'] = df_data['STL_away']
    df_visitor['BLK'] = df_data['BLK_away']
    df_visitor['PF'] = df_data['PF_away']
    df_visitor['TOV'] = df_data['TOV_away']
    df_visitor['+/-'] = df_data['+/-_away']
    df_visitor['PTS'] = df_data['PTS_away']

    # merge dfs

    df = pd.concat([df_home, df_visitor])

    column2 = df.pop('TEAM1')
    column3 = df.pop('TEAM1_home')
    column4 = df.pop('TEAM2')
    column5 = df.pop('TEAM1_win')

    df.insert(2,'TEAM1', column2)
    df.insert(3,'TEAM1_home', column3)
    df.insert(4,'TEAM2', column4)
    df.insert(5,'TEAM1_win', column5)

    df = df.sort_values(by = ['TEAM1', 'GAME_ID'], axis=0, ascending=[True, True], ignore_index=True)

    return df

def add_matchups(df, roll_list):

    df = df.sort_values(by = ['TEAM1', 'TEAM2','GAME_DATE_EST'], axis=0, ascending=[True, True, True], ignore_index=True)

    for roll in roll_list:
        df['MATCHUP_WINPCT_' + str(roll)] = df.groupby(['TEAM1','TEAM2'])['TEAM1_win'].rolling(roll, closed= "left").mean().values

    df['MATCHUP_WIN_STREAK'] = df['TEAM1_win'].groupby((df['TEAM1_win'].shift() != df.groupby(['TEAM1','TEAM2'])['TEAM1_win'].shift(2)).cumsum()).cumcount() + 1
    # if team1 lost the last game of the streak, then the streak must be a losing streak. make it negative
    df['MATCHUP_WIN_STREAK'].loc[df['TEAM1_win'].shift() == 0] = -1 * df['MATCHUP_WIN_STREAK']
  
    
    return df

def add_past_performance_all(df, roll_list):

    df = df.sort_values(by = ['TEAM1','GAME_DATE_EST'], axis=0, ascending=[True, True,], ignore_index=True)
  
    #streak of games won/lost, make negative is a losing streak
    df['WIN_STREAK'] = df['TEAM1_win'].groupby((df['TEAM1_win'].shift() != df.groupby(['TEAM1'])['TEAM1_win'].shift(2)).cumsum()).cumcount() + 1   
    # if team1 lost the last game of the streak, then the streak must be a losing streak. make it negative
    df['WIN_STREAK'].loc[df['TEAM1_win'].shift() == 0]  = -1 * df['WIN_STREAK']
    
    #streak of games played at home/away, make negative if away streak
    df['HOME_AWAY_STREAK'] = df['TEAM1_home'].groupby((df['TEAM1_home'].shift() != df.groupby(['TEAM1'])['TEAM1_home'].shift(2)).cumsum()).cumcount() + 1
    # if team1 played the game of the streak away, then the streak must be an away streak. make it negative
    df['HOME_AWAY_STREAK'].loc[df['TEAM1_home'].shift() == 0]  = -1 * df['HOME_AWAY_STREAK']
    
    #rolling means 
    
    feature_list = ['TEAM1_win', 'PTS',
       'FGM', 'FGA', 'FG_PCT', '3PM', '3PA',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
       'PF', '+/-']
   
    #create new feature names based upon rolling period
    
    roll_feature_list =[]

    for feature in feature_list:
        for roll in roll_list:
            roll_feature_name = feature + '_AVG_LAST_' + str(roll) + '_ALL'
            roll_feature_list.append(roll_feature_name)
            df[roll_feature_name] = df.groupby(['TEAM1'])[feature].rolling(roll, closed= "left").mean().values
    roll_feature_list = [x for x in roll_feature_list if not x.startswith('TEAM1_win')]
    
    df = process_x_minus_league_avg(df, roll_feature_list, 'TEAM1')
    
    
    return df

def combine_new_features(df, df_consecutive):
     
    # add back all the new features created in the consecutive dataframe to the main dataframe
    # all data for TEAM1 will be applied to the home team and then again to the visitor team
    # except for head-to-head MATCHUP data, which will only be applied to home team (redundant to include for both)
    # the letter '_x' will be appeneded to feature names when adding to home team
    # the letter '_y' will be appended to feature names when adding to visitor team
    # to match the existing convention in the dataset
    
    #first select out the new features
    all_features = df_consecutive.columns.tolist()
    link_features = ['GAME_ID', 'TEAM1', ]
    redundant_features = ['GAME_DATE_EST','TEAM1_home','TEAM1_win','TEAM2','SEASON', 'PTS',
       'FGM', 'FGA', 'FG_PCT', '3PM', '3PA',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
       'PF', '+/-']
    matchup_features = [x for x in all_features if "MATCHUP" in x]
    ignore_features = link_features + redundant_features
    
    new_features = [x for x in all_features if x not in ignore_features]
    
    # first home teams
    
    df1 = df_consecutive[df_consecutive['TEAM1_home'] == 1]
    #add "_x" to new features
    df1.columns = [x + '_x' if x in new_features else x for x in df1.columns]
    #drop features that don't need to be merged
    df1 = df1.drop(redundant_features,axis=1)
    #change TEAM1 to HOME_TEAM_ID for easy merging
    df1 = df1.rename(columns={'TEAM1': 'HOME_TEAM_ID'})
    df = pd.merge(df, df1, how="left", on=["GAME_ID", "HOME_TEAM_ID"])
    
    #don't include matchup features for visitor team since they are equivant for both home and visitor
    new_features = [x for x in new_features if x not in matchup_features]
    df_consecutive = df_consecutive.drop(matchup_features,axis=1)
    
    # next visitor teams
    
    df2 = df_consecutive[df_consecutive['TEAM1_home'] == 0]
    #add "_y" to new features
    df2.columns = [x + '_y' if x in new_features else x for x in df2.columns]
    #drop features that don't need to be merged
    df2 = df2.drop(redundant_features,axis=1)
    #change TEAM1 to VISITOR_TEAM_ID for easy merging
    df2 = df2.rename(columns={'TEAM1': 'VISITOR_TEAM_ID'})
    df = pd.merge(df, df2, how="left", on=["GAME_ID", "VISITOR_TEAM_ID"])
    
    return df

def remove_non_rolling(df):
    drop_columns =[]
    
    all_columns = df.columns.tolist()
    
    drop_columns1 = ['HOME_TEAM_WINS', 
       'PTS_home', 'FGM_home', 'FGA_home', 'FG_PCT_home', 
       '3PM_home', '3PA_home', 'FG3_PCT_home', 'FTM_home', 'FTA_home', 
       'FT_PCT_home', 'OREB_home', 'DREB_home', 'REB_home', 'AST_home', 
       'STL_home', 'BLK_home', 'TOV_home', 'PF_home', '+/-_home', 'Team_home']
    drop_columns2 = [
       'PTS_away', 'FGM_away', 'FGA_away', 'FG_PCT_away',
       '3PM_away', '3PA_away', 'FG3_PCT_away', 'FTM_away', 'FTA_away',
       'FT_PCT_away', 'OREB_away', 'DREB_away', 'REB_away', 'AST_away',
       'STL_away', 'BLK_away', 'TOV_away', 'PF_away', '+/-_away', 'Team_away']
    
    drop_columns = drop_columns + drop_columns1
    drop_columns = drop_columns + drop_columns2 
    
    use_columns = [item for item in all_columns if item not in drop_columns]
    
    return df[use_columns]

def process_x_minus_y(df):
    #Subtract visitor teams stats from the home teams stats for key fields
    # field_x - field_y
    
    all_features = df.columns.tolist()
    comparison_features = [x for x in all_features if "_y" in x]
    
    #don't include redunant features. (x - league_avg) - (y - league_avg) = x-y
    comparison_features = [x for x in comparison_features if "_MINUS_LEAGUE_AVG" not in x]
    
    for feature in comparison_features:
        feature_base = feature[:-2] #remove "_y" from the end
        df[feature_base + "_x_minus_y"] = df[feature_base + "_x"] - df[feature_base + "_y"]
        
    #df = df.drop("CONFERENCE_x_minus_y") #category variable not meaningful?
        
    return df
    
def add_all_features(df):
    
    home_visitor_roll_list = [3]
    all_roll_list = [3]
        
    df = remove_playoff_games(df)
    df = fix_datatypes(df)
    df = add_date_features(df)
    df = add_rolling_home_visitor(df, "HOME", home_visitor_roll_list)
    df = add_rolling_home_visitor(df, "VISITOR", home_visitor_roll_list)

    #games must first be processed to sort all games in order per team
    #regardless whether home or away
    df_consecutive = process_games_consecutively(df)
    df_consecutive = add_matchups(df_consecutive, home_visitor_roll_list)
    df_consecutive = add_past_performance_all(df_consecutive, all_roll_list)

    #add these features back to main dataframe
    df = combine_new_features(df,df_consecutive)
    
    df['TARGET'] = df['HOME_TEAM_WINS']

    
    df = remove_non_rolling(df)
    
    df = process_x_minus_y(df)
    
    return df






def encode_categoricals(df: pd.DataFrame, category_columns: list, MODEL_NAME: str, ENABLE_CATEGORICAL: bool) -> pd.DataFrame:
    """
    Encode categorical features as integers for use in XGBoost and LightGBM

    Args:
        df (pd.DataFrame): the DataFrame to process
        category_columns (list): list of columns to encode as categorical
        MODEL_NAME (str): the name of the model being used
        ENABLE_CATEGORICAL (bool): whether or not to enable categorical features in the model
    
    Returns:
        the DataFrame with categorical features encoded
    

    """

    first_team_ID = df['HOME_TEAM_ID'].min()
    first_season = df['SEASON'].min()
   
    # subtract lowest value from each to create a range of 0 thru N-1
    df['HOME_TEAM_ID'] = (df['HOME_TEAM_ID'] - first_team_ID).astype('int8') #team ID - 1610612737 = 0 thru 29
    df['VISITOR_TEAM_ID'] = (df['VISITOR_TEAM_ID'] - first_team_ID).astype('int8') 
    df['SEASON'] = (df['SEASON'] - first_season).astype('int8')
    
    # if xgb experimental categorical capabilities are to be used, then features must be of category type
    if MODEL_NAME == "xgboost":
        if ENABLE_CATEGORICAL:
            for field in category_columns:
                df[field] = df[field].astype('category')

    return df







def XGB_objective(trial, train, target, STATIC_PARAMS, ENABLE_CATEGORICAL, NUM_BOOST_ROUND, OPTUNA_CV, OPTUNA_FOLDS, SEED):
    

    train_oof = np.zeros((train.shape[0],))
    
    train_dmatrix = xgb.DMatrix(train, target,
                         feature_names=train.columns,
                        enable_categorical=ENABLE_CATEGORICAL)
    
    xgb_params= {       
                'num_round': trial.suggest_int('num_round', 2, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1E-3, 1),
                'max_bin': trial.suggest_int('max_bin', 2, 1000),
                'max_depth': trial.suggest_int('max_depth', 1, 8),
                'alpha': trial.suggest_float('alpha', 1E-16, 12),
                'gamma': trial.suggest_float('gamma', 1E-16, 12),
                'reg_lambda': trial.suggest_float('reg_lambda', 1E-16, 12),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 1E-16, 1.0),
                'subsample': trial.suggest_float('subsample', 1E-16, 1.0), 
                'min_child_weight': trial.suggest_float('min_child_weight', 1E-16, 12),
                'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 15),       
                }
    
    xgb_params = xgb_params | STATIC_PARAMS
        
   #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "evaluation-auc")
    
    if OPTUNA_CV == "StratifiedKFold": 
        kf = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    elif OPTUNA_CV == "TimeSeriesSplit":
        kf = TimeSeriesSplit(n_splits=OPTUNA_FOLDS)
    

    for f, (train_ind, val_ind) in (enumerate(kf.split(train, target))):

        train_df, val_df = train.iloc[train_ind], train.iloc[val_ind]
        
        train_target, val_target = target[train_ind], target[val_ind]

        train_dmatrix = xgb.DMatrix(train_df, label=train_target,enable_categorical=ENABLE_CATEGORICAL)
        val_dmatrix = xgb.DMatrix(val_df, label=val_target,enable_categorical=ENABLE_CATEGORICAL)


        model =  xgb.train(xgb_params, 
                           train_dmatrix, 
                           num_boost_round = NUM_BOOST_ROUND,
                           #callbacks=[pruning_callback],
                          )

        temp_oof = model.predict(val_dmatrix)

        train_oof[val_ind] = temp_oof

        #print(roc_auc_score(val_target, temp_oof))
    
    val_score = roc_auc_score(target, train_oof)
    
    return val_score


def LGB_objective(trial, train, target, category_columns, STATIC_PARAMS, ENABLE_CATEGORICAL, NUM_BOOST_ROUND, OPTUNA_CV, OPTUNA_FOLDS, SEED, EARLY_STOPPING):


    
    train_oof = np.zeros((train.shape[0],))
    
    
    lgb_params= {
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 0.5),
                "max_depth": trial.suggest_categorical('max_depth', [5,10,20,40,100, -1]),
                "n_estimators": trial.suggest_int("n_estimators", 50, 10000),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
                "cat_smooth" : trial.suggest_int('min_data_per_groups', 1, 100)
                }

    lgb_params = lgb_params | STATIC_PARAMS
        
    #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    
    if OPTUNA_CV == "StratifiedKFold": 
        kf = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    elif OPTUNA_CV == "TimeSeriesSplit":
        kf = TimeSeriesSplit(n_splits=OPTUNA_FOLDS)
    

    for f, (train_ind, val_ind) in (enumerate(kf.split(train, target))):

        train_df, val_df = train.iloc[train_ind], train.iloc[val_ind]
        
        train_target, val_target = target[train_ind], target[val_ind]

        train_lgbdataset = lgb.Dataset(train_df, label=train_target,categorical_feature=category_columns)
        val_lgbdataset = lgb.Dataset(val_df, label=val_target, reference = train_lgbdataset, categorical_feature=category_columns)


        model =  lgb.train(lgb_params, 
                           train_lgbdataset,
                           valid_sets=val_lgbdataset,
                           #num_boost_round = NUM_BOOST_ROUND,
                           callbacks=[#log_evaluation(LOG_EVALUATION),
                                      early_stopping(EARLY_STOPPING,verbose=False),
                                      #pruning_callback,
                                    ]               
                           #verbose_eval= VERBOSE_EVAL,
                          )

        temp_oof = model.predict(val_df)

        train_oof[val_ind] = temp_oof

        #print(roc_auc_score(val_target, temp_oof))
    
    val_score = roc_auc_score(target, train_oof)
    
    return val_score