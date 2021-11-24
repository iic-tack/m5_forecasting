import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import lightgbm as lgb
from sklearn import metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

with open('params_tuning.yaml') as file:
    yml = yaml.safe_load(file)

id_ = yml["id_"]
slack_notify = yml["slack_notify"]
if slack_notify:
    import sys
    import os
    sys.path.append("../")
    import slackweb
    from slack_url import slack_url
    slack = slackweb.Slack(url=slack_url())
    machine = os.uname()[1]

try:
    # define features and params 
    features = yml["features"]
    # params = yml["params"]

    # loading data
    mx = np.array(yml["data"]["cr_val_list"]).max()
    mn = np.array(yml["data"]["cr_val_list"]).min()
    data_ex = pd.read_csv(yml["csv"])
    tmp = data_ex[(mn <= data_ex["date_id"]) & (data_ex["date_id"] <= mx + 28)]
    del data_ex
    print("main data read.")

    # dataset
    tr_set = []
    for i, date_id in enumerate(yml["data"]["cr_val_list"]):
        train = tmp[(date_id[0] <= tmp["date_id"]) & (tmp["date_id"] <= date_id[1])]
        test = tmp[(date_id[1] + 1 <= tmp["date_id"]) & (tmp["date_id"] <= date_id[1] + 28)]
        train_set = lgb.Dataset(train[features], train['num'], free_raw_data=False)
        test_set = lgb.Dataset(test[features], test['num'], free_raw_data=False)
        tr_set.append((train_set, test_set, test))
    del tmp
    del train
    print("dataset defined.")

    # for train
    def score(params):
        params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
        params["max_depth"] = int(params["max_depth"])
        params["num_leaves"] = int(params["num_leaves"])
        params["bagging_freq"] = int(params["bagging_freq"])

        tmpscore = 0
        for train_set, test_set, test in tr_set:
            model = lgb.train(  params,
                                train_set, 
                                num_boost_round = yml["model"]["num_boost_round"],
                                early_stopping_rounds = yml["model"]["early_stopping_rounds"], 
                                valid_sets = [train_set, test_set], 
                                valid_names=["train", "test"], 
                                verbose_eval = yml["model"]["verbose_eval"])
            pred = model.predict(test[features])
            score = np.sqrt(metrics.mean_squared_error(pred, test["num"]))
            history.append((params, score))
            tmpscore += score
        tmpscore /= len(tr_set)
        return {"loss":tmpscore, "status":STATUS_OK}

    # explore range for params
    param_space = {
        'metrics': 'rmse',
        'max_bin': yml["param_space"]["max_bin"],
        'learning_rate': yml["param_space"]["learning_rate"],
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 
                                        yml["param_space"]["min_data_in_leaf"][0], 
                                        yml["param_space"]["min_data_in_leaf"][1], 
                                        yml["param_space"]["min_data_in_leaf"][2]),
        'num_leaves': hp.quniform('num_leaves', 
                                        yml["param_space"]["num_leaves"][0], 
                                        yml["param_space"]["num_leaves"][1], 
                                        yml["param_space"]["num_leaves"][2]),
        'colsample_bytree': hp.quniform('colsample_bytree', 
                                        yml["param_space"]["colsample_bytree"][0], 
                                        yml["param_space"]["colsample_bytree"][1], 
                                        yml["param_space"]["colsample_bytree"][2]),
        'max_depth': hp.quniform('max_depth', 
                                        yml["param_space"]["max_depth"][0], 
                                        yml["param_space"]["max_depth"][1], 
                                        yml["param_space"]["max_depth"][2]),
        'bagging_fraction': hp.quniform('bagging_fraction', 
                                        yml["param_space"]["bagging_fraction"][0], 
                                        yml["param_space"]["bagging_fraction"][1], 
                                        yml["param_space"]["bagging_fraction"][2]),
        'bagging_freq': hp.quniform('bagging_freq', 
                                        yml["param_space"]["bagging_freq"][0], 
                                        yml["param_space"]["bagging_freq"][1], 
                                        yml["param_space"]["bagging_freq"][2])}
    if yml["param_space"]["lambda"]:
        param_space.update(
            {'lambda_l1' : hp.loguniform('lambda_l1', np.log(1e-8), np.log(1.0)),
             'lambda_l2' : hp.loguniform('lambda_l2', np.log(1e-6), np.log(10.0))}
        )
    if yml["param_space"]["gpu"]:
        param_space.update({'device': 'gpu'})


    # params exploring by hyperopt
    if slack_notify:
        slack.notify(text=f"*params_tuning.py({id_}) starts exploring best params in {machine}!*")
    max_evals = yml["hyperopt"]["max_evals"]
    trials = Trials()
    history = []
    fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

    # get params and score
    history = sorted(history, key=lambda tpl:tpl[1])
    best=history[0]
    print(f"best params:{best[0]}, score:{best[1]:.4f}")
    if slack_notify:
        slack.notify(text=f"*params_tuning.py({id_}) has finished!:*\nbest params:{best[0]}, score:{best[1]:.4f}")

except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
    if slack_notify:
        slack.notify(text=f"*ERROR HAS CAUSED AT params_tuning.py({id_})!:*\n{e}")
        