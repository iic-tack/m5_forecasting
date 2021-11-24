import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import time
import yaml
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

with open('train_pred.yaml') as file:
    yml = yaml.safe_load(file)
print('params read.')

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
    id_ = yml["id_"]
    print(f"id: {id_}")
    if slack_notify:
        slack.notify(text=f"*train_pred.py({id_}) is on {machine}!*")
    # loading data
    data_ex = pd.read_csv(yml["csv"], dtype=yml["dtyp"])
    train = data_ex[(yml["data"]["train_from"] <= data_ex["date_id"]) & (data_ex["date_id"] <= 1913)]
    test = data_ex[(1914 <= data_ex["date_id"]) & (data_ex["date_id"] <= 1941)]
    subm = data_ex[1942 <= data_ex["date_id"]]
    del data_ex
    print('data loaded.')

    # define features and params 
    features = yml["features"]
    params = yml["params"]
    if yml["gpu"]:
        params.update({'device': 'gpu'})

    # dataset
    train_set = lgb.Dataset(train[features], train['num'], free_raw_data=False)
    test_set = lgb.Dataset(test[features], test['num'], free_raw_data=False)
    del train

    # train
    st = time.time()
    results={}
    num_boost_round = 100000
    early_stopping_rounds = 50
    model = lgb.train(  params,
                        train_set, 
                        num_boost_round = yml["model"]["num_boost_round"],
                        early_stopping_rounds = yml["model"]["early_stopping_rounds"], 
                        valid_sets = [train_set, test_set], 
                        valid_names=["train", "test"], 
                        verbose_eval = yml["model"]["verbose_eval"],
                        evals_result=results)
    print(f"time: {time.time()-st:.2f} [sec]")

    ## plot
    loss_train = results['train']['rmse']
    loss_test = results['test']['rmse']

    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('loss')
    ax.plot(loss_train, label='train loss')
    ax.plot(loss_test, label='test loss')
    plt.legend()
    fig.savefig(f"./output/img_{id_}.png")

    ## print scores
    best_iteration = model.best_iteration
    train_score = results['train']['rmse'][best_iteration-1]
    test_score = results['test']['rmse'][best_iteration-1]
    print(f"best iteration: {best_iteration}")
    print(f'best rmse score (train): {train_score}')
    print(f'best rmse score (test) : {test_score}')

    # estimation for val data
    y_pred = model.predict(test[features])
    test_pred = test.copy(deep=True)
    test_pred['num'] = y_pred
    out_valid = test_pred.pivot(index="id", columns="date_id", values="num").reset_index()
    out_valid.columns = ["id"]+[f"F{str(i+1)}" for i in range(28)]
    del test_pred

    # estimation for eval data
    y_pred = model.predict(subm[features])
    subm['num'] = y_pred
    out_eval = subm.pivot(index="id", columns="date_id", values="num").reset_index()
    out_eval.columns = ["id"]+[f"F{str(i+1)}" for i in range(28)]


    # generate submission file
    out_sub = pd.read_csv("./data/sample_submission.csv")

    out_sub.set_index("id", drop=False, inplace=True)
    out_valid.set_index("id", drop=False, inplace=True)
    out_eval.set_index("id", drop=False, inplace=True)

    out_sub.update(out_valid)
    out_sub.update(out_eval)


    # report
    flag = 1
    i = 0
    with open("./report.txt") as f:
        lines = f.readlines()
        while flag==1:
            flag = 0
            for l in lines:
                if l.startswith(id_):
                    print(f'ID:{id_} has already been used!')
                    flag = 1
                    id_ = f"{id_}({i})"
                    break
            i += 1
    ## save logs
    f = open('./report.txt', 'a')
    f.write(f'{id_}: comment  ... {yml["comment"]}\n')
    f.write(f'{id_}: features ... {features}\n')
    f.write(f'{id_}: params1  ... {params}\n')
    f.write(f'{id_}: score    ... iter={best_iteration}, train={train_score}, test={test_score}\n')
    f.close()
    ## save submission files
    out_sub.to_csv(f'./submission/submission_{id_}.csv', index=False)
    ## save scores
    tmp = pd.read_csv('./scores.csv')
    tmp_se = pd.Series([id_, best_iteration, train_score, test_score], index=tmp.columns)
    tmp = tmp.append(tmp_se, ignore_index=True)
    tmp.to_csv("./scores.csv", index=False)
    ## save models
    model.save_model(f'./models/lgb_{id_}.txt', num_iteration=model.best_iteration)

    if slack_notify:
        slack.notify(text=f"*train_pred.py({id_}, on {machine}) finished!*")

except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
    if slack_notify:
        slack.notify(text=f"*ERROR HAS CAUSED AT train_pred.py({id_})!:*\n{e}")