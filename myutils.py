import pandas as pd
import numpy as np
import os, random
import time, datetime
from contextlib import contextmanager
from tqdm import tqdm

from sklearn.metrics import roc_auc_score,mean_squared_error,average_precision_score,log_loss
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold

import lightgbm as lgb

import argparse

from colorama import Fore
y_=Fore.YELLOW; b_=Fore.BLUE; g_=Fore.GREEN; sr_=Fore.RESET


# 
id_name = "index"
label_name = "target"
print(f"{g_}*** id_name:{id_name} // label_name:{label_name}")


# コマンドライン引数の設定 --
# ## myutilsをimportしてる.pyのコマンドラインからでも行ける --
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='./input/')
parser.add_argument("--save_dir", type=str, default='tmp')
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--do_train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--remark", type=str, default='')
args, unknown = parser.parse_known_args()

# 乱数の固定 --
def Seed_no_torch(seed=42) -> None:
    print(f"{g_}*** Seed_no_torch *** {sr_}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
Seed_no_torch()


@contextmanager
def Timer(title):
    """
    with文の時間計測
    
    ex.)
        def wait(sec: float):
            time.sleep(sec)

        with Timer("wait"):
            wait(2.0)  # => wait - done in 2 s --
    """
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"(title, (datetime.datetime.now() - t0).seconds))
    return None


def Metric(labels, preds):
    return log_loss(labels, preds)


def Write_log(logFile, text, isPrint=True):
    if isPrint: print(text)
    logFile.write(text)
    logFile.write("\n")
    return None


def Lgb_train_and_predict(train, test, config, cv="kf", aug=None, output_root="./output/", run_id=None):
    """
    パッとlgbのベースライン作成ができる関数
    これ、objectiveがcategory/regressionを一つの関数で実装するのややこしいか？
    """
    # run_idの作成と実験管理 --
    if not run_id:
        run_id = "run_lgb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        while os.path.exists(output_root+run_id+"/"):
            time.sleep(1)
            run_id = "run_lgb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_root + f"{args.save_dir}"
    else:
        output_path = output_root + run_id + "/"

    if not os.path.exists(output_path): os.mkdir(output_path)

    # 今使用したコード類をoutputにコピペする --
    os.system(f"cp ./*.py {output_path}")
    os.system(f"cp ./*.sh {output_path}")
    os.system(f"cp ./*.ipynb {output_path}")

    # lgbで使用するseedをグローバルのseedとする --
    config["lgb_params"]["seed"] = config["seed"]

    features = config["feature_name"]
    params = config["lgb_params"]
    rounds = config["rounds"]
    verbose = config["verbose_eval"]
    early_stopping_rounds = config["early_stopping_rounds"]
    folds = config["folds"]
    seed = config["seed"]

    oof, sub = None, None
    oof = train[[id_name]]
    oof[label_name] = 0

    # trainセクション --
    if train is not None:
        log = open(output_path + "/train.log", "w", buffering=1)
        log.write(str(config)+"\n")
        
        all_valid_metric, feature_importance = [], []

        # make cv --
        if cv=="kf":
            kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
            split = kf.split(train)

        elif cv=="skf":
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            split = skf.split(train, train[label_name])
        
        for fold, (trn_index, val_index) in enumerate(split):
            evals_result_dic = {}

            if aug:
                print(f"{y_} aug NOT implemented --")
                trn_data = lgb.Dataset(train.loc[trn_index, features], label=train.loc[trn_index, label_name])
            else:
                trn_data = lgb.Dataset(train.loc[trn_index, features], label=train.loc[trn_index, label_name])
            val_data = lgb.Dataset(train.loc[val_index, features], label=train.loc[val_index, label_name])

            model = lgb.train(
                params,
                train_set = trn_data,
                num_boost_round = rounds,
                valid_sets = [trn_data, val_data],
                evals_result = evals_result_dic,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose
            )
            model.save_model(output_path + "/fold%s.ckpt"%fold)

            valid_preds = model.predict(train.loc[val_index, features], num_iteration=model.best_iteration)
            oof.loc[val_index, label_name] = valid_preds

            for i in range(len(evals_result_dic['valid_1'][params['metric']])//verbose):
                Write_log(log, ' - %i round - train_metric: %.6f - valid_metric: %.6f\n'%(i*verbose, evals_result_dic['training'][params['metric']][i*verbose], evals_result_dic['valid_1'][params['metric']][i*verbose]))
            all_valid_metric.append(Metric(train.loc[val_index,label_name], valid_preds))
            Write_log(log,'*** - fold%s valid metric: %.6f\n'%(fold,all_valid_metric[-1]))

            importance_gain = model.feature_importance(importance_type="gain")
            importance_split = model.feature_importance(importance_type="split")
            feature_name = model.feature_name()
            feature_importance.append(
                pd.DataFrame({"feature_name":feature_name, "importance_gain":importance_gain, "importance_split":importance_split})
            )

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(["feature_name"]).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=["importance_gain"], ascending=False)
        feature_importance_df.to_csv(output_path + "/feature_importance.csv", index=False)

        mean_valid_metric = np.mean(all_valid_metric)
        all_valid_metric_std = np.std(all_valid_metric)
        global_valid_metric = Metric(train[label_name].values, oof[label_name].values)
        Write_log(log, "all valid metric (mean, std):(%.6f, %.6f) ..//.. global valid metric:%.6f"%(mean_valid_metric, all_valid_metric_std, global_valid_metric))
        
        oof.to_csv(output_path + "/oof.csv", index=False)

        log.close()
        os.rename(output_path + "/train.log", output_path+"/train_%.6f.log"%mean_valid_metric)

        log_df = pd.DataFrame({
            "run_id":[run_id], 
            "mean metric":[round(mean_valid_metric, 6)],
            "fold std": [round(all_valid_metric_std, 6)],
            "global metric": [round(global_valid_metric, 6)],
            "remark": [args.remark]
            })
        if not os.path.exists(output_root + "/experiment_log.csv"):
            log_df.to_csv(output_root + "/experiment_log.csv", index=False)
        else:
            log_df.to_csv(output_root + "/experiment_log.csv", index=False, header=None, mode="a")

    # testデータへのpreds作成 --
    if test is not None:
        sub = test[[id_name]]
        sub["prediction"] = 0
        for fold in range(folds):
            model = lgb.Booster(model_file=output_path+"/fold%s.ckpt"%fold)
            test_preds = model.predict(test[features], num_iteration=model.best_iteration)
            sub["prediction"] += (test_preds/folds)
        sub[[id_name, "prediction"]].to_csv(output_path+"/submission.csv.zip", compression="zip", index=False)
    
    if args.save_dir in output_path:
        os.rename(output_path, output_root+run_id+"/")

    return oof, sub, (mean_valid_metric, global_valid_metric)


