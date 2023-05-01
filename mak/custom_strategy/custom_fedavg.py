import flwr as fl
import string
import csv
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
)

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self,out_file_path,fraction_fit,fraction_eval,min_fit_clients,min_eval_clients,min_available_clients,eval_fn,initial_parameters,on_fit_config_fn = None) -> None:
        super().__init__(fraction_fit,fraction_eval,min_fit_clients,min_eval_clients,min_available_clients,eval_fn,initial_parameters,on_fit_config_fn)
        self.fraction_fit = fraction_fit
        self.eval_fn = eval_fn
        self.initial_parameters = initial_parameters
        self.out_file_path = out_file_path
        self.on_fit_config_fn = on_fit_config_fn

    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        # for _, fit_res in results:
            # print(f"Round : {rnd}  time : {fit_res.metrics['time']}")
        return aggregated_weights


    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float],time:Optional[float]
    ) -> bool:
    #save the progress as csv file
        print("Accuracy Conclude ",acc)
        acc = acc['accuracy'] if acc is not None else None
        print("Accuracy ",acc)
        if self.out_file_path is not None:
            field_names = ["round","accuracy","loss","time"]
            dict = {"round": rnd,"accuracy":acc,"loss":loss,"time":time}
            with open(self.out_file_path,'a') as f:
                dictwriter_object = csv.DictWriter(f, fieldnames=field_names)
                dictwriter_object.writerow(dict)
                f.close()
        
        # print("Accuarccy : {} type : {} ".format(acc,type(acc)))
        if (acc >= 0.91):
            print("Reached specified accuracy so stopping further rounds")
            return False
        return True