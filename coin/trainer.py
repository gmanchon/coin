
from coin.mlflowbase import MLFlowBase
from coin.combi import expand_combi_list, build_ignore_keys, build_ignore_key

from itertools import product

from termcolor import colored


class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            "[FR] [Paris] [bitcoin] taxifare + 1",
            "https://mlflow.lewagon.co")

    def push_to_mlflow(self, exp_params, model_name, hexp_params, score):

        # create a mlflow training
        self.mlflow_create_run()  # create one training

        # log trainer params
        for key, value in exp_params.items():
            self.mlflow_log_param(key, value)

        # log model
        self.mlflow_log_param("model", model_name)

        # log model hyper params
        for key, value in hexp_params.items():
            self.mlflow_log_param(key, value)

        # log training metrics
        self.mlflow_log_metric("score", score)

    def train(self, trainer_params, hyper_params, ignored_combinations):

        # convert the list of ignored combinations into a list of ignored combination keys
        ignored_combi = expand_combi_list(ignored_combinations)
        all_ignored = build_ignore_keys(ignored_combi)

        print("\nignored combinations:")
        [print(colored(f"- {ignored}", "red")) for ignored in all_ignored]

        train_number = 0

        # step 1 : iterate on trainer params
        for param_combination in product(*trainer_params.values()):

            exp_params = dict(zip(trainer_params.keys(), param_combination))

            # print(exp_params)

            # build ignore key
            ignore_key = build_ignore_key(exp_params)

            # skip ignored combinations
            if ignore_key in all_ignored:

                print(colored(f"\nignore combi {ignore_key}", "blue"))

                # skip training
                continue

            print(colored(f"\ntrain for combi {ignore_key}", "green"))

            # step 2 : iterate on models
            for model_name, model_hparams in hyper_params.items():

                # print(f"model name {model_name}")

                # step 3 : iterate on model hyperparams
                for hparam_combi in product(*model_hparams.values()):

                    hexp_params = dict(zip(model_hparams.keys(), hparam_combi))

                    # print(hexp_params)

                    # list of training params + model + hyper params
                    train_number += 1
                    print(f"\ntraining #{train_number}:")
                    print(exp_params)
                    print(f"model name {model_name}")
                    print(hexp_params)

                    # TODO: train with trainer params + model + hyper params
                    # cros_val(model=model, **exp_params, hparams=hexp_params)

                    # TODO: process score
                    score = 123

                    # log training on mlflow
                    self.push_to_mlflow(exp_params, model_name, hexp_params, score)
