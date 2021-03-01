
from itertools import product


def expand_combi_list(param_combi_list):

    result = []

    for param_combi_dict in param_combi_list:

        for param_combi in product(*param_combi_dict.values()):

            exp_params = dict(zip(param_combi_dict.keys(), param_combi))

            result.append(exp_params)

    return result


def build_ignore_key(param):

    # sorting keys just in case
    param_keys = list(reversed(sorted(param.keys())))

    # building key value strings
    param_list = "_".join([f"{k}-{param[k]}" for k in param_keys])

    # returning global key
    return param_list


def build_ignore_keys(param_list):

    return [build_ignore_key(param) for param in param_list]


if __name__ == '__main__':
    ignored_combinations = [
        dict(
            sample_size=[1_000, 100_000],
            shift_size=[5],
            train_size=[2],
            horizon=[1_000],
            data_start=[1_000],
            data_end=[1_000],
        ),
        dict(
            sample_size=[200_000],
            shift_size=[100],
            train_size=[2],
            horizon=[1_000],
            data_start=[1_000],
            data_end=[1_000],
        ),
    ]

    res = expand_combi_list(ignored_combinations)
    print(res)

    res = build_ignore_keys(res)
    print(res)
