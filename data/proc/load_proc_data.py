import os
import pandas as pd
from collections import OrderedDict
from typing import List
import numpy as np


def merge(dic1, dic2):
    '''
    Returns an OrderedDict whose keys are all those of dic1 and/or dic2,
    and whose values are those in dic2 or (if missing from dic2) dic1.
    '''
    return OrderedDict(dic1, **dic2)


def find_conditions(expanded, conditions):
    '''
    Returns the indices of expanded that only have zero values for unspecified conditions.
    '''
    treatments = list(expanded[0].keys())
    removes = list(set(treatments) - set(conditions))
    locs = [i for i, ex in enumerate(expanded) if all([ex[r] == 0.0 for r in removes])]
    filtered = [OrderedDict((k, ex[k]) for k in conditions) for ex in expanded[locs]]
    return locs, filtered


def process_condition(row):
    '''
    row: a string of the form 'a=b;c=d;...' where each RHS can be converted to float.
    Returns: a dictionary derived from row, plus the key 'Device' with value device.
    '''
    d = OrderedDict()
    if '=' not in row:
        return d
    conditions = row.split(';')
    for cond in conditions:
        els = cond.split('=')
        d[els[0]] = float(els[1])
    return d


def expand_conditions(treatments: List[OrderedDict], conditions):
    '''
    Given a list of "conds", returns a list of dicts, each of which
    is the corresponding member of "conds" expanded with "key: 0.0" members
    so that all the returned dicts have the same set of keys.
    '''
    # Establish all treatments
    zero = OrderedDict()
    for cond in conditions:
        zero[cond] = 0.0
        # Now fill each condition
    return np.array([merge(zero, tr) for tr in treatments])


def extract_signal(s):
    '''
    Returns the portion of s between the (first) pair of parentheses,
    or the whole of s if there are no parentheses.
    '''
    loc0 = s.find("(")
    if loc0 >= 0:
        loc1 = s.find(")")
        if loc1 >= 0:
            return s[loc0 + 1:s.find(")")]
    return s


def load(csv_file, args):
    '''
    Args:
        csv_file (string): Local file name that is to be loaded, with headers, consisting of:
            Content   Device, e.g. R33S32_Y81C76
            Colony    (blank)
            Well Col  int, 1 to 12
            Well Row  letter, A to H
            Content   Condition, e.g. C6=<float> or C12=<float> or EtOH=<float>
            and then for each of EYFP, ECFP, mRFP1 and OD, 100 readings at different times
            and then second line is "timesall": time of each col except for the first 5
    Returns:
        devices
        treatments
        times
        observations
    '''

    data_path = os.path.join(args.data_path, csv_file)
    loaded = pd.read_csv(data_path, sep=',', na_filter=False)
    timesall = loaded.iloc[0, 5:]  # times of the observations
    obs_rows = loaded.iloc[1:, :]  # observation rows
    # Rows we want to keep are those whose first ("Content") value is in the "devices" list.
    rows = obs_rows.iloc[np.isin(obs_rows.iloc[:, 0], args.data.devices), :]

    # Create devices
    devices = np.array([args.data.device_map[dev] for dev in rows.iloc[:, 0]], dtype=int)

    # List of OrderedDicts, each with keys C6 or C12 (i.e. the two "content" columns above)
    # and float values.
    treatment_values = [process_condition(cond) for cond in rows.iloc[:, 4]]
    # print(treatment_values)
    if len(treatment_values) is 0:
        return None  # flag value to indicate the dataset doesn't exist in this file

    # As treatment_values, but each OrderedDict additionally has the keys that the others have, with value 0.0.
    expanded = expand_conditions(treatment_values, args.data.conditions)

    # Filter out time-series that have nonzero values for unspecified conditions
    locs, filtered = find_conditions(expanded, args.data.conditions)
    treatments = np.array([list(cond.values()) for cond in filtered])

    # Collect the time-series observations
    X = rows.iloc[locs, 5:]
    headers = np.array([v.split('.')[0] for v in X.columns.values])
    header_signals = np.array([extract_signal(h) for h in headers])
    x_values = [[row.iloc[header_signals == signal].values for signal in args.data.signals] for idx, row in
                X.iterrows()]
    observations = np.array(x_values)
    times = timesall.iloc[header_signals == 'OD'].values

    if args.data.dtype == 'float32':
        return devices, treatments.astype(np.float32), times.astype(np.float32), observations.astype(np.float32)
    elif args.data.dtype == 'float64':
        return devices, treatments.astype(np.float64), times.astype(np.float64), observations.astype(np.float64)
    else:
        raise Exception("Unknown dtype %s" % args.data.dtype)
