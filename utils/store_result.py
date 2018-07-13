# -*- coding: utf-8 -*-  
"""  
 @version: python2.7 
 @author: luofuli 
 @time: 2018/4/1 19:56
"""

import os

import path

_path = path.WSD_path()


def write_result(results, back_off_result, path, print_logs=True):
    if print_logs:
        print('Writing to file:%s' % path)
    new_results = results + back_off_result
    new_results = sorted(new_results, key=lambda a: a[0])
    with open(path, 'w') as file:
        for instance_id, predicted_sense in new_results:
            file.write('%s %s\n' % (instance_id, predicted_sense))
