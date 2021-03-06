#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import (bytes, dict, int, list, object, range, str,  # noqa
    ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict,

from future.utils import viewitems  # noqa

from pugnlp import regexes
from nlpia.web import requests_get


def generate_download_mccauley():
    # amazon product reviews for recommendation engine training and review sentiment analysis
    response = requests_get('http://jmcauley.ucsd.edu/data/amazon/', allow_redirects=True, timeout=5)
    urls_product_review = [m[0] for m in regexes.cre_url_popular.findall(response.text) if m[0].lower().endswith('.json.gz')]

    response = requests_get('http://jmcauley.ucsd.edu/data/amazon/qa/', allow_redirects=True, timeout=5)
    urls_question_answer = [m[1] for m in regexes.cre_href.findall(response.text) if m[1].lower().endswith('.json.gz')]

    with open('download_mccauley_autogenerated.sh', 'w') as f:
        for pr in urls_product_review:
            f.write('wget ' + pr)
        for qa in urls_question_answer:
            f.write('wget ' + qa)
