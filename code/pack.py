import os
import pandas as pd
import numpy as np
import requests
import bs4
import json
import re
import time
import FinanceDataReader as fdr
import sys
import pickle
import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import stats
from scipy.stats import norm
from scipy.optimize import minimize
import empyrical as ep
import seaborn as sns
import yahoo_fin.stock_info as si
from selenium import webdriver