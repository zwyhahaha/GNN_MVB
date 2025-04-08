# -*- coding: utf-8 -*-
# @author: qiufangkun
# @email: 995916989@qq.com
# @date: 2020/07/03
import pandas as pd
from functools import wraps
import time
import os
import warnings
warnings.filterwarnings(action="ignore")
import sys
model_path = os.path.dirname(__file__)
sys.path.append(model_path)
data_path = os.path.join(model_path, "../data/sichuan")

# generator = [4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31,
#              32, 34, 36, 40, 42, 46, 49, 54, 55, 56, 59, 61, 62,
#              65, 66, 69, 70, 72, 73, 74, 76, 77, 80, 82, 85, 87,
#              89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110, 111, 112,
#              113, 116]
generators = [126, 129, 132, 133, 172, 233, 234, 236, 237, 248, 270, 320, 694, 799, 800, 801, 905, 906]


def func_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print("====================================")
        print('Function: {name} start...'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(
            'Function: {name} finished, spent time: {time:.2f}s'.format(
                name=function.__name__,
                time=t1 - t0))
        print("====================================")
        return result

    return function_timer


class data_reading:
    def __init__(self, s_data_file="thermal_units3.csv", s_bid_file="bid_v2.csv",
                 h_data_file="hydro_units3.csv", demand_file="day96_v2.csv",
                 h_connect_data_file="hydro_connect_v1.csv", method="coef", river=None):
        if river is None:
            river = [2, 3, 4, 5, 6, 7, 8]
        self.generators = generators
        self.T = 96
        self.stages = 10
        self.need_river = river
        s_data_file, s_bid_file, demand_file, h_data_file, h_connect_data_file =\
            [os.path.join(data_path, path)
             for path in [s_data_file, s_bid_file, demand_file, h_data_file, h_connect_data_file]]
        self._read_data(s_data_file, s_bid_file, demand_file, h_data_file, h_connect_data_file)
        self._calculate_piecewise(method)

    def _read_data(self, s_data_file, s_bid_file, demand_file, h_data_file, h_connect_data_file):
        self.s_data = pd.read_csv(s_data_file, index_col="id")
        self.D = pd.read_csv(demand_file, index_col=None)
        self.cost = pd.read_csv(s_bid_file, index_col="id")
        self.H_data = pd.read_csv(h_data_file, index_col="id")
        self.H_data.query("river in @self.need_river", inplace=True)
        self.H_generators = self.H_data.index.unique().to_list()
        self.H_connect_data = pd.read_csv(h_connect_data_file, index_col="id").rename(columns={"from": "from_"})
        self.H_connect_data["delay"] = self.H_connect_data["delay"] * 4
        self.H_connect_data.delay = self.H_connect_data.delay.apply(round)
        self.H_connect_data = self.H_connect_data.query("to in @self.H_generators and from_ in @self.H_generators")

    def _calculate_piecewise(self, method="coef"):

        def _get_R_cost(df, _bid):
            index_ = df.generator.unique()[0]
            v = _bid[index_].iloc[0]
            df['R'] = v / 2
            df['Rs'] = v / 4
            df['R1'] = v / 8
            df['R3'] = v / 8
            return df

        def _calculate_coefs(df, _m, _bid):
            # 计算每一段斜率，截距
            index_ = df.generator.unique()[0]
            m_ = _m[index_]
            bid_ = _bid[index_].values
            df["inter"] = (- m_.cumsum().shift(1) * bid_
                           + (m_ * bid_).cumsum().shift(1)).values
            return df

        def _calculate_points(df, _m, _bid):
            # 计算每一段端点，成本
            index_ = df.generator.unique()[0]
            m_ = _m[index_]
            bid_ = _bid[index_].values
            df.loc[:, "C"].iloc[1:] = (m_ * bid_).cumsum().values
            df.loc[:, "delta"].iloc[1:] = m_.cumsum().values
            return df

        self.bid = self.cost.loc[:, [f"bid{s}" for s in range(1, 11)]].stack()
        self.m = self.cost.loc[:, [f"m{s}" for s in range(1, 11)]].stack()
        self.piecewise_matrix = None
        self.R_bid = pd.DataFrame(index=pd.Index(self.generators, name="generator"),
                                  columns=["R", "Rs", "R1", "R3"]).reset_index()

        if method == "coef":
            self.piecewise_matrix = pd.DataFrame(
                index=pd.MultiIndex.from_product([self.generators, range(self.stages)],
                                                 names=["generator", "stage"]),
                columns=["coef", "inter"]).reset_index()

            self.piecewise_matrix["coef"] = self.bid.values
            self.piecewise_matrix = self.piecewise_matrix.groupby(["generator"]).apply(_calculate_coefs,
                                                                                       _m=self.m, _bid=self.bid)

        if method == "points":
            self.piecewise_matrix = pd.DataFrame(
                index=pd.MultiIndex.from_product([self.generators, range(self.stages + 1)],
                                                 names=["generator", "stage"]),
                columns=["delta", "C"]).reset_index()

            self.piecewise_matrix = self.piecewise_matrix.groupby(["generator"]).apply(_calculate_points,
                                                                                       _m=self.m, _bid=self.bid)

        self.piecewise_matrix = self.piecewise_matrix.fillna(0).set_index(["generator", "stage"])
        self.piecewise_matrix.fillna(0, inplace=True)
        self.R_bid = self.R_bid.groupby(["generator"]).apply(_get_R_cost, _bid=self.bid).set_index(["generator"])

    # def read_data():
    #     demand = pd.read_csv("hour24.csv", usecols=["Hour", "LoadT", "Rr", "Rs", "R1", "R3"])
    #     demand.Hour = demand.Hour - 1
    #     demand.set_index("Hour", inplace=True)
    #     cost = pd.read_csv("hourIndex118.csv", index_col=[0]).loc[generator]
    #     cost.hour -= 1
    #     cost.set_index("hour", append=True, inplace=True)
    #     name_dict = {"MinOn": "T_i_on", "MinOff": "T_i_off", "StartUp": "SU", "Ramp": "RU", "IniProd": "NS"}
    #     generator_data = pd.read_csv("IEEE118_node.csv", index_col=[0]).rename(columns=name_dict)
    #     generator_data = generator_data.loc[generator]
    #     limit = generator_data.loc[:, ["Pmax", "Pmin"]]
    #     lasttime_on = generator_data.loc[:, ["T_i_on"]]
    #     lasttime_off = generator_data.loc[:, ["T_i_off"]]
    #     SU = generator_data.loc[:, ['SU']]
    #     SD = SU.rename(columns={'SU': 'SD'})
    #     # SD.loc[:] = 100
    #     RU = generator_data.loc[:, ['RU']]
    #     RD = RU.rename(columns={'RU': 'RD'})
    #     generator_data.loc[:, ['NS']] = 24
    #     NS = generator_data.loc[:, ['NS']]
    #     ND = NS.rename(columns={'NS': 'ND'})
    #
    #     return demand, cost, limit, lasttime_on, lasttime_off, SU, SD, RU, RD, NS, ND
# Test_R = data_reading()
# Test_R.cost.to_csv("check_bid.csv")
# print(Test_R.cost)
