# -*- coding: utf-8 -*-
# @author: qiufangkun
# @email: 995916989@qq.com
# @date: 2020/07/07
import cplex
import pandas as pd
import docplex.mp.model as ml
from data_reading import data_reading


class SCUC_basic(data_reading):
    def __init__(self, problem=None, river=None):
        if problem is None:
            return
        super(SCUC_basic, self).__init__(river=river)
        self.over_all_cost = None
        self.penalty = None
        self.problem = problem
        self.quad_penalty = None
        if problem == "all":

            self.mip_model = ml.Model(name=f'SCUC_basic_{problem}')
            self.Ps = self.mip_model.continuous_var_cube(self.generators, range(self.T),
                                                         range(self.stages), lb=0, name="Ps")
            self.P = {(ge, t): self.mip_model.sum_vars([self.Ps[ge, t, stage] for stage in range(self.stages)])
                      for ge in self.generators for t in range(self.T)}
            self._general_variables()
            self._hydro_variables()
        elif problem in ["s_lr", "s_al"]:
            self.mip_model = ml.Model(name='SCUC_basic_sub_problem')
            self.Ps = self.mip_model.continuous_var_cube(self.generators, range(self.T),
                                                         range(self.stages), lb=0, name="Ps")
            self.P = {(ge, t): self.mip_model.sum_vars([self.Ps[ge, t, stage] for stage in range(self.stages)])
                      for ge in self.generators for t in range(self.T)}
            self._general_variables()
        elif problem == "h":
            self.mip_model = ml.Model(name='SCUC_hydro')
            self._hydro_variables()
        elif problem == "s_admm":
            self.mip_model = ml.Model(name='SCUC_basic_admm')
            self.Ps = self.mip_model.continuous_var_cube(self.generators, range(self.T),
                                                         range(self.stages), lb=0, name="Ps")
            self.P = {(ge, t): self.mip_model.sum_vars([self.Ps[ge, t, stage] for stage in range(self.stages)])
                      for ge in self.generators for t in range(self.T)}
            self._general_variables()
            self.D_R = self.mip_model.continuous_var_list(self.T, name="DR")
        elif problem == "h_admm":
            self.mip_model = ml.Model(name='SCUC_h_admm')
            self.D_R = self.mip_model.continuous_var_list(self.T, name="DR")
            self._hydro_variables()

    def _general_variables(self):
        """

        • The continuous variable Pi,t,j ∈ R+ represents the power generation in interval of [∆j−1,∆j], j = 1,2,...,J.
          It has Pi,t = 􏰂Jj=1 Pi,t,j; In the basic model, we use Pi,t,j as the decision variables instead of Pi,t.

        • Generator state variable: Ii,t ∈ {0, 1} is commitment decision for i ∈ G and t ∈ {1, · · · , T };
          if Ii,t = 1,the i-th generator is online;
             Ii,t = 0, otherwise;

        • Startup action: Wi,t ∈ {0,1} for i ∈ G and t ∈ {1,··· ,T}, represents the startup action;

        • Shutdown action: Yi,t ∈ {0, 1} for i ∈ G and t ∈ {1, · · · , T }, represents the shutdown action;

        • Rr ∈ R, Rsp ∈ R, Rn1 ∈ R, and Rn3 ∈ R are the amount of reserve for regulating, spinning,
          10-minute non-spinning and 30-minute non-spanning reserve.

        """
        self.I = self.mip_model.binary_var_matrix(self.generators, range(self.T), name='I')
        self.W = self.mip_model.continuous_var_matrix(self.generators, range(self.T), lb=0, name='W')
        self.Y = self.mip_model.continuous_var_matrix(self.generators, range(self.T), lb=0, name='Y')
        self.R = self.mip_model.continuous_var_matrix(self.generators, range(self.T), lb=0, name='Rg')
        self.Rs = self.mip_model.continuous_var_matrix(self.generators, range(self.T), lb=0, name='Rs')
        self.R1 = self.mip_model.continuous_var_matrix(self.generators, range(self.T), lb=0, name='R1')
        self.R3 = self.mip_model.continuous_var_matrix(self.generators, range(self.T), lb=0, name='R3')

    def _hydro_variables(self):
        """
        • PH represents the power generation from hydropower plant j at time t, for j ∈ H and t = j,t 0,1,...,T.

        • Xj,t is the water storage of hydropower plant j at time t.

        • Qj,t is the total water discharge of hydropower plant j at time t, and it is made of two components Q1j,t and
          Q2j,t. The former denotes the amount of water used for charging and the latter is the abandoned water.

        """
        self.Q_H_1 = self.mip_model.continuous_var_matrix(self.H_generators, range(self.T), lb=0, name="Q_H1")
        self.Q_H_2 = self.mip_model.continuous_var_matrix(self.H_generators, range(self.T), lb=0, name="Q_H2")
        self.X = self.mip_model.continuous_var_matrix(self.H_generators, range(self.T + 1),
                                                      lb=lambda keys: self.H_data.loc[keys[0], "level_min"],
                                                      ub=lambda keys: self.H_data.loc[keys[0], "level_max"],
                                                      name="X")
        self.P_H = self.mip_model.continuous_var_matrix(self.H_generators, range(self.T), lb=0, name="PH")

    def set_hydro_objective(self, shadow_price):

        self.calculate_demand_penalty(shadow_price)
        self.mip_model.set_objective("min", self.penalty)

    def set_s_objective(self, shadow_price=None):
        fuel_cost = self.mip_model.dotf(self.Ps,
                                        lambda keys: self.bid.loc[keys[0]].iloc[keys[2]])
        self.set_general_objective(fuel_cost, shadow_price)

    def set_general_objective(self, fuel_cost, shadow_price):
        start_shut_cost = self.mip_model.dotf(self.W, lambda keys: self.s_data.loc[keys[0], "SU"]) + \
                          self.mip_model.dotf(self.Y, lambda keys: self.s_data.loc[keys[0], "SD"])
        reserve_cost = self.mip_model.dotf(self.R, lambda keys: self.R_bid.loc[keys[0], "R"]) + \
            self.mip_model.dotf(self.Rs, lambda keys: self.R_bid.loc[keys[0], "Rs"]) + \
            self.mip_model.dotf(self.R1, lambda keys: self.R_bid.loc[keys[0], "R1"]) + \
            self.mip_model.dotf(self.R3, lambda keys: self.R_bid.loc[keys[0], "R3"])
        self.over_all_cost = fuel_cost + start_shut_cost + reserve_cost
        self.mip_model.set_objective("min", fuel_cost + start_shut_cost + reserve_cost)
        if shadow_price is not None:
            self.calculate_demand_penalty(shadow_price)
            self.mip_model.objective_expr += self.penalty

    def calculate_demand_penalty(self, shadow_price=None):
        if self.problem in ["h", "h_admm"]:
            self.penalty = self.mip_model.dotf(self.P_H, lambda keys: -shadow_price.loc[keys[1], "price"])
        elif self.problem in ["s_lr", "s_admm"]:
            self.penalty = self.mip_model.sum([self.P[ge, t] * -shadow_price.loc[t, "price"]
                                               for ge in self.generators for t in range(self.T)])

    def change_demand_penalty(self, shadow_price=None):
        if self.penalty is not None:
            self.mip_model.objective_expr.add(-self.penalty)
        self.calculate_demand_penalty(shadow_price)
        self.mip_model.objective_expr.add(self.penalty)

    def set_quad_penalty_objective(self, rol):
        if self.problem == "s_admm":
            self.quad_penalty = self.mip_model.sumsq(
                [self.D_R[t] - self.mip_model.sum(self.P[ge, t] for ge in self.generators) for t in range(self.T)]
            )
        elif self.problem == "h_admm":
            self.quad_penalty = self.mip_model.sumsq(
                [self.D_R[t] - self.mip_model.sum(self.P_H[ge, t] for ge in self.H_generators) for t in range(self.T)]
            )
        self.mip_model.objective_expr += 0.5 * rol * self.quad_penalty

    def set_objective(self, shadow_price=None, rol=0.1):
        if self.problem in ["all", "s_al"]:
            self.set_s_objective()
        elif self.problem in ["s_lr"]:
            self.set_s_objective(shadow_price)
        elif self.problem == "h":
            self.set_hydro_objective(shadow_price)
        elif self.problem == "s_admm":
            self.set_s_objective(shadow_price)
            self.set_quad_penalty_objective(rol)
        elif self.problem == "h_admm":
            self.set_hydro_objective(shadow_price)
            self.set_quad_penalty_objective(rol)

    def add_constraints(self):
        if self.problem in ["h", "h_admm"]:
            self._add_hydro_constraints()
        elif self.problem in ["s_lr", "s_admm"]:
            self._add_specific_constraints()
            self._general_P_constraints()
            self._add_shut_up_down_constraints_1()
            self._add_reserve_constraints()
        elif self.problem in ["s_al"]:
            self._general_D_constraints()
            self._add_specific_constraints()
            self._general_P_constraints()
            self._add_shut_up_down_constraints_1()
            self._add_reserve_constraints()
        elif self.problem == "all":
            self._general_D_constraints()
            self._add_specific_constraints()
            self._general_P_constraints()
            self._add_shut_up_down_constraints_1()
            self._add_reserve_constraints()
            self._add_hydro_constraints()

    def _add_specific_constraints(self):
        # 分段上界
        self.mip_model.add_constraints_(
            [self.Ps[ge, t, stage] <= self.m.loc[ge].iloc[stage] * self.I[ge, t]
             for ge in self.generators for t in range(self.T) for stage in range(self.stages)],

            [f"P_U_constraint_{ge}_{t}_{stage}"
             for ge in self.generators for t in range(self.T) for stage in range(self.stages)]
        )

    def _general_D_constraints(self):
        # 需求约束
        if self.problem == "all":
            self.P_demands_cts = self.mip_model.add_constraints(
                [self.mip_model.sum([self.P[ge, t] for ge in self.generators]) +
                 self.mip_model.sum_vars([self.P_H[ge, t] for ge in self.H_generators]) >=
                 self.D.loc[t, "LoadT"]
                 for t in range(self.T)],

                [f"P_D_constraint_{t}" for t in range(self.T)]
            )
        elif self.problem == "s_al":  # Adjust
            self.P_demands_cts = self.mip_model.add_constraints(
                [self.mip_model.sum([self.P[ge, t] for ge in self.generators]) >=
                 self.D.loc[t, "s_demand"]
                 for t in range(self.T)],

                [f"P_D_constraint_{t}" for t in range(self.T)]
            )

    def _general_P_constraints(self):
        # 出力上下界
        self.mip_model.add_constraints_(
            [self.P[ge, t] >= self.s_data.loc[ge, "Pmin"] * self.I[ge, t]
             for ge in self.generators for t in range(self.T)],
            [f"P_L_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.P[ge, t] <= self.s_data.loc[ge, "Pmax"] * self.I[ge, t]
             for ge in self.generators for t in range(self.T)],
            [f"P_U_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.P[ge, t+1] - self.P[ge, t] >= -self.s_data.loc[ge, 'RD']
             for ge in self.generators for t in range(self.T - 1)],
            [f"ramp_L_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.P[ge, t+1] - self.P[ge, t] <= self.s_data.loc[ge, 'RU']
             for ge in self.generators for t in range(self.T - 1)],
            [f"ramp_U_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

    def _general_shut_up_dowm_constraints(self):
        # 开关机约束
        self.mip_model.add_constraints_(
            [self.I[ge, t + 1] - self.I[ge, t] == self.W[ge, t + 1] - self.Y[ge, t + 1]
             for ge in self.generators for t in range(self.T - 1)],
            [f"s_s_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T - 1)]
        )

        self.mip_model.add_constraints_(
            [self.W[ge, 0] - self.Y[ge, 0] == self.I[ge, 0] - self.s_data.loc[ge, "I_init"] for ge in self.generators],
            [f"init_on_constrains_{ge}" for ge in self.generators]
        )

        # 关机次数约束
        self.mip_model.add_constraints_(
            [self.mip_model.sum_vars([self.W[ge, t] for t in range(self.T)]) <=
             self.s_data.loc[ge, "NS"] for ge in self.generators],
            [f"start_limit_constraint_{ge}" for ge in self.generators]
        )

        self.mip_model.add_constraints_(
            [self.mip_model.sum_vars([self.Y[ge, t] for t in range(self.T)]) <=
             self.s_data.loc[ge, "ND"] for ge in self.generators],
            [f"shut_limit_constraint_{ge}" for ge in self.generators]
        )

    def _add_shut_up_down_constraints_1(self):

        self._general_shut_up_dowm_constraints()
        # 最小开关机时间约束
        self.mip_model.add_constraints_(
            [min(self.T - t, self.s_data.loc[ge, "T_i_on"]) * self.W[ge, t] <=
             self.mip_model.sum_vars(self.I[ge, k]
                                     for k in range(t, t + min(self.T - t, self.s_data.loc[ge, "T_i_on"])))
             for ge in self.generators for t in range(self.T)],
            [f'start_last_time_constrains_{ge}_{t}' for ge in self.generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [min(self.T - t, self.s_data.loc[ge, "T_i_off"]) * self.Y[ge, t] +
             self.mip_model.sum_vars(self.I[ge, k]
                                     for k in range(t, t + min(self.T - t, self.s_data.loc[ge, "T_i_off"]))) <=
             min(self.T - t, self.s_data.loc[ge, "T_i_off"])
             for ge in self.generators for t in range(self.T)],
            [f'shut_last_time_constrains_{ge}_{t}' for ge in self.generators for t in range(self.T)]
        )

    def _add_shut_up_down_constraints_2(self):

        self._general_shut_up_dowm_constraints()
        # 最小开关机时间约束
        self.mip_model.add_constraints_(
            [self.I[ge, t + 1] - self.I[ge, t] <= self.I[ge, tau]
             for ge in self.generators for t in range(self.T - 1)
             for tau in range(t + 1, min(t + self.s_data.loc[ge, "T_i_on"] + 1, self.T))],
            [f'start_last_time_constrains_{ge}_{t}_{tau}'
             for ge in self.generators for t in range(self.T - 1)
             for tau in range(t + 1, min(t + self.s_data.loc[ge, "T_i_on"] + 1, self.T))]
        )
        self.mip_model.add_constraints_(
            [self.I[ge, t] - self.I[ge, t + 1] <= 1 - self.I[ge, tau]
             for ge in self.generators for t in range(self.T - 1)
             for tau in range(t + 1, min(t + self.s_data.loc[ge, "T_i_on"] + 1, self.T))],
            [f'shut_last_time_constrains_{ge}_{t}_{tau}'
             for ge in self.generators for t in range(self.T - 1)
             for tau in range(t + 1, min(t + self.s_data.loc[ge, "T_i_on"] + 1, self.T))]
        )

        self.mip_model.add_constraints_(
            [self.I[ge, 0] - self.s_data.loc[ge, "I_init"] <= self.I[ge, tau]
             for ge in self.generators for tau in range(min(self.s_data.loc[ge, "T_i_on"], self.T))],
            [f'start_init_last_time_constrains_{ge}_{tau}'
             for ge in self.generators for tau in range(min(self.s_data.loc[ge, "T_i_on"], self.T))]
        )

        self.mip_model.add_constraints_(
            [self.s_data.loc[ge, "I_init"] - self.I[ge, 0] <= 1 - self.I[ge, tau]
             for ge in self.generators for tau in range(min(self.s_data.loc[ge, "T_i_on"], self.T))],
            [f'shut_init_last_time_constrains_{ge}_{tau}'
             for ge in self.generators for tau in range(min(self.s_data.loc[ge, "T_i_on"], self.T))]
        )

    def _add_reserve_constraints(self):
        # 备用上界
        self.mip_model.add_constraints_(
            [self.R[ge, t] <= self.s_data.loc[ge, "Rr_max"] * self.I[ge, t]
             for ge in self.generators for t in range(self.T)],
            [f"R_U_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.Rs[ge, t] <= self.s_data.loc[ge, "Rs_max"] * self.I[ge, t]
             for ge in self.generators for t in range(self.T)],
            [f"Rs_U_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

        # 备用需求
        self.mip_model.add_constraints_(
            [self.mip_model.sum_vars([self.R[ge, t] for ge in self.generators]) >= self.D.loc[t, "Rr"]
             for t in range(self.T)],

            [f"R_D_constraint_{t}" for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.mip_model.sum_vars([self.Rs[ge, t] for ge in self.generators]) >= self.D.loc[t, "Rs"]
             for t in range(self.T)],

            [f"Rs_D_constraint_{t}" for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.mip_model.sum_vars([self.Rs[ge, t] for ge in self.generators]) +
             self.mip_model.sum_vars([self.R1[ge, t] for ge in self.generators]) >=
             self.D.loc[t, "Rs"] + self.D.loc[t, "R1"]
             for t in range(self.T)],

            [f"R1_D_constraint_{t}" for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.mip_model.sum_vars([self.Rs[ge, t] for ge in self.generators]) + self.mip_model.sum_vars(
                [self.R1[ge, t] for ge in self.generators]) +
             self.mip_model.sum_vars([self.R3[ge, t] for ge in self.generators]) >=
             self.D.loc[t, "Rs"] + self.D.loc[t, "R1"] + self.D.loc[
                 t, "R3"]
             for t in range(self.T)],

            [f"R3_D_constraint_{t}" for t in range(self.T)]
        )

        # 总发电约束
        self.mip_model.add_constraints_(
            [self.P[ge, t] + self.R[ge, t] + self.Rs[ge, t] + self.R1[ge, t] + self.R3[ge, t] <=
             self.s_data.loc[ge, "Pmax"]
             for ge in self.generators for t in range(self.T)],
            [f"all_U_constraint_{ge}_{t}" for ge in self.generators for t in range(self.T)]
        )

    def _add_hydro_constraints(self):
        # 水电子问题约束：进水等于出水
        balance_constraints = []
        for ge in self.H_generators:
            connected = self.H_connect_data.query("to == @ge").set_index("from_").delay.to_dict()
            forecast_com = self.H_data.loc[ge, "forecast_com"]
            balance_constraints += self.mip_model.add_constraints(
                [self.X[ge, t + 1] == self.X[ge, t] - self.Q_H_1[ge, t] - self.Q_H_2[ge, t] +
                 self.mip_model.sum([(self.Q_H_1[ge_from, (t - delay) if (t - delay) >= 0 else (t - delay + 96)] +
                                      self.Q_H_2[ge_from, (t - delay) if (t - delay) >= 0 else (t - delay + 96)])
                                     for ge_from, delay in connected.items()]
                                    ) + forecast_com
                 for t in range(self.T)],
                [f"balance_constrains_{ge}_{t}" for t in range(self.T)]
            )

        # 出水及水位限制
        self.mip_model.add_constraints_(
            [self.H_data.loc[ge, "drop_min"] <= self.Q_H_1[ge, t]
             for ge in self.H_generators for t in range(self.T)],
            [f"Q_Lower_limit_{ge}_{t}" for ge in self.H_generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.Q_H_1[ge, t] <= self.H_data.loc[ge, "drop_max"]
             for ge in self.H_generators for t in range(self.T)],
            [f"Q_Upper_limit_{ge}_{t}" for ge in self.H_generators for t in range(self.T)]
        )

        self.mip_model.add_constraints_(
            [self.X[ge, 0] == 25 for ge in self.H_generators],
            [f"X_init_{ge}" for ge in self.H_generators]
        )

        self.mip_model.add_constraints_(
            [self.P_H[ge, t] == self.H_data.loc[ge, "alpha"] * self.Q_H_1[ge, t]
             for ge in self.H_generators for t in range(self.T)],
            [f"P_Q_function_ctr_{ge}_{t}" for ge in self.H_generators for t in range(self.T)]
        )

    def model_solving(self):
        if self.problem in ["s_lr", "s_al", "s_admm"]:
            return self.get_s_solution()
        elif self.problem == "all":
            return self.get_all_solution()
        elif self.problem in ["h", "h_admm"]:
            return self.get_hydro_solution()

    def get_all_solution(self):
        self.mip_model.solve(log_output=True)
        print("---------- model description ----------")
        print(f"model's number of binary variables: {self.mip_model.number_of_binary_variables}")
        print(f"model's number of continuous variables: {self.mip_model.number_of_continuous_variables}")
        print(f"model's number of constraints: {self.mip_model.number_of_constraints}")
        print(f"model's objective value: {self.mip_model.objective_value}")
        print("---------------------------------------")
        return 

    def get_s_solution(self):
        if self.problem == "s_al":
            temp = cplex.Cplex(self.mip_model.get_cplex())
            start = temp.get_time()
            temp.solve()
            temp.set_problem_type(temp.problem_type.fixed_MILP)
            temp.solve()
            time_sol = temp.get_time()-start
            shadow_price = pd.Series(temp.solution.get_dual_values([ctr.get_name() for ctr in self.P_demands_cts]),
                                     name="price")
            shadow_price.index.set_names(["time"], inplace=True)
            shadow_price = shadow_price.to_frame()

            return shadow_price, temp.solution.get_objective_value(), time_sol
        else:
            self.mip_model.solve(log_output=True)
            time_sol = self.mip_model.solve_details.time
            P_solution = pd.Series({
                t: self.mip_model.sum([self.P[ge, t] for ge in self.generators]).solution_value
                for t in range(self.T)
            })
            return P_solution, self.mip_model.objective_value, time_sol

    def cal_shadow_price(self):
        # 逻辑计算影子价格
        self.mip_model.solve(log_output=True)
        time_sol = self.mip_model.solve_details.time
        P_solution = pd.Series(
            {(ge, t): self.mip_model.sum([self.Ps[ge, t, stage] for stage in range(10)]).solution_value
             for ge in self.generators for t in range(96)}).reset_index().rename(
            columns={"level_0": "generator", "level_1": "t", 0: "P"})
        print(P_solution)
        intervals = self.cost.iloc[:, self.stages:].stack().reset_index(1, drop=True).groupby("id").cumsum().groupby(
            "id").apply(
            lambda df: pd.IntervalIndex.from_breaks(df.values, closed="both"))
        bids = self.cost.iloc[:, :self.stages].stack().reset_index(1, drop=True)
        shadow_price = pd.DataFrame(index=pd.Index(range(self.T), name="time"), columns=["price"])
        for t, data in P_solution.groupby(["t"]):
            shadow_price.loc[t] = data.apply(lambda x: bids.loc[x.generator].loc[
                intervals[x.generator].contains(x.P).tolist() + [False]].max(), axis=1).max()
        return shadow_price, self.mip_model.objective_value, time_sol

    def get_hydro_solution(self):
        self.mip_model.solve(log_output=True)
        time_sol = self.mip_model.solve_details.time
        P_H_solution = pd.Series({
            t: self.mip_model.sum_vars([self.P_H[ge, t] for ge in self.H_generators]).solution_value
            for t in range(self.T)
        })
        return P_H_solution, self.mip_model.objective_value, time_sol

    def Change_Constr_Dt(self, p_demand):
        if self.problem != "all":
            raise TypeError("stupid pig!!!")
        self.D["LoadT"] = p_demand
        for t in range(self.T):
            self.P_demands_cts[t].rhs = self.D.loc[t, "LoadT"]
        return

    def fix_var_value(self, pair_vals):
        for key, value in pair_vals.items():
            f_var = self.mip_model.get_var_by_name(key)
            self.mip_model.set_var_lb(f_var, value)
            self.mip_model.set_var_ub(f_var, value)
        return


if __name__ == '__main__':
    basic_problem = SCUC_basic(problem="all")

    basic_problem.set_objective()
    basic_problem.add_constraints()
    basic_problem.model_solving()
    # basic_problem_s_sub.Change_Constr_Dt(P_demand=basic_problem_s_sub.D["LoadT"])
    # shadow_price, objective_values, timesol = basic_problem_s_sub.model_solving()
    # basic_problem_hydro_sub.set_objective(shadow_price)
    # basic_problem_hydro_sub.add_constraints()
    # basic_problem_hydro_sub.model_solving()
