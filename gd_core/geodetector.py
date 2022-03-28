#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 21:09
# @Author  : gjg
# @Site    : 
# @File    : geodetector.py
# @Software: PyCharm

import os
import numpy as np
import pandas as pd
from itertools import combinations
import xlwt
from xlwt import easyxf
from scipy.stats import f, levene, ncf, ttest_ind

gd_path = os.path.dirname(__file__)


class GeoDetector(object):
    def __init__(self, data, x_names, y_name, save_path=None, alpha=0.05):
        """

        Args:
            data:
            x_names:
            y_name:
            save_path:
            alpha:
        """
        #
        self.file_name = None
        self.x_names = x_names
        self.y_name = y_name
        self.alpha = alpha
        if save_path:
            self.save_path = save_path
        else:
            self.save_path = os.getcwd()
        for x_name in x_names:
            if data[x_name].values[0] is not str:
                data.loc[:, x_name] = data[x_name].astype(str).to_numpy()
        self.data = data
        self.n = len(data)
        #
        self.var_pop = np.var(self.data[self.y_name], ddof=0)
        self.var_sam = np.var(self.data[self.y_name], ddof=1)
        self.sst = self.var_pop * self.n
        self.len_x = len(self.x_names)
        self.factor_detector = self._factor_detector()

        pd.set_option('mode.chained_assignment', None)

    def _factor_detector(self):
        """
        Compares the accumulated dispersion variance of each sub-group with the dispersion variance of the all
        """
        len_x = len(self.x_names)
        factor_result = pd.DataFrame({"q": [0] * len_x, "p-value": [0] * len_x, "num_strata": [0] * len_x})
        factor_result.index = self.x_names
        for x_name in self.x_names:
            data_i = self.data[[x_name, self.y_name]]
            mean_h = data_i.groupby(x_name)[self.y_name].mean()
            var_h = data_i.groupby(x_name)[self.y_name].agg(np.var, ddof=0)
            n_h = data_i.groupby(x_name)[self.y_name].count()
            q_i, sig_i = self._q_calculate(mean_h, var_h, n_h)
            factor_result.loc[x_name, :] = [q_i, sig_i, len(n_h)]

        return factor_result

    @property
    def risk_detector(self):
        """
        Compares the difference of average values between sub-groups
        """
        risk_result = dict()
        for x_name in self.x_names:
            risk_name = self.data.groupby(x_name)[self.y_name].mean()
            strata = np.sort(self.data[x_name].unique())
            t_test = np.empty((len(strata), len(strata)))
            t_test.fill(np.nan)
            t_test_strata = pd.DataFrame(t_test, index=strata, columns=strata)
            for i in range(len(strata) - 1):
                for j in range(i + 1, len(strata)):
                    y_i = self.data.loc[self.data[x_name] == strata[i], [self.y_name]]
                    y_j = self.data.loc[self.data[x_name] == strata[j], [self.y_name]]
                    y_i = np.array(y_i).reshape(-1)
                    y_j = np.array(y_j).reshape(-1)
                    # hypothesis testing of variance homogeneity
                    levene_result = levene(y_i, y_j)
                    if levene_result.pvalue < self.alpha:
                        # variance non-homogeneous
                        ttest_result = ttest_ind(y_i, y_j, equal_var=False)
                    else:
                        ttest_result = ttest_ind(y_i, y_j)

                    t_test_strata.iloc[j, i] = ttest_result.pvalue <= self.alpha

            risk_x_name = dict(xname=x_name, risk=risk_name, ttest_stra=t_test_strata)
            risk_result[x_name] = risk_x_name
        return risk_result

    # @property

    @property
    def interaction_detector(self):
        """
        Compares the sum of the disease contribution of two individual attributes vs. the contribution of the two
        attributes when taken together.
        """

        factor_detector = self.factor_detector
        q = factor_detector['q'].to_numpy()
        q_sig = factor_detector['p-value'].to_numpy()
        # interaction
        index_com = combinations(range(len(q)), 2)
        index_com_value = [q[list(i)] for i in index_com]
        fuc_value = np.array([[i.min(), i.max(), i.sum()] for i in index_com_value])
        df_index = combinations(self.x_names, 2)
        df_index = [i for i in df_index]
        df_interaction = pd.DataFrame(index=df_index, columns=['inter_value', 'inter_action'])

        interaction_result_q = np.diag(q)
        interaction_result_sig = np.diag(q_sig)
        num_index = 0
        for i in range(self.len_x - 1):
            for j in range(i + 1, self.len_x):
                temp_data = pd.DataFrame(self.data[self.y_name])
                temp_data['inter_name'] = self.data[self.x_names[i]] + self.data[self.x_names[j]]
                mean_h = temp_data.groupby("inter_name")[self.y_name].mean()
                var_h = temp_data.groupby("inter_name")[self.y_name].agg(np.var, ddof=0)
                n_h = temp_data.groupby("inter_name")[self.y_name].count()
                q_i, sig_i = self._q_calculate(mean_h, var_h, n_h)
                interaction_result_q[j, i] = q_i
                interaction_result_sig[j, i] = sig_i
                # interaction
                df_interaction.iloc[num_index, 0] = q_i
                num_index += 1

        q_value = pd.DataFrame(data=interaction_result_q, index=self.x_names, columns=self.x_names)
        sig_value = pd.DataFrame(data=interaction_result_sig, index=self.x_names, columns=self.x_names)
        # interaction result
        inter_action = ['Weaken_nonlinear', 'Weaken_uni-', 'Enhance_bi-', 'Independent', 'Enhance_nonlinear']
        # temp_bool
        df_interaction.loc[:, 'inter_action'] = 'Enhance_bi-'
        independent_bool = fuc_value[:, 2] == df_interaction['inter_value'].to_numpy()
        df_interaction.loc[independent_bool, 'inter_action'] = 'Independent'
        enhance_non_bool = (fuc_value[:, 2] < df_interaction['inter_value'].to_numpy())
        df_interaction.loc[enhance_non_bool, 'inter_action'] = 'Enhance_nonlinear'

        interaction_result = dict(q=q_value, sig=sig_value, interaction=df_interaction)
        return interaction_result

    @property
    def ecological_detector(self):
        """
        Compares the variance calculated from each sub-groups divided according to one determinant with that divided
        according to another determinant

        """
        eco_array = np.empty((self.len_x, self.len_x))
        eco_array.fill(np.nan)
        ecological_result = pd.DataFrame(eco_array, columns=self.x_names, index=self.x_names)

        for i in range(self.len_x - 1):
            x_name_i = self.x_names[i]
            ssw_i = self.stat_stratum(x_name_i)
            for j in range(i + 1, self.len_x):
                x_name_j = self.x_names[j]
                ssw_j = self.stat_stratum(x_name_j)
                ecological_result.iloc[j, i] = ssw_i / ssw_j > f.ppf(
                    1 - self.alpha, dfn=len(self.data), dfd=len(self.data)
                )
        return ecological_result

    def _q_calculate(self, mean_h, var_h, n_h):
        len_stratum = var_h.size
        sse = np.dot(var_h, n_h)
        q_i = 1 - sse / self.sst
        # sig
        fv = (self.n - len_stratum) * q_i / ((len_stratum - 1) * (1 - q_i))
        nc_para = (pow(mean_h, 2).sum() - pow(np.dot(np.sqrt(n_h), mean_h), 2) / self.n) / self.var_sam
        sig_i = 1 - ncf.cdf(fv, len_stratum - 1, self.n - len_stratum, nc_para)
        return q_i, sig_i

    def stat_stratum(self, x_name):
        var_st = self.data.groupby(x_name)[self.y_name].agg(
            np.var, ddof=0
        )
        n_st = self.data.groupby(x_name)[self.y_name].count()
        ssw_st = (var_st * n_st).sum()

        return ssw_st

    def save_to_xls(self, save_file):
        num_style = easyxf("borders: left thin, right thin, top thin, bottom thin;" "align: vertical center,wrap off, "
                           "horizontal center;", num_format_str='#,##0.0000')
        str_style = easyxf("borders: left thin, right thin, top thin, bottom thin;" "align: vertical center, wrap off,"
                           "horizontal center;")
        len_x = len(self.x_names)
        gd_xls = xlwt.Workbook()
        ws_input = gd_xls.add_sheet('Input data')
        ws_input = self._xls_write_df(df=self.data, worksheet=ws_input)
        # risk-detector
        ws_risk = gd_xls.add_sheet('Risk detector')
        risk_data = self.risk_detector
        row, col = 0, 0
        for x in self.x_names:
            x_risk = risk_data[x]['risk']
            x_ttest = risk_data[x]['ttest_stra']
            len_strata = len(x_risk)
            ws_risk.write_merge(row, row, col, col + len_strata, x + ': risk')
            ws_risk.write_merge(row + 4, row + 4, col, col + len_strata, x + ' t-test: 0.05')
            ws_risk.write(row + 5, 0, style=str_style)

            for i in range(len_strata):
                ws_risk.write(row + 1, i + col, x_risk.index[i], style=str_style)
                ws_risk.write(row + 2, i + col, x_risk[i], style=num_style)
                # ws_risk = self._xls_write_df(df=self.data, worksheet=ws_input)

                ws_risk.write(row + 5, col + i + 1, str(x_risk.index[i]), style=str_style)
                ws_risk.write(row + 6 + i, col, str(x_risk.index[i]), style=str_style)
                for j in range(i):
                    ws_risk.write(row + 6 + i, col + j + 1, bool(x_ttest.iloc[i, j]), style=str_style)
                for spi in range(i, len_strata):
                    ws_risk.write(row + 6 + i, col + spi + 1, style=str_style)

            row = row + len_strata + 8

        # factor-detector
        ws_fact = gd_xls.add_sheet('Factor detector')
        ws_fact = self._xls_write_df(df=self.factor_detector, worksheet=ws_fact)
        # interaction detector
        ws_inter = gd_xls.add_sheet('Interaction detector')
        if len_x > 1:
            inter_q = self.interaction_detector['q']
            inter_sig = self.interaction_detector['sig']
            interactions = self.interaction_detector['interaction']
            ws_inter.write_merge(0, 0, 0, 0 + len_x, "q-statistic")
            ws_inter.write(1, 0, style=str_style)
            ws_inter.write_merge(len_x + 3, len_x + 3, 0, 0 + len_x, "Sig F test: 0.05")
            ws_inter.write(len_x + 4, 0, style=str_style)
            ws_inter.write_merge(2 * len_x + 6, 2 * len_x + 6, 0, 0, "Interaction between Xs")
            ws_inter.write(2 * len_x + 7, 0, style=str_style)
            ws_inter.write(2 * len_x + 7, 1, "q-value", style=str_style)
            ws_inter.write(2 * len_x + 7, 2, "interaction", style=str_style)
            # write q-statistic values
            row_inter = 0
            for ind, val in enumerate(self.x_names):
                ws_inter.write(1, ind + 1, val, style=str_style)
                ws_inter.write(ind + 2, 0, val, style=str_style)
                ws_inter.write(len_x + 4, ind + 1, val, style=str_style)
                ws_inter.write(len_x + ind + 5, 0, val, style=str_style)
                for inter_col in range(0, ind):
                    ws_inter.write(ind + 2, inter_col + 1, inter_q.iloc[ind, inter_col], style=num_style)
                    ws_inter.write(ind + len_x + 5, inter_col + 1, inter_sig.iloc[ind, inter_col], style=num_style)
                for sp_j in range(ind, len_x):
                    ws_inter.write(ind + 2, sp_j + 1, style=str_style)
                    ws_inter.write(ind + len_x + 5, sp_j + 1, style=str_style)

                for ind2, val2 in enumerate(self.x_names[ind + 1:]):
                    ws_inter.write(2 * len_x + row_inter + 8, 0, val + "&" + val2, style=str_style)
                    ws_inter.write(2 * len_x + row_inter + 8, 1, interactions.iloc[row_inter, 0], style=num_style)
                    ws_inter.write(2 * len_x + row_inter + 8, 2, interactions.iloc[row_inter, 1], style=num_style)
                    row_inter = row_inter + 1
            # insert image
            img_bmp = (os.path.join(gd_path, 'interaction_fig/interaction.bmp'))
            fig_row = 2 * len_x + row_inter + 11
            ws_inter.insert_bitmap(img_bmp, fig_row, 0)

        # ecological detector
        ws_ecol = gd_xls.add_sheet('Ecological detector')
        ws_ecol.write_merge(1, 1, 0, 3, "Sig. F-test: 0.05")
        eco_df = self.ecological_detector
        ws_ecol.write(2, 0, style=str_style)
        for ind, val in enumerate(self.x_names):
            ws_ecol.write(2, 0 + ind + 1, val, style=str_style)
            ws_ecol.write(ind + 3, 0, val, style=str_style)
            for j in range(ind):
                ws_ecol.write(ind + 3, j + 1, bool(eco_df.iloc[ind, j]), style=str_style)
            for sp_j in range(ind, len_x):
                ws_ecol.write(ind + 3, sp_j + 1, style=str_style)

        gd_xls.save(save_file)

    def print_result(self, printFile=None):

        if printFile is None:
            self._print_result()
        else:
            with open(printFile, 'w') as outfile:
                self._print_result(printFile=outfile)

    def _print_result(self, printFile=None):
        """
        """
        risk_result = self.risk_detector
        len_x = len(self.x_names)
        print("-------------------risk results--------------------", file=printFile)
        for i in self.x_names:
            print("%s - risk :" % i, file=printFile)
            print(risk_result[i]['risk'], file=printFile)
            print("%s - ttest :" % i, file=printFile)
            print(risk_result[i]['ttest_stra'], file=printFile)
        print('------------------factor detector------------------', file=printFile)
        print(self.factor_detector, file=printFile)

        if len_x > 1:
            print('--------------interaction detector-----------------', file=printFile)
            print(self.interaction_detector['q'], file=printFile)
            print(self.interaction_detector['interaction'], file=printFile)
            print('--------------ecological detector------------------', file=printFile)
            print(self.ecological_detector, file=printFile)

    @staticmethod
    def _xls_write_df(df, worksheet, startrow=0, startcol=0):
        col_names = df.columns.values
        index = df.index

        for col_ind, col_name in enumerate(col_names):
            worksheet.write(startrow, startcol + col_ind + 1, col_name)
            for row_data, cell in enumerate(df[col_name]):
                worksheet.write(startrow + row_data + 1, startcol + col_ind + 1, cell)

        for index_ind, index_name in enumerate(index):
            worksheet.write(startrow + 1 + index_ind, startcol, index_name)

        return worksheet


if __name__ == '__main__':
    testdata = pd.read_csv("../data/collectdata.csv")
    gd_result = GeoDetector(testdata, ["watershed"], "incidence", alpha=0.05)
