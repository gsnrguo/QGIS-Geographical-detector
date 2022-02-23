#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/29 16:44
# @Author  : gjg
# @Site    :
# @File    : optimal_gd.py
# @Software: PyCharm
import warnings

import numpy as np
import pandas as pd
from math import log
from scipy import stats
from copy import deepcopy


# import pandas as pd


class optimal_geo_detector:
    """
    """

    def __init__(self, x, y,
                 criterion="squared_error",
                 max_group=None,
                 min_group=1,
                 min_samples_group=2,
                 cv_seed=None,
                 min_delta_q=0.001,
                 cv_fold=10,
                 cv_times=1,
                 lst_alpha=0.05,
                 ccp_alpha=0.0):

        if (cv_times > 1) & (cv_seed is not None):
            self.cv_seed = [cv_seed + i for i in range(cv_times)]
        else:
            self.cv_seed = [cv_seed]
        self.criterion = criterion
        self.min_group = min_group
        if max_group is None:
            self.max_group = int(len(y) / min_samples_group)
        else:
            self.max_group = max_group

        self.min_samples_split = min_samples_group * 3
        self.min_samples_group = min_samples_group
        self.cv = cv_fold
        self.cv_time = cv_times
        if min_group == max_group:
            self.fix_group = min_group
        else:
            self.fix_group = None

        if self.fix_group is None:
            self.lst_alpha = lst_alpha
        else:
            self.lst_alpha = 1
            if self.fix_group * min_samples_group > len(y):
                warnings.warn('The fixed number of groups is too lager or the min_samples in groups is too larger')

        self.ccp_alpha = ccp_alpha
        self.ccp_alpha_ls = []
        self.min_delta_q = min_delta_q

        self.y = y
        self.x = x
        self.group = np.ones(len(x))
        self.sst = np.sum(pow(y - np.mean(y), 2))

        self.split_group, self.metric = self.split(x=self.x, y=self.y)
        if len(set(self.split_group)) < self.max_group:
            self.max_group = len(set(self.split_group))

        if (self.max_group != self.min_group) & (self.criterion == 'squared_error'):
            self.alpha_list = self.lost_complexity_info()
        else:
            self.alpha_list = pd.DataFrame([])

        self.groups = self.stratification()
        self.group_interval, self.sort_labs, self.lab_info = self.groups2interval()

    # @property
    def stratification(self):
        """
        stratification function
        """
        metric = self.metric
        f_node = metric['node'].to_numpy()
        delta_q = metric['delta_q'].to_numpy()
        groups_best = self.split_group
        # cc_lost_list = dict.fromkeys(['del_node', 'num_group', 'q_list', 'sse', 'cc_lost_best'])
        if self.max_group == self.min_group:
            if self.criterion == 'squared_error':
                groups_best, cc_lost_list = self.fix_pruning_group(self.split_group, f_node, delta_q)
            if (self.criterion == 'linear_statistic') & (self.max_group < len(set(self.split_group))):
                groups_best, cc_lost_list = self.fix_pruning_group(self.split_group, f_node, delta_q)
        else:
            if self.criterion == 'squared_error':
                if self.alpha_list.shape[0] > 1:
                    groups_best = self.alpha_list.loc[self.alpha_list['cv_error'] == min(self.alpha_list['cv_error']),
                                                      'group']
                    groups_best = groups_best.values[-1]
                else:
                    groups_best = self.alpha_list['group'].values[-1]

        return groups_best

    def split(self, x, y):
        """
        split the x into groups minimizing the y
        Args:
            x:
            y:

        Returns:
            groups: the group labels of unit
            metric_list: the split information

        """
        rank_index = np.argsort(x)
        y = y[rank_index]
        x = x[rank_index]
        alt_cut = np.diff(np.insert(x, 0, x[0])) != 0  # the same x is should not be split
        group_list = [np.arange(len(y))]
        groups = np.ones(len(y)).astype(int)
        group_lab = [1]  # initial group lab
        metric_list = list()
        len_new_node = 1
        # delta_q = 1
        while len_new_node > 0:
            new_node = list()
            lab_index = 0  # initial group labels index
            update_lab = list()
            for group_index in group_list:
                y_len = len(group_index)
                y_val = np.mean(y[group_index])
                y_val_var = np.var(y[group_index]) / y_len
                sst = np.sum(pow((y[group_index]) - y_val, 2))
                if len(group_index) >= self.min_samples_split:
                    sub_cut, metric = self._split_cut(y[group_index], alt_cut[group_index])
                    if sub_cut is not None:
                        delta_q = (sst - metric) / self.sst
                        local_q = (sst - metric) / sst
                        if delta_q < self.min_delta_q:
                            node_type = "leaf_node"
                            x_cut = np.nan
                        else:
                            node_type = "split_node"
                            x_cut = x[group_index][sub_cut]
                            new_node.append(group_index[:sub_cut])
                            new_node.append(group_index[sub_cut:])
                            update_lab.append(2 * group_lab[lab_index])  # the new left label after split
                            update_lab.append(2 * group_lab[lab_index] + 1)  # the new right label after split
                            # update groups
                            groups[group_index[:sub_cut]] = 2 * group_lab[lab_index]
                            groups[group_index[sub_cut:]] = 2 * group_lab[lab_index] + 1

                        metric_list.append(
                            (group_lab[lab_index], sst, delta_q, x_cut, y_len, y_val, y_val_var, node_type, local_q))
                    else:  # alt_cut point is null
                        x_cut = np.nan
                        local_q = np.nan
                        node_type = "leaf_node"
                        delta_q = self.min_delta_q
                        metric_list.append(
                            (group_lab[lab_index], sst, delta_q, x_cut, y_len, y_val, y_val_var, node_type, local_q))
                else:
                    x_cut = np.nan
                    local_q = np.nan
                    node_type = "leaf_node"
                    delta_q = self.min_delta_q
                    metric_list.append(
                        (group_lab[lab_index], sst, delta_q, x_cut, y_len, y_val, y_val_var, node_type, local_q))

                lab_index += 1

            len_new_node = len(new_node)
            group_lab = update_lab
            group_list = new_node
            # group_num = len(set(groups))

        metric_dict = dict.fromkeys(['node', 'sst', 'delta_q', 'next_split', 'length', 'estimate', 'est_var',
                                     'node_type', 'local_q'])
        metric_dict['node'] = np.array([i[0] for i in metric_list], dtype=int)
        metric_dict['sst'] = np.array([i[1] for i in metric_list])
        metric_dict['delta_q'] = np.array([i[2] for i in metric_list])
        metric_dict['next_split'] = np.array([i[3] for i in metric_list])
        metric_dict['length'] = np.array([i[4] for i in metric_list])
        metric_dict['estimate'] = np.array([i[5] for i in metric_list])
        metric_dict['est_var'] = np.array([i[6] for i in metric_list])
        metric_dict['node_type'] = np.array([i[7] for i in metric_list])
        metric_dict['local_q'] = np.array([i[8] for i in metric_list])

        recover_rank = np.argsort(rank_index)
        return groups[recover_rank], pd.DataFrame(metric_dict)

    def fix_pruning_group(self, groups, nodes, delta_q):
        """

        Args:
            groups: list
            nodes: list
            delta_q: the delta-q-statistic of each split

        Returns:
            groups, the resultant groups
            group_info, the pruning information

        """
        cc_lost = np.array(delta_q) * self.sst
        terminal_node = list(set(groups))
        terminal_node.sort()
        group_num = len(terminal_node)
        alt_pruning = (np.array(terminal_node) / 2).astype(int)
        pruning_node = alt_pruning[:-1][np.diff(alt_pruning) == 0]

        del_node_ls = [0]
        num_group = [group_num]
        q_list = [np.sum(delta_q)]
        while group_num > self.min_group:
            alt_del_index = np.in1d(nodes, pruning_node)
            alt_index = np.arange(len(nodes))[alt_del_index]
            del_index = np.argmin(cc_lost[alt_index])
            del_node = pruning_node[del_index]
            c1 = groups == del_node * 2
            c2 = groups == del_node * 2 + 1
            merge_index = [c1[i] or c2[i] for i in range(len(c1))]
            groups[merge_index] = del_node
            # update node, split_metric cc_lost and pruning_node
            del_index_raw = alt_index[del_index]
            nodes = np.delete(nodes, del_index_raw)
            delta_q = np.delete(delta_q, del_index_raw)
            cc_lost = np.delete(cc_lost, del_index_raw)
            pruning_node = np.delete(pruning_node, del_index)
            # update terminal_node
            terminal_node.remove(del_node * 2)
            terminal_node.remove(del_node * 2 + 1)
            terminal_node.insert(0, del_node)

            # update pruning_node
            if del_node % 2 == 0:
                if (del_node + 1) in terminal_node:
                    pruning_node = np.append(pruning_node, int(del_node / 2))
            else:
                if (del_node - 1) in terminal_node:
                    pruning_node = np.append(pruning_node, int(del_node / 2))

            pruning_node.sort()
            del_node_ls.append(del_node)
            num_group.append(group_num - 1)
            q_list.append(np.sum(delta_q))

            group_num -= 1

        sse = self.sst * (1 - np.array(q_list))
        cc_lost_best = sse + self.ccp_alpha * np.array(num_group)
        group_info = {"del_node": del_node_ls, "num_group": num_group,
                      "q_list": q_list,
                      "sse": sse,
                      "cc_lost_best": cc_lost_best}

        return groups, group_info

    def lost_complexity_info(self):
        """
        lost complexity information

        Returns:
            pandas dataframe:
                node: the merging node
                group_num: the number of group after merging the node
                merging_in_node: inner node of node and leaf node, contains the node itself
                groups: groups after merging the node
                cv_error: rmse of k-fold cv
                cv_std: std of k-fold, it is std(rmse)/k-fold
        """
        split_info = self.metric
        nodes = split_info.loc[split_info['node_type'] == 'split_node', 'node'].to_numpy()
        groups = deepcopy(self.split_group)
        alpha_list = [0]
        group_ls = [self.split_group]
        group_num = [len(set(groups))]
        num_group = group_num[0]
        merging_node = [0]
        merge_in_node = [0]
        sse = self.sum_squared_error(groups, self.y)
        q_list = [1 - sse / self.sst]

        while num_group > self.min_group:
            merge_node, alpha, del_node, merging_index = self._regula_para(nodes, groups, self.y, self.sst)
            if merge_node == 0:
                break
            else:
                nodes = list(set(nodes) - del_node)
                groups[merging_index] = merge_node
                num_group = len(set(groups))
                if num_group <= self.max_group:
                    merging_node.append(merge_node)
                    merge_in_node.append(del_node)
                    alpha_list.append(alpha)
                    group_ls.append(deepcopy(groups))
                    group_num.append(num_group)
                    sse = self.sum_squared_error(groups, self.y)
                    q_list.append(1 - sse / self.sst)

        if len(set(self.split_group)) > self.max_group:
            alpha_info = dict(node=merging_node[1:], q_value=q_list[1:], group_num=group_num[1:],
                              merge_in_node=merge_in_node[1:],
                              alpha=alpha_list[1:], group=group_ls[1:])
        else:
            alpha_info = dict(node=merging_node, q_value=q_list, group_num=group_num, merge_in_node=merge_in_node,
                              alpha=alpha_list, group=group_ls)

        alpha_info = pd.DataFrame(alpha_info)
        alpha_info.sort_values(by=['alpha'])
        if alpha_info.shape[0] > 1:
            cv_alpha = alpha_info['alpha'].to_numpy()
            cv_error = 0
            cv_std = 0
            for cv in range(self.cv_time):
                if self.cv_seed is None:
                    cv_info = self.lc_cv_info(cv_alpha)
                else:
                    cv_info = self.lc_cv_info(cv_alpha, seed=self.cv_seed[cv])
                cv_error += np.array(cv_info['cv_error'])
                cv_std += np.array(cv_info['cv_std'])
            alpha_info['cv_error'] = cv_error / self.cv_time
            alpha_info['cv_std'] = cv_std / self.cv_time

        return alpha_info

    def lc_cv_info(self, cv_alpha, seed=None):
        """
        the alpha value of lost-complexity k-fold cross-validation information
        Args:
            cv_alpha: cv parameter
            seed: random seed

        Returns:
            cv precision
                root mean of squared error
                std of k-fold root mean of squared error

        """
        len_y = len(self.y)
        interval = int(np.around(len_y / self.cv))
        # cv
        rng = np.random.default_rng(seed=seed)
        random_index = rng.choice(range(len_y), size=len_y, replace=False)
        cv_cut = [range(i * interval, (i + 1) * interval) for i in range(self.cv - 1)]
        cv_cut.append(range(interval * (self.cv - 1), len_y))
        cv_index = [random_index[i] for i in cv_cut]
        option_cut = range(len_y)

        root_mse_info = []
        root_mse_std = []

        for alpha in cv_alpha:
            root_mse = []
            for i in cv_index:
                select = np.in1d(option_cut, i)
                train_y, test_y = self.y[~select], self.y[select]
                train_x, test_x = self.x[~select], self.x[select]
                groups, split_info = self.split(train_x, train_y)
                nodes = split_info.loc[split_info['node_type'] == 'split_node', 'node'].to_numpy()
                cut_groups = self._cv_info(train_y, groups, nodes, alpha)
                mse_i = self.predict(cut_groups, train_x, train_y, test_x, test_y)
                root_mse.append(pow(mse_i, 0.5))

            mean_cv = np.mean(root_mse)
            std_cv = np.std(root_mse) / self.cv

            root_mse_info.append(mean_cv)
            root_mse_std.append(std_cv)

        return dict(cv_error=root_mse_info, cv_std=root_mse_std)

    def _cv_info(self, train_y, groups, nodes, cut_alpha):
        """

        Args:
            train_y:
            groups:
            nodes:
            cut_alpha: the preset regularization parameter

        Returns:

        """

        mean_y = np.mean(train_y)
        sst = np.sum(pow((train_y - mean_y), 2))
        group_num = len(set(groups))
        while group_num > self.min_group:
            merge_node, alpha, del_node, merging_index = self._regula_para(nodes, groups, train_y, sst)
            if merge_node == 0:
                break
            else:
                if alpha > cut_alpha:
                    break
                nodes = list(set(nodes) - del_node)
                groups[merging_index] = merge_node
                group_num = len(set(groups))

        return groups

    def _regula_para(self, nodes, groups, y, sst):
        """

        Args:
            nodes:
            groups:
            y:
            sst:

        Returns:

        """
        alpha = np.inf
        merge_node = 0
        merging_index = []
        del_node = []
        group_num = len(set(groups))
        for node in nodes[1:]:
            _, del_node_i, group_index = self.sub_node(node, groups, nodes)
            alpha_temp = self._lc_ssb(groups[group_index], y[group_index])
            alpha_temp = alpha_temp / sst
            group_temp = group_num - len(set(groups[group_index])) + 1
            if group_temp >= self.min_group:
                if (alpha_temp < alpha) or ((alpha == alpha_temp) & (node > merge_node)):
                    merge_node = node
                    alpha = alpha_temp
                    del_node = del_node_i
                    merging_index = group_index

        return merge_node, alpha, del_node, merging_index

    @staticmethod
    def sub_node(node, groups, all_node):
        """
        extract the leaf node and its index, child-node and grandchild-node of a given father node
        Args:
            node: inner node
            groups:
            all_node:

        Returns:
            leaves: list, the leaves of node
            sub_in_node: set, the inner node of node
            sub_node_index: array, the index of node groups

        """
        row_index = np.arange(len(groups))
        max_depth = int(log(np.max(groups), 2))
        node_depth = int(log(node, 2))
        group_node = [node]
        for i in range(node_depth + 1, max_depth + 1):
            diff_dep = i - node_depth
            group_node = np.append(group_node,
                                   np.arange(node * pow(2, diff_dep), node * pow(2, diff_dep) + pow(2, diff_dep)))

        group_node = set(group_node)
        # print(group_node)
        leaves = list(group_node & set(groups))
        # leaves is none
        sub_in_node = group_node & set(all_node)
        sub_node_index = row_index[groups == leaves[0]]
        for i in leaves[1::]:
            sub_node_index = np.append(sub_node_index, row_index[groups == i])

        return leaves, sub_in_node, sub_node_index

    # @staticmethod
    def _lc_ssb(self, node_groups, node_y):
        """
        The between sum squared (ssb) error after running merging procedure
        cc_alpha = (sst(node) - sse(node))/((len(node_groups) - 1)* sst(all))
        Args:
            node_groups: sub—group
            node_y:

        Returns:
            cc_alpha:

        """
        mean_y = np.mean(node_y)
        sst = np.sum(pow((node_y - mean_y), 2))
        sse = self.sum_squared_error(node_groups, node_y)
        lc_ssb = (sst - sse) / (len(set(node_groups)) - 1)  #

        return lc_ssb

    @staticmethod
    def sum_squared_error(groups, y):
        """
        sum squared error of grouping
        Args:
            groups:
            y:

        Returns:
            sse: sum squared error

        """
        group_labs = np.unique(groups)
        sse = 0
        for i in group_labs:
            y_group = y[groups == i]
            mean_i = np.mean(y_group)
            sse += np.sum(pow((y_group - mean_i), 2))

        return sse

    @staticmethod
    def predict(groups, train_x, train_y, test_x, test_y):
        """
        prediction accuracy
        Args:
            groups: the result of stratification
            train_x:
            train_y:
            test_x:
            test_y:

        Returns:
            MSE

        """
        low_breaks = list()
        up_breaks = list()
        group_labs = np.unique(groups)
        predict_y = dict.fromkeys(group_labs)
        for i in group_labs:
            train_x_group = train_x[groups == i]
            predict_y[i] = np.mean(train_y[groups == i])
            low_breaks.append(np.min(train_x_group))
            up_breaks.append(np.max(train_x_group))

        low_breaks = np.array(low_breaks)
        low_breaks.sort()
        up_breaks = np.array(up_breaks)
        up_breaks.sort()
        # update the breaks, the interval is (-inf, b1), [b1,b2),..., [bk,inf)
        breaks = (low_breaks[1:] + up_breaks[:-1]) / 2
        breaks = breaks.astype(float)
        breaks[0] = -np.inf
        breaks[-1] = np.inf

        test_groups = list()
        for i in range(len(test_x)):
            _, ind = np.unique(test_x[i] >= breaks, return_index=True)
            test_groups.append(group_labs[ind[0] - 1])  #

        test_labs = np.unique(test_groups)
        mse = 0
        for i in test_labs:
            test_y_group = test_y[test_groups == i]
            mse += np.sum(pow((test_y_group - predict_y[i]), 2))

        return mse

    def _split_cut(self, split_y, alt_cut, optimal_met=None):
        """
        get the best split cut by exhaustive search
        Args:
            split_y: the raw index of y after sorting by x
            alt_cut: the possible split points that x is not same
            optimal_met: None

        Returns:
            best_cut: the index of cut point, means the split is x < x[cut]

        """
        # global optimal_met
        split_index = np.arange(self.min_samples_group, len(split_y) - self.min_samples_group)
        alt_cut = alt_cut[split_index]
        split_index = split_index[alt_cut]

        if self.criterion == 'squared_error':
            optimal_met = np.inf
        elif self.criterion == "linear_statistic":
            optimal_met = self.lst_alpha
        best_cut = None
        # @nb.jit() # using numba accelerate
        for cut in split_index:
            metric = self._criterion_metric(cut, split_y)
            if metric < optimal_met:
                optimal_met = metric
                best_cut = cut

        return best_cut, optimal_met

    def _criterion_metric(self, cut, y):
        """
        criterion of split
        Args:
            cut: cut the raw data into 2 split
            y: the data

        Returns:
            metric: the metric index

        """
        metric = None
        if self.criterion == 'squared_error':
            metric = np.sum(pow(y[0:cut] - np.mean(y[0:cut]), 2)) + np.sum(pow(y[cut::] - np.mean(y[cut::]), 2))

        if self.criterion == "linear_statistic":
            tx = np.ones(len(y))
            tx[cut::] = 0
            _, metric = self._linear_statistic(tx, y)

        return metric

    @staticmethod
    def _linear_statistic(tx, fy):
        """
        linear statistics of MC
        Args:
            tx: transform function of x
            fy: influence function of y

        Returns:

        """
        len_y = len(fy)
        t = np.dot(tx, fy)
        e_h = np.mean(fy)
        e_t = np.sum(tx) * e_h
        sigma_h = np.dot(fy - e_h, fy - e_h) / len_y
        sigma_t = len_y / (len_y - 1) * sigma_h * np.dot(tx, tx) - 1 / (len_y - 1) * sigma_h * pow(np.sum(tx), 2)
        test_t = abs((t - e_t) / pow(sigma_t, 0.5))
        two_side_p = stats.norm.sf(test_t) * 2  # 1d-cdf
        return test_t, two_side_p

    def groups2interval(self):
        """
        convert the groups into interval
        Returns:

        """
        rank_index = np.argsort(self.x)
        recover_rank = np.argsort(rank_index)
        group_x = self.x[rank_index]
        group_info = self.groups[rank_index]
        labels, in1, in2, counts = np.unique(group_info, return_index=True, return_inverse=True, return_counts=True,
                                             axis=None)
        breaks = group_x[in1]
        breaks_sort = np.argsort(breaks)
        breaks_recover = np.argsort(breaks_sort)
        edges = np.append(breaks[breaks_sort], group_x[-1])

        if type(breaks[0]) is int:
            edges = [str(x) for x in edges]
        else:
            edges = ["{:.2f}".format(x) for x in edges]
        max_width = max([len(edge) for edge in edges])
        k = len(edges) - 1
        left = ["["]
        left.extend("[" * (k - 1))
        right = ")" * (k - 1) + "]"
        lower = ["{:>{width}}".format(edges[i], width=max_width) for i in range(k)]
        upper = ["{:>{width}}".format(edges[i], width=max_width) for i in range(1, k + 1)]
        lower = [l + r for l, r in zip(left, lower)]
        upper = [l + r for l, r in zip(upper, right)]
        intervals = [l + ", " + r for l, r in zip(lower, upper)]
        sort_lab = np.arange(1, len(labels) + 1)
        intervals = np.array(intervals)[breaks_recover]
        sort_lab = sort_lab[breaks_recover]
        lab_info = pd.DataFrame(dict(node=labels, num_lab=sort_lab, intervals=intervals, counts=counts))

        inter_labs = intervals[in2]
        sort_labs = sort_lab[in2]

        return inter_labs[recover_rank], sort_labs[recover_rank], lab_info
