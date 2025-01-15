import pickle
# Main computation libraries
import scipy.sparse as sp
import numpy as np
import pandas as pd
# Deep learning related imports
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Subterms added to avoid division by 0 during normalization
EPSILON = 1e-7


class DataProcessor():
    def __init__(self):
        self._is_normalize_feature = None
        self._x_column = None
        self._y_column = None
        self._feature_column = True
        self._is_normalize_variable = True

    def _normalize_feature(self, features):
        min_features = features.min(axis=0)
        max_features = features.max(axis=0)
        # min-max
        return (features - min_features) / (max_features - min_features)

    def _normalize_variable(self, variable):
        # normalize the variable to [0,1] in each column
        variable = variable + EPSILON - np.min(variable, axis=0)
        variable = variable / (np.max(variable, axis=0) - np.min(variable, axis=0) + EPSILON) + EPSILON
        return variable


    def _load_vertex(self, path, feature_column, y_column, x_column=None, is_normalize_feature=True,
                     is_normalize_variable=True):
        """
        Load node information from csv file, id starts from 0 by default.
        :param path:
        :param feature_column:
        :param y_column:
        :param x_column:
        :param is_normalize_feature:
        :param is_normalize_variable:
        :return:
        """
        self._vertex_path = path
        self._feature_column = feature_column
        self._y_column = y_column
        self._x_column = x_column
        self._is_normalize_feature = is_normalize_feature
        self._is_normalize_variable = is_normalize_variable

        if x_column is None:
            x_column = []
        df = pd.read_csv(path)
        F = self._normalize_feature(df[feature_column].values.astype('float32')) if is_normalize_feature else df[
            feature_column].values.astype('float32')
        y = df[y_column].values.astype('float32')
        if len(x_column) != 0:
            df_x = df[x_column]
            X = self._normalize_variable(
                df_x.values.astype('float32')) if is_normalize_variable else df_x.values.astype('float32')
            constant = np.ones(X.shape[0])
            X = np.insert(X, 0, values=constant, axis=1)
        else:
            X = None
        return F, y, X

    def _load_edge(self, path, node_num):
        """
        Load the edge information from a csv file as a matrix in coo_matrix format.
        We assume that there are no two points with distance 0, therefore, the 0 element in the matrix represents no connection.
        :param path:
        :param node_num:
        :return:
        """
        df = pd.read_csv(path)
        # randomly select (OPTIONAL)
        # df = df.sample(frac=0.95, replace=False, random_state=11)
        E = sp.coo_matrix((df.values[:, 2], (df.values[:, 0], df.values[:, 1])), shape=(node_num, node_num),
                          dtype=np.float32).todense()
        return E

    def load_all_data(self, edge_path, vertex_path, feature_column, y_column, x_column=[], is_normalize_feature=True,
                      is_normalize_variable=True, device="cpu"):
        """
        Load all Y, X, F and S from csv files
        :param edge_path:
        :param vertex_path:
        :param feature_column:
        :param y_column:
        :param x_column:
        :param is_normalize_feature:
        :param is_normalize_variable:
        :param device:
        :return:
        """
        self._device = device
        print('NOTICE: VERTEX ID SHOULD START FROM 0 AND BE CONTINUOUS!')
        F, y, X = self._load_vertex(vertex_path, feature_column, y_column, x_column, is_normalize_feature,
                                    is_normalize_variable)
        node_num = len(y)
        E = self._load_edge(edge_path, node_num)
        edge = torch.tensor(E, dtype=torch.float32, device=device)
        vertex_feature = torch.tensor(F, dtype=torch.float32, device=device)
        vertex_y = torch.tensor(y, dtype=torch.float32, device=device)
        if X is not None:
            vertex_x = torch.tensor(X, dtype=torch.float32, device=device)
        else:
            vertex_x = None
        print(f"Load data with {node_num} nodes into {device}.")
        return edge, vertex_feature, vertex_y, vertex_x

    def print_and_output_result(self, path, result_name, y_hat, indicators, icv=0, x_betas=None, spatial_weights=None):
        """
        Output the prediction results to a file
        :param path:
        :param y_hat: Tensor
        :param y_data: Tensor
        :param indicators: Tensor
        :param x_betas: Tensor
        :param x_data: Tensor
        :return:
        """
        if self._x_column is None:
            return

        other_df = pd.read_csv(self._vertex_path)
        x_data = other_df[self._x_column].values
        x_data = torch.tensor(x_data, dtype=torch.float32, device=self._device)
        # drop the repeated columns in other_df
        other_df = other_df.drop(columns=self._feature_column)
        # move first column from Tensor x_beta
        intercepts = x_betas[:, 0]
        x_betas = x_betas[:, 1:]
        if self._is_normalize_variable:
            # Convert the regression coefficients of the normalized X into the regression coefficients of the original X values
            # get max and min of x_data Tensor in each column
            x_max = torch.max(x_data, dim=0).values.repeat(x_betas.shape[0], 1)
            x_min = torch.min(x_data, dim=0).values.repeat(x_betas.shape[0], 1)
            x_betas = ((x_data + EPSILON - x_min) * (x_max - x_min + EPSILON) ** -1 + EPSILON) * (
                        x_data + EPSILON) ** -1 * x_betas

        results = torch.cat((y_hat.reshape(len(y_hat), 1), x_betas, intercepts.reshape(len(y_hat), 1)), 1)
        results = results.cpu().detach().numpy()
        result_columns = [self._y_column + '_hat']
        for i in range(len(self._x_column)):
            result_columns.append('beta_' + self._x_column[i])
        result_columns.append('intercept')
        result_df = pd.DataFrame(results, columns=result_columns)

        result_df = pd.concat([other_df, result_df], axis=1)

        indicators = indicators.cpu().detach().numpy()
        indicator_df = pd.DataFrame(indicators, columns=['RSS', 'RMSE', 'MAE', 'MAPE', 'R2', 'adj-R2'])
        dataset_s = pd.Series([f'TRAIN{icv}', f'VAL{icv}', f'TEST{icv}'])
        indicator_df.insert(0, 'dataset', dataset_s)

        # transform spatial_weights from N*N adjacency matrix to ijv format
        if spatial_weights is not None:
            spatial_weights = spatial_weights.cpu().detach().numpy()
            spatial_weights = sp.coo_matrix(spatial_weights)
            spatial_weights = pd.DataFrame(
                {'i': spatial_weights.row, 'j': spatial_weights.col, 'v': spatial_weights.data})

        writer = pd.ExcelWriter(path)
        result_df.to_excel(writer, sheet_name='result', index=False)
        indicator_df.to_excel(writer, sheet_name='indicator', index=False)
        if spatial_weights is not None:
            spatial_weights.to_csv(path_or_buf=path.replace('.xlsx', '_spatial_weights.csv'), index=False)
        writer.close()
        print(f"************ {result_name} ************")
        print(indicator_df)

        return indicator_df
