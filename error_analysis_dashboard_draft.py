import pandas as pd
import datetime as dt
from pathlib import Path
import panel as pn
import panel.widgets as pnw
pn.extension('plotly')
import param
import numpy as np
import utils

np.random.seed(1)

N = 100
d = {'predicted': np.random.randn(N), 'actual': np.random.randn(N)}
df_example = pd.DataFrame(data=d)

class BasicDashboard(param.Parameterized):
    X = param.ObjectSelector(default=df_example.columns.tolist()[0], objects=df_example.columns.tolist())
    Y = param.ObjectSelector(default=df_example.columns.tolist()[1], objects=df_example.columns.tolist())
    filter_by_col = param.ObjectSelector(default=df_example.columns.tolist()[0], objects=df_example.columns.tolist())
    filter_by_value = param.Integer(default=0, bounds=(0, 120), allow_None=True)

    def __init__(self, df, *args, **kwargs):
        super(type(self), self).__init__()
        self.df = df
        cls = type(self) #gets the class (BasicDashboard in this case)
        # cls.X = param.ObjectSelector(objects=self.df.columns.tolist())
        # cls.Y = param.ObjectSelector(objects=self.df.columns.tolist())
        # kwargs["X"] = param.ObjectSelector(objects=self.df.columns.tolist())
        # kwargs["Y"] = param.ObjectSelector(objects=self.df.columns.tolist())


    def get_data(self):
        class_df = self.df.copy()
        return class_df

    def filter_data_rows(self):
        class_df = self.get_data()
        class_df = class_df[class_df[self.filter_by_col] >= self.filter_by_value]
        #here use a function for preprocessing (e.g. filter cols or rows) that is defined in the utils.py. needs to be able to use parameters from outside.
        return class_df

    def create_error_df(self):
        error_df = self.filter_data_rows()
        error_df['error'] = error_df['actual'] - error_df['predicted']
        error_df['error_relative_to_predicted'] = error_df['error'] / error_df['predicted']
        error_df['error_relative_to_actual'] = error_df['error'] / error_df['actual']
        return error_df

    def metrics_table_view(self):
        class_df = self.create_error_df()
        metrics_dict = utils.calc_metrics_regression(class_df['actual'].to_numpy(), class_df['predicted'].to_numpy())
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
        return pn.pane.DataFrame(metrics_df)

    def scatter_view(self):
        class_df = self.get_data()
        array_x = class_df[self.X].to_numpy()
        array_y = class_df[self.Y].to_numpy()
        fig = utils.plot_scatter(x=array_x, y=array_y, add_unit_line=True, add_R2=True)
        return fig

    def plot_error_dist(self):
        #make distplots for error and relative error
        pass

    def plot_error_boxplot(self):
        #make boxplots for error and relative error
        pass

    def error_summary_stats_table(self):
        data = self.create_error_df()
        stats_df = data.describe(percentiles=[0.025, 0.2, 0.3, 0.5, 0.7, 0.8, 0.975]).T
        return pn.pane.DataFrame(stats_df, width=1000)

    @classmethod
    def set_class_attr(cls, df):
    #     self.choose_columns = param.ListSelector(objects=df.columns.tolist())
        cls.X = param.ObjectSelector(objects=df.columns.tolist())
        cls.Y = param.ObjectSelector(objects=df.columns.tolist())



dash = BasicDashboard(df_example)

dashboard = pn.Column(dash.param['X'],
                      dash.param['Y'],
                      dash.param['filter_by_col'],
                      dash.param['filter_by_value'],
                      dash.metrics_table_view,
                      dash.scatter_view,
                      )

print('stop here')
# class BasicDashboard(param.Parameterized):
#     # X = param.ObjectSelector(default=df_example.columns.tolist()[0], objects=df_example.columns.tolist())
#     # Y = param.ObjectSelector(default=df_example.columns.tolist()[1], objects=df_example.columns.tolist())
#     filter_by_col = param.ObjectSelector(default=df_example.columns.tolist()[0], objects=df_example.columns.tolist())
#     filter_by_value = param.Integer(default=0, bounds=(0, 120), allow_None=True)
#
#     def __init__(self, df, *args, **kwargs):
#         self.df = df
#         cls = type(self) #gets the class (BasicDashboard in this case)
#         cls.X = pnw.RadioButtonGroup(name='X', value=self.df.columns.tolist()[0], options=self.df.columns.tolist())
#         cls.Y = pnw.RadioButtonGroup(name='Y', value=self.df.columns.tolist()[0], options=self.df.columns.tolist())
#         # cls.X = param.ObjectSelector(default=self.df.columns.tolist()[0], objects=self.df.columns.tolist())
#         # cls.Y = param.ObjectSelector(default=self.df.columns.tolist()[0], objects=self.df.columns.tolist())
#         # kwargs["X"] = param.ObjectSelector(default=self.df.columns.tolist()[0],objects=self.df.columns.tolist())
#         # kwargs["Y"] = param.ObjectSelector(default=self.df.columns.tolist()[0], objects=self.df.columns.tolist())
#         # self.X = param.ObjectSelector(default=self.df.columns.tolist()[0], objects=self.df.columns.tolist())
#         # self.Y = param.ObjectSelector(default=self.df.columns.tolist()[0], objects=self.df.columns.tolist())
#         # super(type(self), self).__init__(*args, **kwargs)

#     def get_data(self):
#         class_df = self.df.copy()
#         return class_df
#
#     def filter_data_rows(self):
#         class_df = self.get_data()
#         class_df = class_df[class_df[self.filter_by_col] >= self.filter_by_value]
#         #here use a function for preprocessing (e.g. filter cols or rows) that is defined in the utils.py. needs to be able to use parameters from outside.
#         return class_df
#
#     def create_error_df(self):
#         error_df = self.filter_data_rows()
#         error_df['error'] = error_df['actual'] - error_df['predicted']
#         error_df['error_relative_to_predicted'] = error_df['error'] / error_df['predicted']
#         error_df['error_relative_to_actual'] = error_df['error'] / error_df['actual']
#         return error_df
#
#     def metrics_table_view(self):
#         class_df = self.create_error_df()
#         metrics_dict = utils.calc_metrics_regression(class_df['actual'].to_numpy(), class_df['predicted'].to_numpy())
#         metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
#         return pn.pane.DataFrame(metrics_df)
#
#     def scatter_view(self):
#         class_df = self.get_data()
#         array_x = class_df[self.X].to_numpy()
#         array_y = class_df[self.Y].to_numpy()
#         fig = utils.plot_scatter(x=array_x, y=array_y, add_unit_line=True, add_R2=True)
#         return fig
#
#     def plot_error_dist(self):
#         #make distplots for error and relative error
#         pass
#
#     def plot_error_boxplot(self):
#         #make boxplots for error and relative error
#         pass
#
#     def error_summary_stats_table(self):
#         data = self.create_error_df()
#         stats_df = data.describe(percentiles=[0.025, 0.2, 0.3, 0.5, 0.7, 0.8, 0.975]).T
#         return pn.pane.DataFrame(stats_df, width=1000)
#
#     @classmethod
#     def set_class_attr(cls, df):
#     #     self.choose_columns = param.ListSelector(objects=df.columns.tolist())
#         cls.X = param.ObjectSelector(objects=df.columns.tolist())
#         cls.Y = param.ObjectSelector(objects=df.columns.tolist())
#
#
#
# dash = BasicDashboard(df_example)
# # pn.Param(BasicDashboard.param, widgets={'X': pn.widgets.RadioButtonGroup})
#
# dashboard = pn.Column(dash.X,
#                       dash.Y,
#                       dash.param['filter_by_col'],
#                       dash.param['filter_by_value'],
#                       dash.metrics_table_view,
#                       dash.scatter_view,
#                       )

# import pandas as pd
# import datetime as dt
# from pathlib import Path
# import panel as pn
# import panel.widgets as pnw
# pn.extension('plotly')
# import param
# import numpy as np
# import sys
#
# sys.path.append(r'C:\Users\user\Google Drive\my projects\DS dashboard tool')
# import utils
#
# path = Path(r'C:\Users\user\Google Drive\Sadna_for_public_knowledge\BudgetKey\municipal_budget_data')
# fname = 'lamas-municipal-data.csv'
# df_muni = pd.read_csv(path / fname)
#
# np.random.seed(1)
#
# N = 100
# d = {'predicted': np.random.randn(N), 'actual': np.random.randn(N)}
# df = pd.DataFrame(data=d)
#
#
#
# '''
# panel without Param package. compiles but it is no interactive. not responding to widgets
# '''
# X = pnw.RadioButtonGroup(name='X', value=df.columns.tolist()[0], options=df.columns.tolist())
# Y = pnw.RadioButtonGroup(name='Y', value=df.columns.tolist()[0], options=df.columns.tolist())
# filter_by_col = pnw.RadioButtonGroup(name='filter_col', value=df.columns.tolist()[0], options=df.columns.tolist())
# filter_by_value = 0.5  # param.Integer(default=0, bounds=(0, 120), allow_None=True)
#
#
# def get_data(df):
#     class_df = df.copy()
#     return class_df
#
# @pn.depends(filter_by_col.param.value) #perhaps param.depends
# def filter_data_rows(df):
#     class_df = get_data(df)
#     class_df = class_df[class_df[filter_by_col.value] >= filter_by_value]
#     # here use a function for preprocessing (e.g. filter cols or rows) that is defined in the utils.py. needs to be able to use parameters from outside.
#     return class_df
#
#
# def create_error_df(df):
#     error_df = filter_data_rows(df)
#     error_df['error'] = error_df['actual'] - error_df['predicted']
#     error_df['error_relative_to_predicted'] = error_df['error'] / error_df['predicted']
#     error_df['error_relative_to_actual'] = error_df['error'] / error_df['actual']
#     return error_df
#
#
# def metrics_table_view(df):
#     class_df = create_error_df(df)
#     metrics_dict = utils.calc_metrics_regression(class_df['actual'].to_numpy(), class_df['predicted'].to_numpy())
#     metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
#     return pn.pane.DataFrame(metrics_df)
#
# @pn.depends(X.param.value, Y.param.value)
# def scatter_view(df,X, Y):
#     class_df = get_data(df)
#     array_x = class_df[X.value].to_numpy()
#     array_y = class_df[Y.value].to_numpy()
#     fig = utils.plot_scatter(x=array_x, y=array_y, add_unit_line=True, add_R2=True)
#     return fig
#
#
# dashboard = pn.Column(X,
#                       Y,
#                       filter_by_col,
#                       filter_by_value,
#                       metrics_table_view(df),
#                       scatter_view(df=df, X=X, Y=Y),
#                       )
#### it works but not interactive
print('not done yet')
print('done')