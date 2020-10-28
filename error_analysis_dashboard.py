import pandas as pd
import datetime as dt
from pathlib import Path
import panel as pn
import panel.widgets as pnw
pn.extension('plotly')
import param
import numpy as np
import sys
sys.path.append(r'C:\Users\user\Google Drive\my projects\DS dashboard tool')
import utils

# path = Path(r'C:\Users\user\Google Drive\Sadna_for_public_knowledge\BudgetKey\municipal_budget_data')
# fname = 'lamas-municipal-data.csv'
# df_muni = pd.read_csv(path/fname)

np.random.seed(1)

N = 100
d = {'predicted': np.random.randn(N), 'actual': np.random.randn(N)}
df_example = pd.DataFrame(data=d)
df_error = utils.make_error_df(actual=d['actual'], predicted=d['predicted'])

df_dashboard = df_error.copy()

class BasicDashboard(param.Parameterized):
    X = param.ObjectSelector(default=df_dashboard.columns.tolist()[0], objects=df_dashboard.columns.tolist())
    Y = param.ObjectSelector(default=df_dashboard.columns.tolist()[1], objects=df_dashboard.columns.tolist())
    filter_by_col = param.ObjectSelector(default=df_dashboard.columns.tolist()[0], objects=df_dashboard.columns.tolist())
    filter_by_value = param.Integer(default=0, bounds=(0, 120), allow_None=True)

    def __init__(self, df, *args, **kwargs):
        self.df = df
        #add assert that headers are 'actual' and 'predicted', or specify colnames that correspond to them.

        cls = type(self) #gets the class (BasicDashboard in this case)
        # cls.X = param.ObjectSelector(objects=self.df.columns.tolist())
        # cls.Y = param.ObjectSelector(objects=self.df.columns.tolist())
        # kwargs["X"] = param.ObjectSelector(objects=self.df.columns.tolist())
        # kwargs["Y"] = param.ObjectSelector(objects=self.df.columns.tolist())
        super(type(self), self).__init__(*args, **kwargs) #*args, **kwargs


    def filter_data_rows(self):
        data = self.df.copy()
        data = utils.filter_df_rows(df=data, filter_by_col=self.filter_by_col, filter_by_value=self.filter_by_value)
        return data

    def metrics_table_view(self):
        data = self.filter_data_rows()
        metrics_dict = utils.calc_metrics_regression(data['actual'].to_numpy(), data['predicted'].to_numpy())
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
        return pn.pane.DataFrame(metrics_df)

    def scatter_view(self):
        data = self.filter_data_rows()
        array_x = data[self.X].to_numpy()
        array_y = data[self.Y].to_numpy()
        fig = utils.plot_scatter(x=array_x, y=array_y, add_unit_line=True, add_R2=True)
        return fig

    def plot_error_dist(self):
        #make distplots for error and relative error
        pass

    def plot_error_boxplot(self):
        #make boxplots for error and relative error
        pass

    def error_summary_stats_table(self):
        data = self.filter_data_rows()
        stats_df = data.describe(percentiles=[0.025, 0.2, 0.3, 0.5, 0.7, 0.8, 0.975]).T
        return pn.pane.DataFrame(stats_df, width=1000)

    # @classmethod
    # def set_class_attr(cls, df):
    # #     self.choose_columns = param.ListSelector(objects=df.columns.tolist())
    #     cls.X = param.ObjectSelector(default=df.columns.tolist()[0], objects=df.columns.tolist())
    #     cls.Y = param.ObjectSelector(default=df.columns.tolist()[1], objects=df.columns.tolist())



dash = BasicDashboard(df_dashboard)
# pn.Param(BasicDashboard.param, widgets={'X': pn.widgets.RadioButtonGroup})
dashboard = pn.Column(dash.param['X'],
                      dash.param['Y'],
                      dash.param['filter_by_col'],
                      dash.param['filter_by_value'],
                      dash.error_summary_stats_table,
                      dash.metrics_table_view,
                      dash.scatter_view,
                      )


print('not done yet')
print('done')

'''
I would like to have an ObjectSelector where the optional values are column names of a dataframe that is created by another parameter set by the user.
Or have the range of a parameter dependent on the input data to the dashboard.'''