import pandas as pd
import datetime as dt
from pathlib import Path
import panel as pn
import panel.widgets as pnw
pn.extension('plotly')
import plotly.express as px
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
    cutoff_by_column = param.ObjectSelector(default=df_dashboard.columns.tolist()[0], objects=df_dashboard.columns.tolist())
    cutoff_value = param.Number(default=0, bounds=(0, 120), allow_None=True)
    var_to_inspect = param.ObjectSelector(default=df_dashboard.columns.tolist()[0], objects=df_dashboard.columns.tolist())
    updated_param = param.ObjectSelector(default='default', objects=['default', 'default1', 'default2'])


    def __init__(self, df, *args, **kwargs):
        self.df = df
        #add assert that headers are 'actual' and 'predicted', or specify colnames that correspond to them.

        cls = type(self) #gets the class (BasicDashboardComponents in this case)
        # cls.X = param.ObjectSelector(objects=self.df.columns.tolist())
        # cls.Y = param.ObjectSelector(objects=self.df.columns.tolist())
        # kwargs["X"] = param.ObjectSelector(objects=self.df.columns.tolist())
        # kwargs["Y"] = param.ObjectSelector(objects=self.df.columns.tolist())
        super(type(self), self).__init__(*args, **kwargs) #*args, **kwargs


    def filter_data_rows(self):
        data = self.df.copy()
        data = utils.filter_df_rows(df=data, cutoff_by_column=self.cutoff_by_column, cutoff_value=self.cutoff_value)
        return data

    def metrics_table_view(self):
        data = self.filter_data_rows()
        metrics_dict = utils.calc_metrics_regression(data['actual'].to_numpy(), data['predicted'].to_numpy())
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
        return pn.pane.DataFrame(metrics_df, width=1350)

    def scatter_view(self):
        data = self.filter_data_rows()
        array_x = data[self.X].to_numpy()
        array_y = data[self.Y].to_numpy()
        fig = utils.plot_scatter(x=array_x, y=array_y,
                                 add_unit_line=True, add_R2=True,
                                 layout_kwargs=dict(title='', xaxis_title=self.X, yaxis_title=self.Y))
        return fig

    def plot_error_dist(self):
        #make distplots for error and relative error
        pass

    def plot_error_boxplot(self):
        #make boxplots for error and relative error
        pass

    # @param.depends('var_to_inspect', watch=True)
    def error_boxplot_view(self):
        data = self.filter_data_rows()
        fig = px.box(data, y=self.var_to_inspect, color_discrete_sequence=['green'])
        return fig

    def error_summary_stats_table(self):
        from bokeh.models.widgets.tables import NumberFormatter
        data = self.filter_data_rows()
        stats_df = data.describe(percentiles=[0.025, 0.2, 0.3, 0.5, 0.7, 0.8, 0.975]).T
        stats_df = stats_df.round(3)
        formatter = NumberFormatter(format='0.000')
        return pn.pane.DataFrame(stats_df, width=1350, formatters={'float': formatter}, widths={'index': 300})
        # return pn.widgets.DataFrame(stats_df, width=1000, formatters={'float': formatter}, widths={'index': 300})


    def wrap_param_var_to_inspect(self):
        return pn.Param(self.param, parameters=['var_to_inspect'],
                 name='Choose variable to inspect distribution',
                 show_name=True,
                 widgets={'var_to_inspect': {'type': pn.widgets.RadioButtonGroup}},
                 width=300)

    @param.depends('X', watch=True)
    def _update_param(self):
        self.param['updated_param'].objects = self.df.columns.tolist()
        self.updated_param = self.df.columns.tolist()[0]
        # self.param['updated_param'].objects = self.param['X'].objects
        # self.updated_param = self.X


    @param.depends('cutoff_by_column', watch=True)
    def _update_cutoff_value(self):
        self.param['cutoff_value'].bounds = (self.df[self.cutoff_by_column].min(), self.df[self.cutoff_by_column].max())
        self.cutoff_value = self.df[self.cutoff_by_column].min()


# pn.Param(BasicDashboardComponents.param, parameters=['var_to_inspect'],
#                                  name='Choose variable to inspect distribution',
#                                  show_name=True,
#                                  widgets={'var_to_inspect': {'type': pn.widgets.RadioButtonGroup}},
#                                  width =500),

dash = BasicDashboard(df_dashboard)
# pn.Param(BasicDashboardComponents.param, widgets={'X': pn.widgets.RadioButtonGroup})

widgets_panel = pn.Column(dash.param['X'],
                          dash.param['Y'],
                          dash.param['cutoff_by_column'],
                          dash.param['cutoff_value'], width=200)

scatter_panel = pn.Column(pn.pane.Markdown('## '),
                          pn.pane.Markdown('## '),
                          dash.scatter_view)
boxplot_panel = pn.Column(dash.wrap_param_var_to_inspect,
                          dash.error_boxplot_view, width=300)

plots_panel = pn.Row(scatter_panel, boxplot_panel, width=1000)

tables_panel = pn.Column(pn.pane.Markdown('## Model Performance Metrics', style={'font-family': "serif"}),
                         dash.metrics_table_view,
                         pn.pane.Markdown('## Descriptive Stats Table', style={'font-family': "serif"}),
                         dash.error_summary_stats_table)

dashboard = pn.Column(pn.pane.Markdown('# Error Analysis Dashboard', style={'font-family': "serif"}),
                      pn.Row(widgets_panel, plots_panel),
                            tables_panel)

# dashboard = pn.Column(dash.param['X'],
#                       dash.param['Y'],
#                       dash.param['cutoff_by_column'],
#                       dash.param['cutoff_value'],
#                       dash.error_summary_stats_table,
#                       dash.metrics_table_view,
#                       dash.scatter_view,
#                       dash.wrap_param_var_to_inspect,
#                       dash.error_boxplot_view,
#                       dash.param['updated_param'],
#                       )

dashboard.show()
print('not done yet')
print('done')

'''
I would like to have an ObjectSelector where the optional values are column names of a dataframe that is created by another parameter set by the user.
Or have the range of a parameter dependent on the input data to the dashboard.'''