import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

N = 100
d = {'predicted': np.random.randn(N), 'actual': np.random.randn(N)}
df_example = pd.DataFrame(data=d)

def plot_scatter(x, y, add_unit_line=True, add_R2=True, layout_kwargs=None):
    '''
    :param x: numpy array
    :param y: numpy array
    :param add_unit_line: boolean. whether to add a line x=y
    :param add_R2: boolean. whether to add text of R2 on the plot
    :param layout_kwargs: a dictionary, according to plotly API:
    https://plotly.com/python/figure-labels/#manual-labelling-with-graph-objects
    https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html?highlight=update%20layout#plotly.graph_objects.Figure.update_layout
    https://plotly.com/python-api-reference/generated/plotly.graph_objects.Layout.html
    for example:
    1. layout_kwargs = dict(title='scatter', xaxis_title='X', legend_title='legend', showlegend=True)
    2. layout_kwargs = dict(title=dict(text='scatter'), xaxis=dict(title={'text': 'X'}))
    :return: plotly figure object
    '''
    # need to add titles, colors, fonts, (in **kwargs), option to add more traces (perhaps input should be tuples of [(x1,y1), (x2,y2)...])
    #add layout_kwargs for 'showlegend', 'title', 'xaxis_title', 'legend_title', 'fonts', etc.)
    unit_line = [np.nanmin(np.append(x, y)), np.nanmax(np.append(x, y))]
    # calculate explained variance:
    from sklearn.metrics import explained_variance_score
    R2 = round(explained_variance_score(x, y), 3)
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='markers',
                             name='',
                            #  marker=dict(
                            #     color=color='rgb(139,0,139)',
                            #     colorscale='Viridis',
                            #     line_width=None
                            # ) #accept input of marker_kwargs
    ))

    if add_R2:
        fig.add_annotation(
            x=unit_line[0],
            y=unit_line[1],
            showarrow=False,
            text=f'R2 score is: {R2}')
    # fig.add_trace(go.Scatter(x=random_x, y=random_y1,
    #                          mode='lines+markers',
    #                          name='lines+markers'))
    if add_unit_line:
        fig.add_trace(go.Scatter(x=unit_line, y=unit_line,
                                 mode='lines',
                                 name=''))
    fig.update_layout(showlegend=False)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    # Set options common to all traces with fig.update_traces
    # fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    # fig.update_layout(title='Styled Scatter',
    #                   yaxis_zeroline=False, xaxis_zeroline=False)
    return fig


# layout_kwargs = dict(title=dict(text='scatter'), xaxis=dict(title={'text': 'X'})) #dict(showlegend=True) #dict(title='scatter', xaxis_title='X', legend_title='legend', showlegend=True)
# fig = plot_scatter(x=d['actual'], y=d['predicted'], layout_kwargs=layout_kwargs)



from sklearn.metrics import explained_variance_score, mean_absolute_error

def calc_metrics_regression(actual, predicted, digits=3):
    R2_score = round(explained_variance_score(actual, predicted), digits)
    # mae = round(mean_absolute_error(actual, predicted), digits)
    mae = round(np.nanmean(np.abs(predicted - actual)), digits)
    mse = round(np.nanmean((predicted - actual) ** 2), digits)
    rmse = round(np.nanmean((predicted - actual) ** 2) ** .5, digits)
    corr = round(np.corrcoef(predicted, actual)[0, 1], digits)
    mape = round(np.nanmean(np.abs(predicted - actual) / np.abs(actual)), digits)
    mdape = round(np.median(np.abs(predicted - actual) / np.abs(actual)), digits)
    return({'mape': mape, 'mdape': mdape, 'mae': mae,
            'mse': mse, 'rmse': rmse,
            'corr': corr, 'R2_score': R2_score})


def filter_df_rows(df, cutoff_by_column=None, cutoff_value=None):
    # need to verify that if cutoff_by_column or cutoff_value are None then the function returns the dataframe intact.
    if cutoff_by_column is None:
        cutoff_by_column = df.columns.to_list()[0]
    if cutoff_value is None:
        # cutoff_value = np.min(df[cutoff_by_column])
        return df
    else:
        return df[df[cutoff_by_column] >= cutoff_value]


# out = filter_data_rows(df_example, cutoff_by_column=df_example.columns.to_list()[1], cutoff_value=0)
# print(out.head())


def make_error_df(actual, predicted):
    #actual and predicted are numpy arrays
    dict = {'actual': actual, 'predicted': predicted}
    error_df = pd.DataFrame(data=dict)
    error_df['error'] = error_df['actual'] - error_df['predicted']
    error_df['error_relative_to_predicted'] = error_df['error'] / error_df['predicted']
    error_df['error_relative_to_actual'] = error_df['error'] / error_df['actual']
    return error_df


print('done')
print('done')