import pandas as pd
import plotly.express as px


df = pd.read_excel('shotBreakdown.xlsx')


def summarize_shot_data(df, group_col):
    """
    Summarizes shot data by the given grouping column(s), calculating
    total shots, made shots, and make percentage for each group.
    """
    summary = (
        df.groupby(group_col)
        .agg(
            total_shots=('result', 'count'),
            made_shots=('result', lambda x: (x == 'Make').sum()),
            shot_percentage=('result', lambda x: (x == 'Make').mean() * 100),
        )
        .reset_index()
    )
    summary['shot_percentage'] = summary['shot_percentage'].round(1)
    return summary.sort_values(by='shot_percentage', ascending=False)


def plot_shot_percentage(summary, x_col, color_col, title, palette):
    """
    Builds a grouped bar chart of shot percentage, split by x_col and
    colored by color_col, using the given color palette. Reused for
    every chart in this script.
    """
    fig = px.bar(
        summary,
        x=x_col,
        y='shot_percentage',
        color=color_col,
        color_discrete_sequence=palette,
        barmode='group',
        text='shot_percentage',
        hover_data={'total_shots': True, 'made_shots': True},
        title=title,
        template='plotly_white',
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(
        yaxis_title='Shot Percentage (%)',
        yaxis_range=[0, 100],
        xaxis_title=x_col,
        legend_title=color_col,
    )
    return fig

# Each entry: (columns to group by, x-axis column, color column, chart title, palette)
chart_specs = [
        (['shotType', 'contestLevel'], 'shotType', 'contestLevel',
         'Shot % by Shot Type and Contest Level', px.colors.qualitative.Set1),
        (['shotLocation', 'contestLevel'], 'shotLocation', 'contestLevel',
         'Shot % by Shot Location and Contest Level', px.colors.qualitative.Set2),
        (['Origin', 'shotType'], 'Origin', 'shotType',
         'Shot % by Origin and Shot Type', px.colors.qualitative.Pastel),
        (['Origin', 'shotLocation'], 'Origin', 'shotLocation',
         'Shot % by Origin and Shot Location', px.colors.qualitative.Bold),
        (['Origin', 'contestLevel'], 'Origin', 'contestLevel',
         'Shot % by Origin and Contest Level', px.colors.qualitative.Dark2),
        (['shotType', 'playType'], 'shotType', 'playType', 
         'Shot % by Shot Type and Play Type', px.colors.qualitative.Set3),
        (['shotLocation', 'playType'], 'shotLocation', 'playType',
          'Shot % by Shot Location and Play Type', px.colors.qualitative.Set1)
          
    ]

final = []
for group_cols, x_col, color_col, title, palette in chart_specs:
        summary = summarize_shot_data(df, group_cols)
        fig = plot_shot_percentage(summary, x_col, color_col, title, palette)
        final.append(fig)
        fig.show()