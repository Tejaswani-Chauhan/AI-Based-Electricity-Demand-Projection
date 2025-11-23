# Set matplotlib backend to Agg before any other imports
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import io
import base64
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
from statsmodels.tsa.api import Holt
from prophet import Prophet
import atexit
from datetime import datetime
import calendar

# Plotly imports with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will not work.")
app = Flask(__name__)
warnings.filterwarnings("ignore")

# ----------------------------
# DATA LOADING AND PREPROCESSING
# ----------------------------
df1 = pd.read_csv("Daily_Power_Gen_Source_march_23.csv")
df1_source = df1.drop(df1[df1['source'] == 'Total'].index)
df1_source['date'] = pd.to_datetime(df1['date'])
df1_source['day'] = df1_source['date'].dt.day
df1_source['month'] = df1_source['date'].dt.month
df1_source['year'] = df1_source['date'].dt.year
df1_source['month_name'] = df1_source['date'].dt.month_name()

df2 = pd.read_csv("Daily_Power_Gen_States_march_23.csv")
df2['Shortage during maximum Demand(MW)'] = df2['Shortage during maximum Demand(MW)'].fillna(df2['Shortage during maximum Demand(MW)'].mean())
df2['Energy Met (MU)'] = df2['Energy Met (MU)'].fillna(0)
df2['date'] = pd.to_datetime(df2['date'])
df2['day'] = df2['date'].dt.day
df2['month'] = df2['date'].dt.month
df2['year'] = df2['date'].dt.year
df2['month_name'] = df2['date'].dt.month_name()
df2.rename(columns={'Energy Met (MU)': 'Energy Met (MW)'}, inplace=True)
df2['Energy Met (MW)'] = df2['Energy Met (MW)'] * 1000

# Ensure month names are in correct order
month_order = list(calendar.month_name)[1:]
df1_source['month_name'] = pd.Categorical(df1_source['month_name'], categories=month_order, ordered=True)
df2['month_name'] = pd.Categorical(df2['month_name'], categories=month_order, ordered=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def cleanup_plots():
    plt.close('all')

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics with enhanced accuracy calculation"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'mse': mse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    relative_error = np.mean(np.abs(y_true - y_pred) / (np.max(y_true) - np.min(y_true)))
    metrics['accuracy'] = max(0, min(100, (1 - relative_error) * 100))
    
    # Add rating based on accuracy
    if metrics['accuracy'] >= 90:
        metrics['rating'] = 'Excellent'
        metrics['rating_color'] = 'success'
    elif metrics['accuracy'] >= 75:
        metrics['rating'] = 'Good'
        metrics['rating_color'] = 'primary'
    elif metrics['accuracy'] >= 50:
        metrics['rating'] = 'Fair'
        metrics['rating_color'] = 'warning'
    else:
        metrics['rating'] = 'Poor'
        metrics['rating_color'] = 'danger'
    
    return metrics

# ----------------------------
# GRAPH GENERATION FUNCTIONS
# ----------------------------
def generate_plotly_figure(graph_type, filtered_df1, filtered_df2, source, region, state, year, month):
    """Generate interactive Plotly figure with enhanced explanations and metrics"""
    explanation = ""
    model_metrics = None
    
    if graph_type == 'time_series':
        if source and source != 'All':
            if region and region != 'All':
                fig = px.line(filtered_df1, x='date', y=region, 
                             title=f'{source} Power Generation in {region} Over Time',
                             labels={'date': 'Date', region: 'Power Generation (MW)'})
            else:
                fig = px.line(filtered_df1, x='date', y='All India',
                             title=f'{source} Power Generation Over Time',
                             labels={'date': 'Date', 'All India': 'Power Generation (MW)'})
        else:
            if region and region != 'All':
                fig = px.line(filtered_df1, x='date', y=region, color='source',
                            title=f'Power Generation by Source in {region}',
                            labels={'date': 'Date', region: 'Power Generation (MW)', 'source': 'Power Source'})
            else:
                fig = px.line(filtered_df1, x='date', y='All India', color='source',
                            title='Power Generation by Source Over Time',
                            labels={'date': 'Date', 'All India': 'Power Generation (MW)', 'source': 'Power Source'})
        
        # Add range slider and better formatting
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        explanation = f"""
        <div class="alert alert-info">
            <h5><i class="bi bi-info-circle"></i> Time Series Analysis Insights</h5>
            <ul>
                <li>The graph shows power generation trends from {filtered_df1['date'].min().strftime('%Y-%m-%d')} to {filtered_df1['date'].max().strftime('%Y-%m-%d')}</li>
                <li>Peaks indicate periods of high electricity demand (typically summer months)</li>
                <li>Troughs show periods of low demand (typically monsoon/winter seasons)</li>
                <li>Seasonal patterns may be visible in the data (use zoom to examine)</li>
                <li>Hover over data points for exact values and dates</li>
            </ul>
        </div>
        """
        
    elif graph_type == 'bar_chart':
        if source and source != 'All':
            if region and region != 'All':
                agg_df = filtered_df1.groupby('month_name')[region].mean().reset_index()
                fig = px.bar(agg_df, x='month_name', y=region, 
                            title=f'Average Monthly {source} Power Generation in {region}',
                            labels={'month_name': 'Month', region: 'Average Power Generation (MW)'})
            else:
                agg_df = filtered_df1.groupby('month_name')['All India'].mean().reset_index()
                fig = px.bar(agg_df, x='month_name', y='All India', 
                            title=f'Average Monthly {source} Power Generation',
                            labels={'month_name': 'Month', 'All India': 'Average Power Generation (MW)'})
        else:
            if region and region != 'All':
                agg_df = filtered_df1.groupby(['month_name', 'source'])[region].mean().reset_index()
                fig = px.bar(agg_df, x='month_name', y=region, color='source', barmode='group',
                            title=f'Average Monthly Power Generation by Source in {region}',
                            labels={'month_name': 'Month', region: 'Average Power Generation (MW)', 'source': 'Power Source'})
            else:
                agg_df = filtered_df1.groupby(['month_name', 'source'])['All India'].mean().reset_index()
                fig = px.bar(agg_df, x='month_name', y='All India', color='source', barmode='group',
                            title='Average Monthly Power Generation by Source',
                            labels={'month_name': 'Month', 'All India': 'Average Power Generation (MW)', 'source': 'Power Source'})
        
        explanation = f"""
        <div class="alert alert-info">
            <h5><i class="bi bi-info-circle"></i> Monthly Average Analysis</h5>
            <ul>
                <li>Shows average power generation by month across selected period</li>
                <li>Helps identify seasonal patterns in power generation</li>
                <li>Grouped bars show comparison between different sources</li>
                <li>Hover over bars for exact values</li>
            </ul>
        </div>
        """
    
    elif graph_type == 'pie_chart':
        if source and source != 'All':
            explanation = "<div class='alert alert-warning'>Pie charts are only available when showing all sources</div>"
            fig = px.pie(title='Select "All Sources" to view composition')
        else:
            if region and region != 'All':
                agg_df = filtered_df1.groupby('source')[region].sum().reset_index()
                fig = px.pie(agg_df, values=region, names='source',
                            title=f'Power Source Composition in {region}',
                            hole=0.3)
            else:
                agg_df = filtered_df1.groupby('source')['All India'].sum().reset_index()
                fig = px.pie(agg_df, values='All India', names='source',
                            title='All India Power Source Composition',
                            hole=0.3)
            
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-info-circle"></i> Power Source Composition</h5>
                <ul>
                    <li>Shows relative contribution of each power source</li>
                    <li>Helps understand energy mix in selected region/period</li>
                    <li>Hover over slices for exact percentages</li>
                    <li>Donut chart format improves readability</li>
                </ul>
            </div>
            """
    
    elif graph_type == 'box_plot':
        if source and source != 'All':
            if region and region != 'All':
                fig = px.box(filtered_df1, x='month_name', y=region,
                           title=f'Distribution of {source} Power Generation in {region} by Month',
                           labels={'month_name': 'Month', region: 'Power Generation (MW)'})
            else:
                fig = px.box(filtered_df1, x='month_name', y='All India',
                           title=f'Distribution of {source} Power Generation by Month',
                           labels={'month_name': 'Month', 'All India': 'Power Generation (MW)'})
        else:
            if region and region != 'All':
                fig = px.box(filtered_df1, x='source', y=region,
                           title=f'Distribution of Power Generation by Source in {region}',
                           labels={'source': 'Power Source', region: 'Power Generation (MW)'})
            else:
                fig = px.box(filtered_df1, x='source', y='All India',
                           title='Distribution of Power Generation by Source',
                           labels={'source': 'Power Source', 'All India': 'Power Generation (MW)'})
        
        explanation = f"""
        <div class="alert alert-info">
            <h5><i class="bi bi-info-circle"></i> Distribution Analysis</h5>
            <ul>
                <li>Shows statistical distribution of power generation values</li>
                <li>Box represents interquartile range (IQR)</li>
                <li>Whiskers show 1.5*IQR from the quartiles</li>
                <li>Outliers are shown as individual points</li>
                <li>Helps identify variability and anomalies</li>
            </ul>
        </div>
        """
    
    elif graph_type == 'heatmap':
        if state and state != 'All':
            # State-wise heatmap of demand vs shortage
            pivot_df = filtered_df2.pivot_table(index='month_name', columns='year', 
                                              values='Max.Demand Met during the day(MW)', aggfunc='mean')
            fig = px.imshow(pivot_df, 
                           labels=dict(x="Year", y="Month", color="Power Demand (MW)"),
                           title=f'Monthly Power Demand Heatmap for {state}')
        elif region and region != 'All':
            # Region-wise heatmap by month and year
            pivot_df = filtered_df1.pivot_table(index='month_name', columns='year', 
                                              values=region, aggfunc='mean')
            fig = px.imshow(pivot_df, 
                           labels=dict(x="Year", y="Month", color="Power Generation (MW)"),
                           title=f'Monthly Power Generation Heatmap for {region}')
        else:
            # All-India heatmap by month and year
            pivot_df = filtered_df1.pivot_table(index='month_name', columns='year', 
                                              values='All India', aggfunc='mean')
            fig = px.imshow(pivot_df, 
                           labels=dict(x="Year", y="Month", color="Power Generation (MW)"),
                           title='Monthly Power Generation Heatmap for All India')
        
        explanation = f"""
        <div class="alert alert-info">
            <h5><i class="bi bi-info-circle"></i> Heatmap Analysis</h5>
            <ul>
                <li>Visualizes patterns across months and years</li>
                <li>Warmer colors indicate higher values</li>
                <li>Helps identify seasonal patterns and long-term trends</li>
                <li>Hover over cells for exact values</li>
            </ul>
        </div>
        """
    
    elif graph_type == 'prediction':
        # Prepare data for prediction
        if state and state != 'All':
            y = filtered_df2['Max.Demand Met during the day(MW)']
            X = filtered_df2[['Energy Met (MW)', 'Shortage during maximum Demand(MW)']]
            title = f"Energy Demand Prediction for {state}"
            data_description = f"state {state}"
        elif region and region != 'All':
            y = filtered_df1[region]
            X = filtered_df1[['day', 'month', 'year']]
            title = f"Energy Demand Prediction for {region} Region"
            data_description = f"{region} region"
        else:
            y = filtered_df1['All India']
            X = filtered_df1[['day', 'month', 'year']]
            title = "All India Energy Demand Prediction"
            data_description = "all India"
        
        train_size = int(len(y) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=y_train.index,
            y=y_train,
            mode='lines',
            name='Train Data',
            line=dict(color='blue'),
            hovertemplate='Index: %{x}<br>Value: %{y:.2f} MW<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_test,
            mode='lines',
            name='Test Data',
            line=dict(color='pink'),
            hovertemplate='Index: %{x}<br>Value: %{y:.2f} MW<extra></extra>'
        ))
        
        # Train and plot models with enhanced parameters
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
            'Holt-Winters': Holt(y_train)
        }
        
        colors = ['green', 'red', 'purple', 'orange']
        model_metrics = {}
        
        for i, (name, model) in enumerate(models.items()):
            try:
                if name == 'Holt-Winters':
                    fit = model.fit(smoothing_level=0.8, smoothing_trend=0.02)
                    pred = fit.forecast(len(y_test))
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                
                metrics = calculate_metrics(y_test, pred)
                model_metrics[name] = metrics
                
                fig.add_trace(go.Scatter(
                    x=y_test.index,
                    y=pred,
                    mode='lines',
                    name=f'{name} ({metrics["rating"]})',
                    line=dict(color=colors[i], dash='dash'),
                    hovertemplate='Index: %{x}<br>Predicted: %{y:.2f} MW<extra></extra>'
                ))
            except Exception as e:
                warnings.warn(f"Error fitting {name}: {str(e)}")
                continue
        
        # Update layout with enhanced information
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Time Index',
            yaxis_title='Power Demand (MW)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(b=100),
            hovermode="x unified",
            annotations=[
                dict(
                    x=0.5,
                    y=-0.3,
                    xref='paper',
                    yref='paper',
                    text="Closer lines between actual (pink) and predicted (dashed) indicate better performance",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        # Generate detailed explanation with model performance
        best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        explanation = f"""
        <div class="alert alert-info">
            <h5><i class="bi bi-lightbulb"></i> Prediction Analysis for {data_description}</h5>
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-body">
                            <h6 class="card-title"><i class="bi bi-trophy-fill text-warning"></i> Best Model</h6>
                            <div class="d-flex align-items-center">
                                <h4 class="mb-0">{best_model[0]}</h4>
                                <span class="badge bg-{best_model[1]['rating_color']} ms-2">{best_model[1]['rating']}</span>
                            </div>
                            <p class="mb-0 small text-muted">Accuracy: {best_model[1]['accuracy']:.1f}%</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-body">
                            <h6 class="card-title"><i class="bi bi-info-circle"></i> Model Guide</h6>
                            <ul class="small mb-0">
                                <li>90-100%: Excellent</li>
                                <li>75-89%: Good</li>
                                <li>50-74%: Fair</li>
                                <li>Below 50%: Poor</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-2">
                <p class="mb-1"><strong>Analysis Notes:</strong></p>
                <ul class="mb-0">
                    <li>Linear models work best for clear trends</li>
                    <li>Random Forest captures complex patterns</li>
                    <li>Holt-Winters excels with seasonality</li>
                    <li>SVR requires careful parameter tuning</li>
                </ul>
            </div>
        </div>
        """
    
    return fig, model_metrics, explanation

# ----------------------------
# FLASK ROUTES (INTEGRATION POINTS)
# ----------------------------

@app.route('/')
def index():
    # Get only sources that have data
    sources = sorted(df1_source['source'].unique().tolist())
    
    # Regions available in the dataset
    regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']
    
    # Get years with data
    years = sorted(df1_source['year'].unique().tolist())
    
    # Get months with data
    months = sorted(df1_source['month'].unique().tolist())
    
    # Get states with data (excluding NaN values)
    states = sorted(df2['States'].dropna().unique().tolist())
    
    stats = {
        'years_count': df1_source['year'].nunique(),
        'sources_count': df1_source['source'].nunique(),
        'states_count': df2['States'].nunique(),
        'total_generation': "{:,.0f}".format(df1_source['All India'].sum()/1000) + "K",
        'renewable_growth': "{:.1f}x".format(
            df1_source[df1_source['source'].str.contains('Solar')]['All India'].sum() / 
            df1_source[df1_source['source'].str.contains('Solar') & 
                      (df1_source['year'] == 2015)]['All India'].sum()
        ),
        'western_region_share': "{:.0f}%".format(
            df1_source['WR'].sum() / df1_source['All India'].sum() * 100
        ),
        'plotly_available': PLOTLY_AVAILABLE
    }
    
    return render_template('index.html', 
                         sources=sources,
                         regions=regions,
                         years=years,
                         months=months,
                         states=states,
                         stats=stats)

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    cleanup_plots()
    
    try:
        graph_type = request.form.get('graph_type')
        interactive = request.form.get('interactive') == 'true' and PLOTLY_AVAILABLE
        source = request.form.get('source')
        region = request.form.get('region')
        year = request.form.get('year')
        month = request.form.get('month')
        state = request.form.get('state')
        
        filtered_df1 = df1_source.copy()
        filtered_df2 = df2.copy()
        
        if source and source != 'All':
            filtered_df1 = filtered_df1[filtered_df1['source'] == source]
        
        if region and region != 'All':
            y_col = region if region != 'All India' else 'All India'
        
        if year and year != 'All':
            year = int(year)
            filtered_df1 = filtered_df1[filtered_df1['year'] == year]
            filtered_df2 = filtered_df2[filtered_df2['year'] == year]
        
        if month and month != 'All':
            month = int(month)
            filtered_df1 = filtered_df1[filtered_df1['month'] == month]
            filtered_df2 = filtered_df2[filtered_df2['month'] == month]
        
        if state and state != 'All':
            filtered_df2 = filtered_df2[filtered_df2['States'] == state]

        if interactive:
            fig, model_metrics, explanation = generate_plotly_figure(
                graph_type, filtered_df1, filtered_df2, 
                source, region, state, year, month
            )
            return jsonify({
                'plotly_json': fig.to_json(),
                'model_metrics': model_metrics,
                'explanation': explanation,
                'interactive': True
            })
        
        # Static plot generation
        fig, ax = plt.subplots(figsize=(12, 7))
        model_metrics = {}
        explanation = ""
        
        if graph_type == 'time_series':
            if source and source != 'All':
                if region and region != 'All':
                    ax.plot(filtered_df1['date'], filtered_df1[y_col])
                    ax.set_title(f'{source} Power Generation in {region} Over Time')
                else:
                    ax.plot(filtered_df1['date'], filtered_df1['All India'])
                    ax.set_title(f'{source} Power Generation Over Time')
            else:
                if region and region != 'All':
                    sns.lineplot(data=filtered_df1, x='date', y=y_col, hue='source', ax=ax)
                    ax.set_title(f'Power Generation by Source in {region}')
                else:
                    sns.lineplot(data=filtered_df1, x='date', y='All India', hue='source', ax=ax)
                    ax.set_title('Power Generation by Source Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Power Generation (MW)')
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-info-circle"></i> Time Series Analysis</h5>
                <ul>
                    <li>Showing data from {filtered_df1['date'].min().strftime('%Y-%m-%d')} to {filtered_df1['date'].max().strftime('%Y-%m-%d')}</li>
                    <li>Vertical axis shows power generation in Megawatts (MW)</li>
                    <li>Multiple sources shown when "All Sources" selected</li>
                    <li>Seasonal patterns may be visible in the data</li>
                </ul>
            </div>
            """
            
        elif graph_type == 'bar_chart':
            if source and source != 'All':
                if region and region != 'All':
                    agg_df = filtered_df1.groupby('month_name')[y_col].mean().reset_index()
                    sns.barplot(data=agg_df, x='month_name', y=y_col, ax=ax)
                    ax.set_title(f'Average Monthly {source} Power Generation in {region}')
                else:
                    agg_df = filtered_df1.groupby('month_name')['All India'].mean().reset_index()
                    sns.barplot(data=agg_df, x='month_name', y='All India', ax=ax)
                    ax.set_title(f'Average Monthly {source} Power Generation')
            else:
                if region and region != 'All':
                    agg_df = filtered_df1.groupby(['month_name', 'source'])[y_col].mean().reset_index()
                    sns.barplot(data=agg_df, x='month_name', y=y_col, hue='source', ax=ax)
                    ax.set_title(f'Average Monthly Power Generation by Source in {region}')
                else:
                    agg_df = filtered_df1.groupby(['month_name', 'source'])['All India'].mean().reset_index()
                    sns.barplot(data=agg_df, x='month_name', y='All India', hue='source', ax=ax)
                    ax.set_title('Average Monthly Power Generation by Source')
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Power Generation (MW)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-info-circle"></i> Monthly Average Analysis</h5>
                <ul>
                    <li>Shows average power generation by month</li>
                    <li>Helps identify seasonal patterns</li>
                    <li>Grouped bars show comparison between sources</li>
                </ul>
            </div>
            """
            
        elif graph_type == 'pie_chart':
            if source and source != 'All':
                explanation = "<div class='alert alert-warning'>Pie charts are only available when showing all sources</div>"
                ax.text(0.5, 0.5, 'Select "All Sources" to view composition', 
                       ha='center', va='center', fontsize=12)
                ax.set_title('Power Source Composition')
                ax.axis('off')
            else:
                if region and region != 'All':
                    agg_df = filtered_df1.groupby('source')[y_col].sum().reset_index()
                    ax.pie(agg_df[y_col], labels=agg_df['source'], autopct='%1.1f%%',
                          wedgeprops=dict(width=0.3))
                    ax.set_title(f'Power Source Composition in {region}')
                else:
                    agg_df = filtered_df1.groupby('source')['All India'].sum().reset_index()
                    ax.pie(agg_df['All India'], labels=agg_df['source'], autopct='%1.1f%%',
                          wedgeprops=dict(width=0.3))
                    ax.set_title('All India Power Source Composition')
                
                explanation = f"""
                <div class="alert alert-info">
                    <h5><i class="bi bi-info-circle"></i> Power Source Composition</h5>
                    <ul>
                        <li>Shows relative contribution of each power source</li>
                        <li>Helps understand energy mix</li>
                        <li>Donut chart format improves readability</li>
                    </ul>
                </div>
                """
                
        elif graph_type == 'box_plot':
            if source and source != 'All':
                if region and region != 'All':
                    sns.boxplot(data=filtered_df1, x='month_name', y=y_col, ax=ax)
                    ax.set_title(f'Distribution of {source} Power Generation in {region} by Month')
                else:
                    sns.boxplot(data=filtered_df1, x='month_name', y='All India', ax=ax)
                    ax.set_title(f'Distribution of {source} Power Generation by Month')
            else:
                if region and region != 'All':
                    sns.boxplot(data=filtered_df1, x='source', y=y_col, ax=ax)
                    ax.set_title(f'Distribution of Power Generation by Source in {region}')
                else:
                    sns.boxplot(data=filtered_df1, x='source', y='All India', ax=ax)
                    ax.set_title('Distribution of Power Generation by Source')
            
            ax.set_xlabel('Source' if source == 'All' else 'Month')
            ax.set_ylabel('Power Generation (MW)')
            plt.xticks(rotation=45)
            
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-info-circle"></i> Distribution Analysis</h5>
                <ul>
                    <li>Shows statistical distribution of values</li>
                    <li>Box represents interquartile range (IQR)</li>
                    <li>Whiskers show 1.5*IQR from the quartiles</li>
                    <li>Outliers shown as individual points</li>
                </ul>
            </div>
            """
            
        elif graph_type == 'heatmap':
            if state and state != 'All':
                pivot_df = filtered_df2.pivot_table(index='month_name', columns='year', 
                                                  values='Max.Demand Met during the day(MW)', aggfunc='mean')
                sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
                ax.set_title(f'Monthly Power Demand Heatmap for {state}')
            elif region and region != 'All':
                pivot_df = filtered_df1.pivot_table(index='month_name', columns='year', 
                                                  values=y_col, aggfunc='mean')
                sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
                ax.set_title(f'Monthly Power Generation Heatmap for {region}')
            else:
                pivot_df = filtered_df1.pivot_table(index='month_name', columns='year', 
                                                  values='All India', aggfunc='mean')
                sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
                ax.set_title('Monthly Power Generation Heatmap for All India')
            
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-info-circle"></i> Heatmap Analysis</h5>
                <ul>
                    <li>Visualizes patterns across months and years</li>
                    <li>Warmer colors indicate higher values</li>
                    <li>Helps identify seasonal patterns and trends</li>
                </ul>
            </div>
            """
            
        elif graph_type == 'prediction':
            if state and state != 'All':
                y = filtered_df2['Max.Demand Met during the day(MW)']
                X = filtered_df2[['Energy Met (MW)', 'Shortage during maximum Demand(MW)']]
                title = f"Energy Demand Prediction for {state}"
                data_description = f"state {state}"
            elif region and region != 'All':
                y = filtered_df1[y_col]
                X = filtered_df1[['day', 'month', 'year']]
                title = f"Energy Demand Prediction for {region} Region"
                data_description = f"{region} region"
            else:
                y = filtered_df1['All India']
                X = filtered_df1[['day', 'month', 'year']]
                title = "All India Energy Demand Prediction"
                data_description = "all India"
            
            train_size = int(len(y) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
                'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
                'Holt-Winters': Holt(y_train)
            }
            
            predictions = {}
            model_metrics = {}
            
            for name, model in models.items():
                try:
                    if name == 'Holt-Winters':
                        fit = model.fit(smoothing_level=0.8, smoothing_trend=0.02)
                        pred = fit.forecast(len(y_test))
                    else:
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                    
                    predictions[name] = pred
                    model_metrics[name] = calculate_metrics(y_test, pred)
                except Exception as e:
                    warnings.warn(f"Error fitting {name}: {str(e)}")
                    continue
            
            # Plot results
            y_train.plot(color="blue", label='Train', ax=ax)
            y_test.plot(color="pink", label='Test', ax=ax)
            
            colors = ['green', 'red', 'purple', 'orange']
            for i, (name, pred) in enumerate(predictions.items()):
                ax.plot(y_test.index, pred, color=colors[i], linestyle='--', 
                       label=f'{name} ({model_metrics[name]["rating"]})')
            
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylabel('Power Demand (MW)')
            ax.set_xlabel('Time Index')
            ax.set_ylim(bottom=0, top=max(y.max(), max(pred.max() for pred in predictions.values())) * 1.2)
            
            best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
            explanation = f"""
            <div class="alert alert-info">
                <h5><i class="bi bi-lightbulb"></i> Prediction Analysis for {data_description}</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6 class="card-title"><i class="bi bi-trophy-fill text-warning"></i> Best Model</h6>
                                <div class="d-flex align-items-center">
                                    <h4 class="mb-0">{best_model[0]}</h4>
                                    <span class="badge bg-{best_model[1]['rating_color']} ms-2">{best_model[1]['rating']}</span>
                                </div>
                                <p class="mb-0 small text-muted">Accuracy: {best_model[1]['accuracy']:.1f}%</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6 class="card-title"><i class="bi bi-info-circle"></i> Model Guide</h6>
                                <ul class="small mb-0">
                                    <li>90-100%: Excellent</li>
                                    <li>75-89%: Good</li>
                                    <li>50-74%: Fair</li>
                                    <li>Below 50%: Poor</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="mt-2">
                    <p class="mb-1"><strong>Analysis Notes:</strong></p>
                    <ul class="mb-0">
                        <li>Training period: {len(y_train)} days</li>
                        <li>Test period: {len(y_test)} days</li>
                        <li>Dashed lines show model predictions vs actual (pink)</li>
                    </ul>
                </div>
            </div>
            """
        
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig)
        
        return jsonify({
            'image': plot_url,
            'model_metrics': model_metrics,
            'explanation': explanation,
            'interactive': False
        })
    
    except Exception as e:
        error_message = f"""
        <div class="alert alert-danger">
            <h5><i class="bi bi-exclamation-triangle-fill"></i> Error Generating Visualization</h5>
            <p><strong>Error Details:</strong> {str(e)}</p>
            <p class="mb-0"><strong>Suggested Action:</strong> Try adjusting your filters or select a different visualization type.</p>
        </div>
        """
        return jsonify({'error': error_message}), 400

@app.route('/check_plotly')
def check_plotly():
    return jsonify({'plotly_available': PLOTLY_AVAILABLE})

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down server...")