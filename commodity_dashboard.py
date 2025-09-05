import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define commodity tickers
COMMODITY_TICKERS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Crude Oil': 'CL=F',
    'Natural Gas': 'NG=F',
    'Copper': 'HG=F',
    'Platinum': 'PL=F',
    'Wheat': 'ZW=F',
    'Corn': 'ZC=F',
    'Soybeans': 'ZS=F'
}

# Helper functions
def fetch_commodity_data(ticker, start_date, end_date):
    """Fetch commodity price data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            data = data[['Close', 'Volume']]
            data.columns = ['Price', 'Volume']
            data = data.reset_index()
            return data
    except:
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    # Moving averages
    df['MA_7'] = df['Price'].rolling(window=7).mean()
    df['MA_30'] = df['Price'].rolling(window=30).mean()
    df['MA_50'] = df['Price'].rolling(window=50).mean()
    
    # RSI (14-day)
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Price'].rolling(window=20).mean()
    bb_std = df['Price'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df['Price'].ewm(span=12, adjust=False).mean()
    exp2 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Daily returns
    df['Daily_Return'] = df['Price'].pct_change() * 100
    
    # Add day of month and month
    df['Day_of_Month'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.month_name()
    
    return df

def calculate_statistics(df):
    """Calculate investment statistics"""
    if df.empty:
        return {}
    
    # Average daily price change by day of month
    day_stats = df.groupby('Day_of_Month')['Daily_Return'].agg(['mean', 'std', 'count'])
    best_days = day_stats.nlargest(5, 'mean').index.tolist()
    worst_days = day_stats.nsmallest(5, 'mean').index.tolist()
    
    # Monthly statistics
    monthly_stats = df.groupby('Month_Name')['Daily_Return'].agg(['mean', 'std'])
    best_month = monthly_stats.idxmax()['mean']
    
    # Volatility
    volatility = df['Daily_Return'].std()
    
    # Trend strength
    if len(df) > 50:
        current_price = df['Price'].iloc[-1]
        ma50 = df['MA_50'].iloc[-1]
        trend = "Bullish" if current_price > ma50 else "Bearish"
    else:
        trend = "Insufficient data"
    
    # RSI signal
    current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].isna().all() else 50
    if current_rsi < 30:
        rsi_signal = "Oversold (Buy Signal)"
    elif current_rsi > 70:
        rsi_signal = "Overbought (Sell Signal)"
    else:
        rsi_signal = "Neutral"
    
    return {
        'best_days': best_days,
        'worst_days': worst_days,
        'best_month': best_month,
        'volatility': volatility,
        'trend': trend,
        'rsi_signal': rsi_signal,
        'current_rsi': current_rsi
    }

def generate_investment_recommendation(stats, df):
    """Generate investment recommendation based on analysis"""
    if not stats or df.empty:
        return "Insufficient data for recommendation"
    
    recommendations = []
    score = 0
    
    # RSI Analysis
    if stats['current_rsi'] < 30:
        recommendations.append("✅ RSI indicates oversold condition - potential buying opportunity")
        score += 2
    elif stats['current_rsi'] > 70:
        recommendations.append("⚠️ RSI indicates overbought condition - consider waiting or selling")
        score -= 2
    else:
        recommendations.append("➖ RSI is neutral")
    
    # Trend Analysis
    if stats['trend'] == "Bullish":
        recommendations.append("✅ Price is above 50-day MA - bullish trend")
        score += 1
    elif stats['trend'] == "Bearish":
        recommendations.append("⚠️ Price is below 50-day MA - bearish trend")
        score -= 1
    
    # Volatility Analysis
    if stats['volatility'] < 2:
        recommendations.append("✅ Low volatility - stable investment")
        score += 1
    elif stats['volatility'] > 5:
        recommendations.append("⚠️ High volatility - increased risk")
        score -= 1
    else:
        recommendations.append("➖ Moderate volatility")
    
    # Best days recommendation
    if stats['best_days']:
        recommendations.append(f"📅 Best days to invest: {', '.join(map(str, stats['best_days'][:3]))}")
    
    # Overall recommendation
    if score >= 3:
        overall = "🟢 STRONG BUY - Multiple positive indicators"
    elif score >= 1:
        overall = "🟡 BUY - Favorable conditions"
    elif score >= -1:
        overall = "⚪ HOLD - Mixed signals"
    else:
        overall = "🔴 SELL/WAIT - Unfavorable conditions"
    
    return "\n".join([overall] + recommendations)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Commodity Investment Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Select Parameters"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Commodity"),
                            dcc.Dropdown(
                                id='commodity-dropdown',
                                options=[{'label': k, 'value': v} for k, v in COMMODITY_TICKERS.items()],
                                value='GC=F',
                                clearable=False
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Date Range"),
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date=(datetime.now() - timedelta(days=365)).date(),
                                end_date=datetime.now().date(),
                                display_format='YYYY-MM-DD'
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Button("Refresh Data", id="refresh-btn", color="primary", className="mt-4", n_clicks=0)
                        ], md=2)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Current Price", className="text-muted"),
                    html.H3(id="current-price", children="-"),
                    html.P(id="price-change", children="-")
                ])
            ])
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("7-Day MA", className="text-muted"),
                    html.H3(id="ma7", children="-")
                ])
            ])
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("RSI (14)", className="text-muted"),
                    html.H3(id="rsi-value", children="-"),
                    html.P(id="rsi-status", children="-")
                ])
            ])
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Volatility", className="text-muted"),
                    html.H3(id="volatility", children="-")
                ])
            ])
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Trend", className="text-muted"),
                    html.H3(id="trend", children="-")
                ])
            ])
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Best Investment Days", className="text-muted"),
                    html.H4(id="best-days", children="-")
                ])
            ])
        ], md=2),
    ], className="mb-4"),
    
    # Investment Recommendation Card
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Investment Recommendation", className="mb-0"),
                    dbc.Button("?", id="help-btn", color="info", size="sm", className="float-end")
                ]),
                dbc.CardBody([
                    html.Pre(id="recommendation", style={'whiteSpace': 'pre-wrap', 'fontSize': '14px'})
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Charts
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="price-chart", style={'height': '400px'})
        ], md=6),
        dbc.Col([
            dcc.Graph(id="indicator-chart", style={'height': '400px'})
        ], md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="day-analysis", style={'height': '350px'})
        ], md=6),
        dbc.Col([
            dcc.Graph(id="monthly-analysis", style={'height': '350px'})
        ], md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="volume-chart", style={'height': '300px'})
        ], md=12)
    ]),
    
    # Help Modal
    dbc.Modal([
        dbc.ModalHeader("Understanding the Indicators"),
        dbc.ModalBody([
            html.H6("RSI (Relative Strength Index)"),
            html.P("RSI measures momentum. Below 30 = oversold (buy signal), Above 70 = overbought (sell signal)"),
            html.Hr(),
            html.H6("Moving Averages"),
            html.P("MA shows average price over time. Price above MA = bullish, below = bearish"),
            html.Hr(),
            html.H6("Bollinger Bands"),
            html.P("Shows volatility. Price near lower band = potential buy, near upper = potential sell"),
            html.Hr(),
            html.H6("MACD"),
            html.P("Shows trend changes. MACD above signal line = bullish, below = bearish"),
            html.Hr(),
            html.H6("Best Investment Days"),
            html.P("Days of the month with historically positive average returns"),
            html.Hr(),
            html.H6("Volatility"),
            html.P("Standard deviation of daily returns. Higher = more risk")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        )
    ], id="help-modal", is_open=False, size="lg"),
    
    # Store component for data
    dcc.Store(id='commodity-data')
    
], fluid=True, className="p-4")

# Callbacks
@app.callback(
    Output('commodity-data', 'data'),
    [Input('refresh-btn', 'n_clicks')],
    [State('commodity-dropdown', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_data(n_clicks, ticker, start_date, end_date):
    df = fetch_commodity_data(ticker, start_date, end_date)
    if not df.empty:
        df = calculate_indicators(df)
        return df.to_dict('records')
    return []

@app.callback(
    [Output('current-price', 'children'),
     Output('price-change', 'children'),
     Output('price-change', 'style'),
     Output('ma7', 'children'),
     Output('rsi-value', 'children'),
     Output('rsi-status', 'children'),
     Output('volatility', 'children'),
     Output('trend', 'children'),
     Output('best-days', 'children'),
     Output('recommendation', 'children')],
    [Input('commodity-data', 'data')]
)
def update_kpis(data):
    if not data:
        return ["-"] * 9 + ["No data available"]
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    stats = calculate_statistics(df)
    
    # Current price and change
    current_price = f"${df['Price'].iloc[-1]:.2f}"
    price_change_val = df['Daily_Return'].iloc[-1] if not pd.isna(df['Daily_Return'].iloc[-1]) else 0
    price_change = f"{price_change_val:+.2f}%"
    price_style = {'color': 'green' if price_change_val >= 0 else 'red'}
    
    # MA7
    ma7 = f"${df['MA_7'].iloc[-1]:.2f}" if not pd.isna(df['MA_7'].iloc[-1]) else "-"
    
    # RSI
    rsi_val = f"{stats['current_rsi']:.1f}"
    rsi_status = stats['rsi_signal']
    
    # Volatility
    volatility = f"{stats['volatility']:.2f}%"
    
    # Trend
    trend = stats['trend']
    
    # Best days
    best_days = ", ".join(map(str, stats['best_days'][:3])) if stats['best_days'] else "-"
    
    # Recommendation
    recommendation = generate_investment_recommendation(stats, df)
    
    return (current_price, price_change, price_style, ma7, rsi_val, rsi_status, 
            volatility, trend, best_days, recommendation)

@app.callback(
    Output('price-chart', 'figure'),
    [Input('commodity-data', 'data')]
)
def update_price_chart(data):
    if not data:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Candlestick/Price line
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Price'], name='Price', line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MA_7'], name='MA 7', line=dict(color='orange', dash='dash')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MA_30'], name='MA 30', line=dict(color='green', dash='dash')),
        secondary_y=False
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', 
                   line=dict(color='gray', width=1), opacity=0.3),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', 
                   line=dict(color='gray', width=1), opacity=0.3, fill='tonexty'),
        secondary_y=False
    )
    
    fig.update_layout(
        title="Price Chart with Moving Averages & Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

@app.callback(
    Output('indicator-chart', 'figure'),
    [Input('commodity-data', 'data')]
)
def update_indicator_chart(data):
    if not data:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=('RSI', 'MACD'),
                        row_heights=[0.5, 0.5])
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Signal'], name='Signal', line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram', marker_color='gray'),
        row=2, col=1
    )
    
    fig.update_layout(height=400, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="RSI", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    
    return fig

@app.callback(
    Output('day-analysis', 'figure'),
    [Input('commodity-data', 'data')]
)
def update_day_analysis(data):
    if not data:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = pd.DataFrame(data)
    day_stats = df.groupby('Day_of_Month')['Daily_Return'].agg(['mean', 'std']).reset_index()
    day_stats = day_stats.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=day_stats['Day_of_Month'],
        y=day_stats['mean'],
        name='Avg Return',
        marker_color=np.where(day_stats['mean'] > 0, 'green', 'red'),
        error_y=dict(type='data', array=day_stats['std'], visible=True)
    ))
    
    fig.update_layout(
        title="Average Daily Return by Day of Month",
        xaxis_title="Day of Month",
        yaxis_title="Average Return (%)",
        showlegend=False,
        height=350
    )
    
    return fig

@app.callback(
    Output('monthly-analysis', 'figure'),
    [Input('commodity-data', 'data')]
)
def update_monthly_analysis(data):
    if not data:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    monthly_stats = df.groupby('Month_Name')['Daily_Return'].agg(['mean', 'std']).reset_index()
    
    # Sort by calendar month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats['Month_Name'] = pd.Categorical(monthly_stats['Month_Name'], categories=month_order, ordered=True)
    monthly_stats = monthly_stats.sort_values('Month_Name')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_stats['Month_Name'],
        y=monthly_stats['mean'],
        name='Avg Return',
        marker_color=np.where(monthly_stats['mean'] > 0, 'lightgreen', 'lightcoral'),
        error_y=dict(type='data', array=monthly_stats['std'], visible=True)
    ))
    
    fig.update_layout(
        title="Average Monthly Return",
        xaxis_title="Month",
        yaxis_title="Average Return (%)",
        showlegend=False,
        height=350
    )
    
    return fig

@app.callback(
    Output('volume-chart', 'figure'),
    [Input('commodity-data', 'data')]
)
def update_volume_chart(data):
    if not data:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=False,
        height=300
    )
    
    return fig

@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-btn", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("help-modal", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run(debug=True, port=8050)