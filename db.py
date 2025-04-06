import sqlite3
import csv
import os
import pandas as pd
import yfinance as yf
import prettytable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import defaultdict

#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


DB_NAME = "portfolio.db"
CHARTS_DIR = "static/charts"

if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)  # Create directory if not exists

##---------------------------------
#Database Setup
##---------------------------------
def init_db():
    """Initialize database and create tables if not exist."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            security_question TEXT,
            security_answer TEXT,
            risk_tolerance TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            exchange TEXT,
            sector TEXT,
            industry TEXT,
            market_cap REAL,
            sales REAL,
            profits REAL,
            assets REAL,
            market_value REAL
        )
    ''')

        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS companies_fts USING fts5 (
            name,
            tokenize = 'trigram'
        )
    ''')
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS companies_fts_trigger
            AFTER INSERT ON companies
        BEGIN
            INSERT INTO companies_fts (rowid, name) VALUES (new.rowid, new.name);
        END
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            ticker TEXT,
            shares INTEGER,
            share_check INTEGER,
            live_price REAL,
            purchase_price REAL,
            purchase_date TEXT,
            sale_price REAL DEFAULT NULL,
            sale_date TEXT DEFAULT NULL,
            realized_profit_loss REAL DEFAULT NULL,
            unrealized_profit_loss REAL DEFAULT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    return conn

# ------------------------------
# Data Loading Functions
# ------------------------------
def load_csv_data():
    usa_large_companies = pd.read_csv("data/USA large companies.csv", delimiter="\t")
    stock_info = pd.read_csv("data/stock_info_tickers_exchange.csv")
    nasdaq_data = pd.read_csv("data/nasdaq_tickers_sector.csv")
    sp_data = pd.read_csv("data/SnP_tickers_sector.csv")
    return usa_large_companies, stock_info, nasdaq_data, sp_data

# ------------------------------
# Data Insertion into SQLite
# ------------------------------
def insert_data_to_db(conn, stock_info):
    cursor = conn.cursor()
    for _, row in stock_info.iterrows():
        cursor.execute('''
            INSERT OR REPLACE INTO companies (ticker, name, exchange)
            VALUES (?, ?, ?)
        ''', (row['Ticker'], row['Name'], row['Exchange']))
    conn.commit()

# ------------------------------
# Yahoo Finance Integration
# ------------------------------
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    returns = hist['Close'].pct_change().dropna()
    return {
        'Ticker': ticker,
        'Company': info.get('shortName', 'N/A'),
        'Sector': info.get('sector', 'N/A'),
        'Industry': info.get('industry', 'N/A'),
        'EBITDA': info.get('ebitda', 'N/A'),
        'Book Value': info.get('bookValue', 'N/A'),
        'Market Cap': info.get('marketCap', 'N/A'),
        'Previous Close': info.get('previousClose', 'N/A'),
        'Trailing PE': info.get('trailingPE', 'N/A'),
        'Forward PE': info.get('forwardPE', 'N/A'),
        'Beta': info.get('beta', 'N/A'),
        'Returns': returns
    }
# ------------------------------
# User Guide
# ------------------------------
def search_company_by_name(conn, name_query):
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rowid FROM companies_fts WHERE companies_fts MATCH ? ORDER BY rank LIMIT 5",
            (name_query,)
        )
        fts_res = cursor.fetchall()
        if not fts_res:
            return []

        param_ls = ','.join(['?'] * len(fts_res))
        cursor.execute(
            f"SELECT ticker, name FROM companies WHERE rowid IN ({param_ls})",
            [x[0] for x in fts_res]
        )
        return cursor.fetchall()
    except Exception as e:
        print("Search error:", e)
        return []


##---------------------------------
#View Portfolio (Table)
##---------------------------------
def view_portfolio(username, conn):
    cursor = conn.cursor()

    # Query to get only stocks with remaining shares that have not been fully sold
    cursor.execute('''
               SELECT ticker, SUM(shares), purchase_price, purchase_date FROM portfolios
               WHERE username = ? AND shares > 0 AND sale_date IS NULL
               GROUP BY ticker, purchase_price, purchase_date
           ''', (username,))

    holdings = cursor.fetchall()

    # Query total realized profit/loss
    cursor.execute('''
                   SELECT SUM(realized_profit_loss) FROM portfolios
                   WHERE username = ? AND realized_profit_loss IS NOT NULL
               ''', (username,))
    total_realized_pnl = cursor.fetchone()[0] or 0  # Default to 0 if no realized P&L

    if not holdings:
        print("\nYour portfolio is empty.")
        return {
            "portfolio": [],
            "total_realized_pnl": 0,
            "total_unrealized_pnl": 0,
            "ticker_values": {},
            "total_portfolio_value": 0
        }

    portfolio_data = []  # For CSV export
    total_unrealized_pnl = 0  # Track total unrealized P/L
    ticker_values = defaultdict(float)
    total_portfolio_value = 0

    for ticker, shares, purchase_price, purchase_date in holdings:
        stock_data = fetch_stock_data(ticker)
        current_price = stock_data.get('Previous Close', 0)
        sector = stock_data.get('Sector', 'N/A')

        # Fetch realized P/L for this stock
        cursor.execute('''
                           SELECT SUM(realized_profit_loss) FROM portfolios
                           WHERE username = ? AND ticker = ?
                       ''', (username, ticker))
        realized_pnl = cursor.fetchone()[0] or 0  # Default to 0 if no realized P&L

        # Calculate Unrealized P/L
        if current_price > 0:
            unrealized_pnl = (current_price - purchase_price) * shares
            total_unrealized_pnl += unrealized_pnl
            ticker_values[ticker] += current_price * shares
        else:
            unrealized_pnl = "N/A"

        portfolio_data.append({
            "ticker": ticker,
            "shares": shares,
            "sector": sector,
            "purchase_price": f"{float(purchase_price):,.2f}",
            "current_price": current_price if current_price > 0 else "N/A",
            "unrealized_pnl": f"{float(unrealized_pnl):,.2f}",
            "realized_pnl": f"{float(realized_pnl):,.2f}"
            })

        total_portfolio_value = sum(ticker_values.values()) if ticker_values else 0

    return {
        "portfolio": portfolio_data,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "ticker_values": ticker_values,
        "total_portfolio_value": total_portfolio_value
    }

##---------------------------------
#Visualise Portfolio (Pie Chart)
##---------------------------------
def visualise_portfolio(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
                SELECT ticker, SUM(shares) FROM portfolios
                WHERE username = ? AND (sale_date IS NULL OR shares > 0)
                GROUP BY ticker
            ''', (username,))

    holdings = cursor.fetchall()

    if not holdings:
        return None

    ticker_values = defaultdict(float)

    for ticker, shares in holdings:
        # Ensure that the stock is still available (not fully sold)
        cursor.execute('''
                    SELECT SUM(shares) FROM portfolios
                    WHERE username = ? AND ticker = ? AND sale_date IS NULL
                ''', (username, ticker))
        remaining_shares = cursor.fetchone()[0] or 0

        if remaining_shares > 0:  # Only include if there are still shares available
            stock_data = fetch_stock_data(ticker)
            stock_price = stock_data.get('Previous Close', 0)

            if stock_price>0:
                total_value = stock_price * remaining_shares
                ticker_values[ticker] += total_value

    total_portfolio_value = sum(ticker_values.values())

    if total_portfolio_value == 0:
        return None

    # Plot pie chart
    plt.figure(figsize=(8, 5))
    plt.pie(
        ticker_values.values(),
        labels=ticker_values.keys(),
        autopct=lambda p: f'{p:.1f}% (${p * total_portfolio_value / 100:.2f})',
        startangle=140
    )
    plt.title(f"Portfolio Allocation (by Value) for {username}")

    # Save the chart
    chart_path = os.path.join(CHARTS_DIR, f"{username}_portfolio.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

    return chart_path  # Return file path

##---------------------------------
#Import Portfolio
##---------------------------------
def import_portfolio_from_csv(username, file_path):
    """Process the CSV file and insert portfolio data into the database."""
    expected_headers = ["Ticker", "Shares", "Sector", "Purchase Price", "Live Price", "Unrealized P/L", "Realized P/L"]

    if not os.path.exists(file_path):
        return "File not found. Please upload a valid file."

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)

        headers = next(reader, None)
        if headers != expected_headers:
            return f"Invalid CSV format! Expected headers: {expected_headers}, but found: {headers}"

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        for row in reader:
            try:
                ticker, shares, sector, purchase_price, live_price, unrealized_pl, realized_pl = row
                shares = int(shares)
                purchase_price = float(purchase_price)
                live_price = float(live_price)
                unrealized_pl = float(unrealized_pl)
                realized_pl = float(realized_pl) if realized_pl != "Not Available" else None

                # Insert data into the portfolio table
                cursor.execute('''
                    INSERT INTO portfolios (username, ticker, shares, purchase_price, purchase_date, sale_price, sale_date, realized_profit_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (username, ticker, shares, purchase_price, "Unknown", None, None, realized_pl))

                conn.commit()
            except ValueError:
                continue  # Skip invalid rows

        conn.close()

    os.remove(file_path)  # Remove the file after processing
    return None  # Return None if no errors occurred





