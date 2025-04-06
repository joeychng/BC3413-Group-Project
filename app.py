from flask import Flask, render_template, url_for, request, redirect, flash, send_file, jsonify, session
import sqlite3
import json
import csv
import datetime
import nltk
from db import init_db, view_portfolio, visualise_portfolio, load_csv_data, insert_data_to_db, fetch_stock_data
from werkzeug.utils import secure_filename
import os
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

app = Flask(__name__)
app.secret_key = "your_secret_key"
# Define the folder for uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}



conn = init_db()
usa_large_companies, stock_info, nasdaq_data, sp_data = load_csv_data()
insert_data_to_db(conn, stock_info)

# Homepage -----------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        return redirect(url_for('homepage'))
    return render_template('homepage.html')

#Dashboard ---------------------------------------------------------------------------------
@app.route("/dashboard/<username>")
def dashboard(username):
    conn = init_db()
    portfolio_data = view_portfolio(username, conn)  # Fetch portfolio data for table

    if not portfolio_data:
        portfolio = []
        total_realized_pnl = 0
        total_unrealized_pnl = 0
        total_portfolio_value = 0
    else:
        portfolio = portfolio_data.get("portfolio", [])
        total_realized_pnl = portfolio_data.get("total_realized_pnl",0)
        total_unrealized_pnl = portfolio_data.get("total_unrealized_pnl", 0)
        total_portfolio_value = portfolio_data.get("total_portfolio_value", 0)

    chart_path = visualise_portfolio(username)  # Fetch pie chart data

    is_empty = len(portfolio) == 0 #Check if portfolio is empty

    return render_template(
        "dashboard.html",
        username=username,
        portfolio=portfolio,
        total_realized_pnl=total_realized_pnl,
        total_unrealized_pnl=total_unrealized_pnl,
        total_portfolio_value=total_portfolio_value,
        is_empty=is_empty,
        import_error=None,
        import_success=None
    )

@app.route("/portfolio_chart/<username>")
def portfolio_chart(username):
    chart_path = visualise_portfolio(username)
    if chart_path and os.path.exists(chart_path):
        return send_file(chart_path, mimetype='image/png')
    return "No chart available", 404


@app.route("/export_portfolio/<username>")
def export_portfolio(username):
    conn = init_db()
    portfolio = view_portfolio(username, conn)["portfolio"]  # Fetch portfolio data from DB
    filename = f"portfolio_{username}_{datetime.date.today()}.csv"
    filepath = os.path.join("exports", filename)

    if not os.path.exists("exports"):
        os.makedirs("exports")

    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Ticker", "Shares", "Sector", "Purchase Price", "Live Price", "Unrealized P/L", "Realized P/L"])

        # Iterate through the list of dictionaries
        for stock in portfolio:
            writer.writerow([
                stock["ticker"],
                stock["shares"],
                stock["sector"],
                stock["purchase_price"],
                stock["current_price"],
                stock["unrealized_pnl"],
                stock["realized_pnl"]
            ])

    return send_file(filepath, as_attachment=True)


@app.route("/dashboard/<username>/import", methods=["POST"])
def import_portfolio(username):
    if 'csv_file' not in request.files:
        flash("No file part", "error")
        return render_template('dashboard.html', username=username, import_error="No file selected.")

    file = request.files['csv_file']
    if file.filename == '':
        flash("No selected file", "error")
        return render_template('dashboard.html', username=username, import_error="No selected file.")

    if file:
        filename = secure_filename(f"portfolio_{username}.csv")
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process CSV File
        try:
            with open(file_path, mode="r") as f:
                reader = csv.reader(f)
                headers = next(reader, None)  # Read the header

                expected_headers = ["Ticker", "Shares", "Sector", "Purchase Price", "Live Price", "Unrealized P/L",
                                    "Realized P/L"]
                if headers != expected_headers:
                    flash("Invalid CSV format! Please use the correct template.", "error")
                    return redirect(url_for("dashboard", username=username))

                conn = sqlite3.connect("portfolio.db")
                cursor = conn.cursor()

                for row in reader:
                    try:
                        ticker, shares, sector, purchase_price, live_price, unrealised_pnl, realized_pnl = row
                        shares = int(shares)
                        purchase_price = float(purchase_price)
                        live_price = float(live_price)
                        unrealised_pnl = float(unrealised_pnl)
                        realized_pnl = float(realized_pnl) if realized_pnl != "Not Available" else None

                        cursor.execute('''
                                INSERT INTO portfolios (username, ticker, shares, purchase_price, purchase_date, sale_price, sale_date, realized_profit_loss)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (username, ticker, shares, purchase_price, "Unknown", None, None, realized_pnl))

                    except ValueError:
                        flash(f"Skipping invalid row: {row}", "error")

                conn.commit()
                conn.close()

            flash("Portfolio imported successfully!", "success")
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")

    return redirect(url_for("dashboard", username=username))

# Log out -----------------------------------------------------------------
@app.route("/logout", methods=["POST"])
def logout():
    # Clear the session
    session.pop('username', None)

    # Redirect to the homepage after logout
    return redirect(url_for('homepage'))

cache = {}
cache_expired = timedelta(minutes=30)

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
        'Market Cap': info.get('marketCap', 'N/A'),
        'Previous Close': info.get('previousClose', 'N/A'),
        'Trailing PE': info.get('trailingPE', 'N/A'),
        'Forward PE': info.get('forwardPE', 'N/A'),
        'Returns': returns.to_dict()  # Optional, if you want to send it
    }

def generate_chart_base64(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="6mo")

    if history.empty:
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(history.index, history['Close'], label=f"{ticker} Price", color="blue")
    plt.title(f"{ticker} Stock Price (Last 6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Closing Price ($)")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_img

def recommend_stocks(ticker, risk_tolerance):
    stock_data = fetch_stock_data(ticker)
    risk_info = calculate_risk_info(stock_data)

    if stock_data['Trailing PE'] is None or risk_info is None:
        return {"error": f"No P/E ratio or risk data available for {ticker}."}

    pe_ratio = stock_data['Trailing PE']
    industry_avg_pe = 20  # Assume industry avg P/E

    if risk_tolerance == "Low":
        sharpe_threshold = 1.5
    elif risk_tolerance == "Medium":
        sharpe_threshold = 1.0
    else:
        sharpe_threshold = 0.5

    if pe_ratio < industry_avg_pe * 0.8 and risk_info["sharpe_ratio"] > sharpe_threshold:
        recommendation = f"✅ {ticker} is undervalued and within your risk tolerance. Consider adding."
    elif pe_ratio > industry_avg_pe * 1.2 or risk_info["sharpe_ratio"] < sharpe_threshold:
        recommendation = f"⚠️ {ticker} may be overvalued or outside your risk tolerance. Consider avoiding."
    else:
        recommendation = f"⏸️ {ticker} is fairly valued and within risk tolerance. Consider holding."

    return {
        "stock_data": stock_data,
        "risk_info": risk_info,
        "recommendation": recommendation
    }

def show_overall_results(ticker, sentiment_scores):
    if not sentiment_scores:
        return {"label": "Neutral", "score": 0}

    scores = [score for _, score, _ in sentiment_scores]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.05:
        label = "Positive"
    elif avg_score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "score": avg_score}

def format_headline_data(sentiment_scores):
    data = []
    for headline, score, url in sentiment_scores:
        data.append({
            "headline": headline,
            "score": round(score, 2),
            "url": url
        })
    return data
@app.route('/')
def index():
    return render_template('Stock_information.html')  # This serves the HTML page

@app.route('/get_stock_info')
def get_stock_info():
    ticker_symbol = request.args.get('ticker', '').upper()
    if not ticker_symbol:
        return jsonify({'error': 'No ticker provided'}), 400

    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        stock_data = {
            'Ticker': ticker_symbol,
            'Company': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Previous Close': info.get('previousClose', 'N/A'),
            'Trailing PE': info.get('trailingPE', 'N/A'),
            'Forward PE': info.get('forwardPE', 'N/A'),
        }

        # Generate chart
        hist = stock.history(period="6mo")
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'], label='Close Price')
        plt.title(f'{ticker_symbol} Price - Last 6 Months')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({'stock_data': stock_data, 'chart': chart_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_stock_recommendation_and_metrics')
def get_stock_recommendation_and_metrics():
    ticker = request.args.get('ticker', '').upper()

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    try:
        session['risk_tolerance']=risk_tolerance  
        result = recommend_stocks(ticker, risk_tolerance)

        if "error" in result:
            return jsonify({"error": result["error"]}), 404
    
        # Render the template and pass the data to it
        return render_template(
            "Recommendations.html",
            stock_data=result["stock_data"],
            risk_info=result["risk_info"],
            recommendation=result["recommendation"]
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_market_sentiment")
def market_sentiment():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400

    try:
        result = fetch_market_sentiment(ticker)
        if not result:
            return jsonify({"error": "Failed to retrieve data"}), 500

        overall_data, headline_data = result

        # Render the template and pass the data to it
        return render_template(
            "Market_Sentiment.html",
            ticker=ticker,
            overall_sentiment=overall_data,
            headlines=headline_data
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
