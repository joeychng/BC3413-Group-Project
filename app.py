from flask import Flask, render_template, url_for, request, redirect, flash, send_file, jsonify, session
import sqlite3
import json
import csv
import datetime
import nltk
nltk.download("vader_lexicon")
from db import init_db, view_portfolio, visualise_portfolio, load_csv_data, insert_data_to_db, fetch_stock_data
from werkzeug.utils import secure_filename
import os
import yfinance as yf
import matplotlib.pyplot as plt
import io
import re
import hashlib
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

# Register & Login -----------------------------------------------------------------------------
class User:
    def __init__(self, db_name='portfolio.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.current_user_id = None # Stored Logged-in user ID

        self.conn.commit()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def is_strong_password(self, password):
        if (len(password) >= 8 and
                re.search(r"[A-Z]", password) and
                re.search(r"[a-z]", password) and
                re.search(r"[0-9]", password) and
                re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
            return True
        return False

    def register_user(self, username, password, security_question, security_answer):
        hashed_password = self.hash_password(password)
        hashed_answer = self.hash_password(security_answer)

        self.cursor.execute(
            "INSERT INTO users (username, password, security_question, security_answer) VALUES (?, ?, ?, ?)",
            (username, hashed_password, security_question, hashed_answer))
        self.conn.commit()

    def login_user(self, username, password):
        hashed_password = self.hash_password(password)
        self.cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
        user = self.cursor.fetchone()

        if user:
            self.current_user_id = username  # Store logged-in username
            return True
        return False

    def get_user_by_username(self, username):
        self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return self.cursor.fetchone()

    def reset_password(self, username, new_password):
        hashed_new_password = self.hash_password(new_password)
        self.cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_new_password, username))
        self.conn.commit()

    def fetch_risk_tolerance(self, username):
        self.cursor.execute("SELECT risk_tolerance FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchone()
        if not result or result[0] is None:
            default_tolerance = 'Medium'
            self.cursor.execute('''
                    UPDATE users SET risk_tolerance = ? WHERE username = ?
                ''', (default_tolerance, username))
            self.conn.commit()
            return default_tolerance

        return result[0]  # Return the existing risk tolerance

    def update_risk_tolerance(self, username, new_risk_tolerance):
        self.cursor.execute("UPDATE users SET risk_tolerance = ? WHERE username = ?", (new_risk_tolerance, username))
        self.conn.commit()


#==============
# Register
#==============
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        security_question = request.form['security_question']
        security_answer = request.form['security_answer']
        username_error= None
        password_error = None
        user = User()
        if user.get_user_by_username(username):
            username_error="Username already exists. Please try registering with another username."
            if not user.is_strong_password(password):
                password_error = "Weak password. Please follow the guidelines."
        else:
            if user.is_strong_password(password):
                user.register_user(username, password, security_question, security_answer)
                return render_template('login.html')  # Redirect to login after registration
            else:
                password_error = "Weak password. Please follow the guidelines."
        return render_template('register.html', username_error=username_error, password_error=password_error)
    return render_template('register.html')

#==============
# Login
#==============
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User()
        session['login_username'] = username
        if request.form.get('forget_password') == 'true':
            return redirect(url_for('forget_password'))
        elif user.login_user(username, password):
            return redirect(url_for('dashboard', username=username))  # Redirect to page after successful login
        else:
            return render_template('login.html', error="Invalid credentials, please try again.", )
    return render_template('login.html')

@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    username_error = None
    if request.method == 'POST':
        username = request.form['username']
        user = User()
        if user.get_user_by_username(username):
            session['forget_username'] = username  # ✅ Store username in session
            return redirect(url_for('test'))
        else:
            username_error = "Username does not exist."
    return render_template('forget_password.html', username_error=username_error)

@app.route('/test', methods=['GET', 'POST'])
def test():
    security_ans_error = None
    security_question = None
    user = User()
    username = session.get('forget_username')

    if not username:
        return redirect(url_for('forget_password'))  # If no username, go back

        # Get the security question for the username
    user.cursor.execute("SELECT security_question FROM users WHERE username = ?", (username,))
    security = user.cursor.fetchone()
    if security:
        security_question = security[0]
        if request.method == 'POST':
            security_answer = request.form['security_answer']
            hashed_answer = user.hash_password(security_answer)

            user.cursor.execute("SELECT * FROM users WHERE username = ? AND security_answer = ?",
                                (username, hashed_answer))
            if user.cursor.fetchone():
                return render_template('reset_password.html', security_question=security_question)

            else:
                security_ans_error = "Incorrect answer to the security question. Please try again."

    return render_template('test.html', security_question=security_question, security_ans_error=security_ans_error)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    password_error = None
    user = User()
    username = session.get('forget_username')
    if not username:
        return redirect(url_for('forget_password'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        print(new_password)
        if user.is_strong_password(new_password):
            user.reset_password(username, new_password)
            session.pop('forget_username', None)
            print("yellow")
            return redirect(url_for('login'))
        else:
            password_error = "Weak password. Please follow the guidelines."
    return render_template('reset_password.html', password_error=password_error)

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

# User Guide -----------------------------------------------------------------
@app.route("/dashboard/<username>/userguide", methods=["GET", "POST"])
def userguide(username):
    search_results = []
    query = ""
    user = User()
    username = session.get('login_username')

    if request.method == 'POST':
        query = request.form.get('company_name', '').strip()
        if query:
            conn = sqlite3.connect(DB_NAME)
            search_results = search_company_by_name(conn, query)
            conn.close()

    return render_template("userguide.html", results=search_results, query=query, username=username)

# Risk Tolerance -----------------------------------------------------------------
@app.route('/dashboard/<username>/risk_tolerance', methods=['GET', 'POST'])
def risk_tolerance(username):
    user = User()
    username = session.get('login_username')
    new_risk_tolerance = None

    if request.method == 'POST':
        new_risk_tolerance = request.form['risk_tolerance']
        user.update_risk_tolerance(username, new_risk_tolerance)
        print(new_risk_tolerance)
    else:
        new_risk_tolerance = user.fetch_risk_tolerance(username)
    return render_template('risk_tolerance.html', risk_tolerance=new_risk_tolerance, username=username)
    

# Function to add stock 
def add_stock_to_db(ticker, shares, purchase_date, purchase_price):
    conn = init_db()
    live_price = fetch_stock_data(ticker)  # Assuming this function fetches current stock price
    unrealized_pnl = (live_price - purchase_price) * shares

    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO portfolios (ticker, shares, purchase_date, purchase_price, unrealized_profit_loss)
        VALUES (?, ?, ?, ?, ?)
    ''', (ticker, shares, purchase_date, purchase_price, unrealized_pnl))

    conn.commit()
    conn.close()

@app.route('/addstock', methods=['GET', 'POST'])
def addstock():
    if request.method == 'POST':
        ticker = request.form['ticker']
        shares = int(request.form['shares'])
        purchase_date = request.form['purchase_date']
        purchase_price = float(request.form['purchase_price'])

        add_stock_to_db(ticker, shares, purchase_date, purchase_price)

        return redirect(url_for('homepage'))  # Redirect to the homepage after form submission

    return render_template('addstock.html')

@app.route('/removestock', methods=['GET', 'POST'])
def removestock():
    if request.method == 'POST':
        ticker = request.form['ticker']
        shares = int(request.form['shares'])
        purchase_date = request.form['sale_date']
        purchase_price = float(request.form['sale_price'])

        add_stock_to_db(ticker, shares, sale_date, sale_price)

    return render_template('removestock.html')

#Transaction history --------------------------------------------------------------------------------------------------
@app.route('/dashboard/<username>/transaction_history')
def transaction_history(username):
    user = User()
    username = session.get('login_username')
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT purchase_date, ticker, shares, purchase_price,
               sale_date, sale_price, realized_profit_loss
        FROM portfolios
        WHERE username = ?
        ORDER BY purchase_date DESC
    ''', (session['login_username'],))

    rows = cursor.fetchall()
    conn.close()

    # Format as list of dictionaries
    data = []
    for row in rows:
        (purchase_date, ticker, shares, buy_price, sell_date, sell_price, realized_pl) = row
        trans_type = 'Buy' if shares > 0 else 'Sell'
        unrealised_pl = 0.0 if sell_price else (fetch_stock_data(ticker).get('Bid', 0) - buy_price) * shares
        data.append({
            'purchase_date': purchase_date,
            'ticker': ticker,
            'company': get_company_name(ticker),  # Use the function to get company name
            'type': trans_type,
            'shares': abs(shares),
            'buy_price': buy_price if shares > 0 else '',
            'sell_date': sell_date or '',
            'sell_price': sell_price or '',
            'realised_pl': realized_pl or '',
            'unrealised_pl': round(unrealised_pl, 2) if not sell_price else ''
        })

    return render_template("transaction_history.html", transactions=data, username = username)

#Stock Information--------------------------------------------------------------------------------------------------
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
        'Returns': returns.to_dict()  
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

def calculate_risk_info(stock_data):
    returns_dict = stock_data.get('Returns')

    if not returns_dict or len(returns_dict) == 0:
        return {
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "var_95": 0.0
        }

    # Convert dict values to a list or NumPy array
    return_values = np.array(list(returns_dict.values()))

    # Volatility
    volatility = np.std(return_values) * math.sqrt(252)

    # Sharpe Ratio
    avg_return = np.mean(return_values) * 252
    risk_free_rate = 0.02
    sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility != 0 else 0.0

    # Value at Risk (VaR) at 95% confidence
    confidence_level = 0.05
    var_95 = norm.ppf(confidence_level, np.mean(return_values), np.std(return_values)) * 100

    return {
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "var_95": var_95
    }

def format_headline_data(sentiment_scores):
    data = []
    for headline, score, url in sentiment_scores:
        data.append({
            "headline": headline,
            "score": round(score, 2),
            "url": url
        })
    return data

@app.route("/info_menu")
def info_menu():
    return render_template("stock_information.html")

@app.route("/get_stock_info")
def get_stock_info():
    ticker = request.args.get("ticker", "").upper()  # Retrieve the ticker from the query string
    if not ticker:
        return jsonify({"error": "Missing ticker symbol"}), 400  # Error if ticker is missing

    try:
        stock_data = fetch_stock_data(ticker)  # Fetch stock data based on the ticker
        chart_base64 = generate_chart_base64(ticker)  # Generate a stock chart
        if chart_base64 is None:
            return jsonify({"error": "Chart could not be generated."}), 500  # Error if chart generation fails

        # Render a new template and pass stock data and chart to it
        return render_template(
            "get_stock_info.html", 
            stock_data=stock_data, 
            chart=chart_base64,
            ticker=ticker
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error if something goes wrong

@app.route('/recommendation_menu')
def recommendation_menu():
    return render_template("recommendation_menu.html")

@app.route('/get_stock_recommendation')
def get_stock_recommendation():
    ticker = request.args.get("ticker", "").upper()  # Retrieve the ticker from the query string
    if not ticker:
        return jsonify({"error": "Missing ticker symbol"}), 400  # Error if ticker is missing

    # Assuming fetch_stock_data and calculate_risk_info are defined elsewhere
    stock_data = fetch_stock_data(ticker)
    risk_info = calculate_risk_info(stock_data)
    risk_tolerance = session.get('risk_tolerance', 'Medium')

    if stock_data is None or risk_info is None:
        return jsonify({"error": f"No P/E ratio or risk data available for {ticker}."})

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

    return render_template(
        "recommendations.html", 
        stock_data=stock_data, 
        risk_info=risk_info,
        ticker=ticker,
        recommendation=recommendation
    )

cache = {}
cache_expired = timedelta(minutes=30)  # Adjust cache expiry time as needed

# Define the function for getting market sentiment
def get_market_sentiment(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if ticker in cache:
        return cache[ticker]  # Return cached result if available

    if response.status_code != 200:
        return {"error": "Failed to fetch market sentiment."}

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract headlines and URLs
    all_headlines = set()
    all_urls = set()

    for anchor in soup.find_all('a', href=True):
        headline = anchor.get_text(strip=True)
        url = anchor['href']
        if ticker.upper() in headline.upper() and headline:
            all_headlines.add(headline)
            all_urls.add(f"{url}")

    if not all_headlines:
        return {"error": f"No news headlines found for {ticker}."}

    headlines_with_urls = [(headline, url) for headline, url in zip(all_headlines, all_urls)]

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for headline, url in headlines_with_urls:
        score = analyzer.polarity_scores(headline)['compound']
        sentiment_scores.append((headline, score, url))

    # Calculate overall sentiment by calling show_overall_results
    overall_sentiment = show_overall_results(sentiment_scores)

    # Cache the results without timestamp
    cache[ticker] = {
        'overall_sentiment': overall_sentiment['label'],
        'sentiment_score': overall_sentiment['score'],
        'headlines': [{"headline": h, "score": score, "url": u} for h, score, u in sentiment_scores]
    }

    return {
        "overall_sentiment": overall_sentiment['label'],
        "sentiment_score": overall_sentiment['score'],
        "headlines": [{"headline": h, "score": score, "url": u} for h, score, u in sentiment_scores]
    }

def show_overall_results(sentiment_scores):
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

@app.route('/get_market_sentiment')
def market_sentiment_page():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return render_template("get_market_sentiment.html", error="Ticker symbol is required.")

    sentiment_data = get_market_sentiment(ticker)

    if "error" in sentiment_data:
        return render_template("get_market_sentiment.html", error=sentiment_data["error"])

    return render_template("get_market_sentiment.html",
                           ticker=ticker,
                           overall_sentiment=sentiment_data['overall_sentiment'],
                           sentiment_score=sentiment_data['sentiment_score'],
                           headlines=sentiment_data['headlines'])

# Log out -----------------------------------------------------------------
@app.route("/logout", methods=["POST"])
def logout():
    # Clear the session
    session.pop('username', None)

    # Redirect to the homepage after logout
    return redirect(url_for('homepage'))

cache = {}
cache_expired = timedelta(minutes=30)

if __name__ == '__main__':
    app.run(debug=True)
