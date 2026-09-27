import io
# .venv\Scripts\python.exe app.py
# http://predictx.local:5000
# .\.venv\Scripts\Activate.ps1
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()


from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.preprocessing import MinMaxScaler

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Email, Length


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "dev-only-change-me"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "sqlite:///stock_predictor.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


# ============================================================
# EXTENSIONS
# ============================================================

db = SQLAlchemy(app)

csrf = CSRFProtect(app)

login_manager = LoginManager(app)

login_manager.login_view = "login"

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["50 per minute"],
)


# ============================================================
# DATABASE MODELS
# ============================================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(80),
        nullable=False
    )

    email = db.Column(
        db.String(160),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    predictions = db.relationship(
        "Prediction",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Prediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    ticker = db.Column(
        db.String(30),
        nullable=False
    )

    predicted_price = db.Column(
        db.Float,
        nullable=False
    )

    actual_price = db.Column(
        db.Float,
        nullable=False
    )

    mae = db.Column(
        db.Float,
        nullable=False
    )

    rmse = db.Column(
        db.Float,
        nullable=False
    )

    r2 = db.Column(
        db.Float,
        nullable=False
    )

    model = db.Column(
        db.String(50),
        nullable=False,
        default="Linear Regression"
    )

    period = db.Column(
        db.String(10),
        nullable=False,
        default="1y"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class Watchlist(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    ticker = db.Column(
        db.String(30),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class PriceAlert(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    ticker = db.Column(
        db.String(30),
        nullable=False
    )

    target_price = db.Column(
        db.Float,
        nullable=False
    )

    direction = db.Column(
        db.String(10),
        nullable=False,
        default="above"
    )

    active = db.Column(
        db.Boolean,
        default=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class PortfolioHolding(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    ticker = db.Column(
        db.String(30),
        nullable=False
    )

    quantity = db.Column(
        db.Float,
        nullable=False
    )

    avg_price = db.Column(
        db.Float,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# ============================================================
# FORMS
# ============================================================

class LoginForm(FlaskForm):

    email = StringField(
        "Email",
        validators=[
            DataRequired(),
            Email(),
            Length(max=160)
        ]
    )

    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8, max=128)
        ]
    )


class RegisterForm(FlaskForm):

    name = StringField(
        "Full name",
        validators=[
            DataRequired(),
            Length(min=2, max=80)
        ]
    )

    email = StringField(
        "Email",
        validators=[
            DataRequired(),
            Email(),
            Length(max=160)
        ]
    )

    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(min=8, max=128)
        ]
    )


# ============================================================
# LOGIN MANAGER
# ============================================================

@login_manager.user_loader
def load_user(uid):

    return db.session.get(
        User,
        int(uid)
    )


# ============================================================
# GLOBAL SETTINGS
# ============================================================

PERIODS = {
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
}


# ============================================================
# VALIDATION
# ============================================================

def valid_ticker(ticker):

    return bool(
        re.fullmatch(
            r"[A-Za-z0-9.\-]{1,20}",
            ticker or ""
        )
    )


# ============================================================
# YAHOO FINANCE DATA
# ============================================================

def fetch_data(
    ticker,
    period="1y"
):

    ticker = str(
        ticker
    ).upper().strip()

    if not valid_ticker(ticker):

        raise ValueError(
            "Invalid stock symbol."
        )

    if period not in PERIODS:

        period = "1y"

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:

        raise ValueError(
            "No Yahoo Finance data found. "
            "Check the symbol."
        )

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    df = df.reset_index()

    df["Date"] = (
        pd.to_datetime(
            df["Date"]
        ).dt.strftime("%Y-%m-%d")
    )

    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    return (
        df[columns]
        .dropna()
        .reset_index(drop=True)
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):

    out = df.copy()

    close = (
        out["Close"]
        .astype(float)
    )

    delta = close.diff()

    gain = (
        delta
        .clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = (
        gain /
        loss.replace(
            0,
            np.nan
        )
    )

    # Simple Moving Average
    out["SMA20"] = (
        close
        .rolling(20)
        .mean()
    )

    # Exponential Moving Average
    out["EMA20"] = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # RSI
    out["RSI14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # MACD
    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    out["MACD"] = (
        ema12 -
        ema26
    )

    out["MACDSignal"] = (
        out["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # Bollinger Bands
    std20 = (
        close
        .rolling(20)
        .std()
    )

    out["BBUpper"] = (
        out["SMA20"] +
        2 * std20
    )

    out["BBLower"] = (
        out["SMA20"] -
        2 * std20
    )

    # Daily percentage return
    out["Returns"] = (
        close.pct_change() * 100
    )

    return out


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

def make_features(
    df,
    window=10
):

    close = (
        df["Close"]
        .astype(float)
        .values
    )

    X = []
    y = []

    for i in range(
        window,
        len(close)
    ):

        X.append(
            close[
                i - window:i
            ]
        )

        y.append(
            close[i]
        )

    return (
        np.asarray(X),
        np.asarray(y)
    )


# ============================================================
# MODEL SUITE
# ============================================================

def model_suite(df):

    X, y = make_features(df)

    if len(X) < 40:

        raise ValueError(
            "Use a period containing at least "
            "50 trading sessions."
        )

    split = max(
        int(len(X) * 0.8),
        1
    )

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=250,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=3,
                random_state=42
            ),
    }

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        if len(X_test):

            test_pred = model.predict(
                X_test
            )

            mae = float(
                mean_absolute_error(
                    y_test,
                    test_pred
                )
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        test_pred
                    )
                )
            )

            r2 = float(
                r2_score(
                    y_test,
                    test_pred
                )
            ) if len(test_pred) > 1 else 0.0

        else:

            test_pred = np.array([])

            mae = 0.0
            rmse = 0.0
            r2 = 0.0

        latest_window = (
            df["Close"]
            .astype(float)
            .values[-10:]
        )

        next_price = float(
            model.predict(
                [latest_window]
            )[0]
        )

        results[name] = {

            "model": model,

            "mae": mae,

            "rmse": rmse,

            "r2": r2,

            "next_price": next_price,
        }

    return results


# ============================================================
# FAST WALK-FORWARD BACKTEST
# ============================================================

def backtest(
    df,
    model_name="Random Forest"
):

    X, y = make_features(df)

    if len(X) < 50:

        raise ValueError(
            "Not enough data for backtesting. "
            "Use 2y or 5y."
        )

    # Smaller models for repeated
    # walk-forward training.
    model_factory = {

        "Linear Regression":
            lambda:
            LinearRegression(),

        "Random Forest":
            lambda:
            RandomForestRegressor(
                n_estimators=60,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            lambda:
            GradientBoostingRegressor(
                n_estimators=80,
                learning_rate=0.04,
                max_depth=3,
                random_state=42
            ),
    }

    if model_name not in model_factory:

        model_name = "Random Forest"

    factory = model_factory[
        model_name
    ]

    predictions = []
    actuals = []
    dates = []

    # Start testing after 65%
    # of the historical observations.
    start = max(
        40,
        int(len(X) * 0.65)
    )

    # IMPORTANT:
    # Test every 5th trading session
    # instead of every single session.
    #
    # This preserves the chronological
    # walk-forward methodology while
    # greatly reducing computation time.
    test_indices = range(
        start,
        len(X),
        5
    )

    for i in test_indices:

        model = factory()

        # Only historical data before
        # the current prediction is used.
        model.fit(
            X[:i],
            y[:i]
        )

        prediction = float(
            model.predict(
                [X[i]]
            )[0]
        )

        predictions.append(
            prediction
        )

        actuals.append(
            float(y[i])
        )

        dates.append(
            df["Date"].iloc[
                i + 10
            ]
        )

    if not predictions:

        raise ValueError(
            "Unable to generate "
            "backtest predictions."
        )

    # MAE
    mae = float(
        mean_absolute_error(
            actuals,
            predictions
        )
    )

    # RMSE
    rmse = float(
        np.sqrt(
            mean_squared_error(
                actuals,
                predictions
            )
        )
    )

    # --------------------------------------------------------
    # Directional accuracy
    # --------------------------------------------------------

    if len(actuals) > 1:

        actual_direction = np.sign(
            np.diff(actuals)
        )

        predicted_direction = np.sign(
            np.diff(predictions)
        )

        directional_accuracy = float(
            np.mean(
                actual_direction ==
                predicted_direction
            ) * 100
        )

    else:

        directional_accuracy = 0.0

    return {

        "dates": dates,

        "actual": actuals,

        "predicted": predictions,

        "mae": mae,

        "rmse": rmse,

        "directional_accuracy":
            directional_accuracy,
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):

    return (
        round(
            float(value),
            4
        )
        if pd.notna(value)
        else None
    )


def current_price(ticker):

    df = fetch_data(
        ticker,
        "1mo"
    )

    return float(
        df["Close"].iloc[-1]
    )


# ============================================================
# DATABASE MIGRATION
# ============================================================

def migrate_schema():

    tables = {

        "prediction": {

            "model":
                "VARCHAR(50) "
                "DEFAULT "
                "'Linear Regression'",

            "period":
                "VARCHAR(10) "
                "DEFAULT '1y'",
        }
    }

    for table, columns in tables.items():

        existing = {
            row[1]
            for row in db.session.execute(
                db.text(
                    f"PRAGMA table_info({table})"
                )
            )
        }

        if existing:

            for column, definition in columns.items():

                if column not in existing:

                    db.session.execute(
                        db.text(
                            f"ALTER TABLE "
                            f"{table} "
                            f"ADD COLUMN "
                            f"{column} "
                            f"{definition}"
                        )
                    )

    db.session.commit()


# ============================================================
# SECURITY
# ============================================================

@app.context_processor
def inject_year():

    return {
        "current_year":
            datetime.now().year
    }


@app.after_request
def add_security_headers(
    response
):

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = (
        "strict-origin-when-cross-origin"
    )

    response.headers[
        "Permissions-Policy"
    ] = (
        "geolocation=(), "
        "microphone=(), "
        "camera=()"
    )

    return response


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ------------------------------------------------------------
# REGISTER
# ------------------------------------------------------------

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    form = RegisterForm()

    if form.validate_on_submit():

        email = (
            form.email.data
            .strip()
            .lower()
        )

        if User.query.filter_by(
            email=email
        ).first():

            flash(
                "An account with this email "
                "already exists.",
                "error"
            )

        else:

            user = User(

                name=
                    form.name.data.strip(),

                email=email,

                password_hash=
                    generate_password_hash(
                        form.password.data
                    ),
            )

            db.session.add(user)

            db.session.commit()

            flash(
                "Account created successfully. "
                "Please sign in.",
                "success"
            )

            return redirect(
                url_for("login")
            )

    return render_template(
        "register.html",
        form=form
    )


# ------------------------------------------------------------
# LOGIN
# ------------------------------------------------------------

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    form = LoginForm()

    if form.validate_on_submit():

        email = (
            form.email.data
            .strip()
            .lower()
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if (
            user
            and check_password_hash(
                user.password_hash,
                form.password.data
            )
        ):

            login_user(user)

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template(
        "login.html",
        form=form
    )


# ------------------------------------------------------------
# LOGOUT
# ------------------------------------------------------------

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("index")
    )


# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    history = (
        Prediction.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Prediction.created_at.desc()
        )
        .limit(8)
        .all()
    )

    watchlist = (
        Watchlist.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Watchlist.created_at.desc()
        )
        .limit(8)
        .all()
    )

    holdings = (
        PortfolioHolding.query
        .filter_by(
            user_id=current_user.id
        )
        .all()
    )

    alerts = (
        PriceAlert.query
        .filter_by(
            user_id=current_user.id,
            active=True
        )
        .count()
    )

    total = (
        Prediction.query
        .filter_by(
            user_id=current_user.id
        )
        .count()
    )

    symbols = (
        db.session
        .query(Prediction.ticker)
        .filter_by(
            user_id=current_user.id
        )
        .distinct()
        .count()
    )

    latest = (
        Prediction.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Prediction.created_at.desc()
        )
        .first()
    )

    portfolio_value = 0.0
    portfolio_cost = 0.0

    portfolio_rows = []

    for holding in holdings:

        try:

            price = current_price(
                holding.ticker
            )

            value = (
                price *
                holding.quantity
            )

            cost = (
                holding.avg_price *
                holding.quantity
            )

            portfolio_value += value

            portfolio_cost += cost

            portfolio_rows.append({

                "ticker":
                    holding.ticker,

                "quantity":
                    holding.quantity,

                "avg_price":
                    holding.avg_price,

                "price":
                    price,

                "value":
                    value,

                "pnl":
                    value - cost,
            })

        except Exception:

            portfolio_rows.append({

                "ticker":
                    holding.ticker,

                "quantity":
                    holding.quantity,

                "avg_price":
                    holding.avg_price,

                "price":
                    None,

                "value":
                    None,

                "pnl":
                    None,
            })

    return render_template(

        "dashboard.html",

        history=history,

        watchlist=watchlist,

        alerts=alerts,

        total=total,

        symbols=symbols,

        latest=latest,

        portfolio_rows=
            portfolio_rows,

        portfolio_value=
            portfolio_value,

        portfolio_cost=
            portfolio_cost,
    )


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

@app.route("/predict")
@login_required
def predict_page():

    return render_template(
        "predict.html"
    )


# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

@app.route("/history")
@login_required
def history():

    rows = (
        Prediction.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Prediction.created_at.desc()
        )
        .all()
    )

    return render_template(
        "history.html",
        rows=rows
    )


# ------------------------------------------------------------
# WATCHLIST
# ------------------------------------------------------------

@app.route("/watchlist")
@login_required
def watchlist_page():

    items = (
        Watchlist.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Watchlist.created_at.desc()
        )
        .all()
    )

    return render_template(
        "watchlist.html",
        items=items
    )


# ------------------------------------------------------------
# PORTFOLIO
# ------------------------------------------------------------

@app.route("/portfolio")
@login_required
def portfolio_page():

    holdings = (
        PortfolioHolding.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            PortfolioHolding.created_at.desc()
        )
        .all()
    )

    return render_template(
        "portfolio.html",
        holdings=holdings
    )


# ------------------------------------------------------------
# ALERTS
# ------------------------------------------------------------

@app.route("/alerts")
@login_required
def alerts_page():

    alerts = (
        PriceAlert.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            PriceAlert.created_at.desc()
        )
        .all()
    )

    return render_template(
        "alerts.html",
        alerts=alerts
    )


# ------------------------------------------------------------
# NEWS
# ------------------------------------------------------------

@app.route("/news")
@login_required
def news_page():

    return render_template(
        "news.html"
    )


# ------------------------------------------------------------
# ANALYTICS
# ------------------------------------------------------------

@app.route("/analytics")
@login_required
def analytics_page():

    return render_template(
        "analytics.html"
    )


# ------------------------------------------------------------
# PROFILE
# ------------------------------------------------------------

@app.route(
    "/profile",
    methods=["GET", "POST"]
)
@login_required
def profile():

    if request.method == "POST":

        name = (
            request.form
            .get("name", "")
            .strip()
        )

        if len(name) < 2 or len(name) > 80:

            flash(
                "Name must be between "
                "2 and 80 characters.",
                "error"
            )

        else:

            current_user.name = name

            db.session.commit()

            flash(
                "Profile updated.",
                "success"
            )

    return render_template(
        "profile.html"
    )


# ------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# STOCK API
# ============================================================

@app.route("/api/stock")
@login_required
def api_stock():

    try:

        ticker = (
            request.args
            .get("ticker", "AAPL")
            .strip()
            .upper()
        )

        if not valid_ticker(ticker):

            return jsonify(
                success=False,
                error="Invalid stock symbol."
            ), 400

        period = request.args.get(
            "period",
            "1y"
        )

        df = add_indicators(
            fetch_data(
                ticker,
                period
            )
        )

        results = model_suite(df)

        selected = results[
            "Linear Regression"
        ]

        latest = float(
            df["Close"].iloc[-1]
        )

        return jsonify(

            success=True,

            ticker=ticker,

            latest=round(
                latest,
                2
            ),

            predicted=round(
                selected["next_price"],
                2
            ),

            difference=round(
                selected["next_price"]
                - latest,
                2
            ),

            mae=round(
                selected["mae"],
                2
            ),

            rmse=round(
                selected["rmse"],
                2
            ),

            r2=round(
                selected["r2"],
                4
            ),

            dates=df[
                "Date"
            ].tolist(),

            close=[
                round(
                    float(x),
                    2
                )
                for x in df["Close"]
            ],

            open=[
                round(
                    float(x),
                    2
                )
                for x in df["Open"]
            ],

            high=[
                round(
                    float(x),
                    2
                )
                for x in df["High"]
            ],

            low=[
                round(
                    float(x),
                    2
                )
                for x in df["Low"]
            ],

            volume=[
                int(x)
                for x in df["Volume"]
            ],

            indicators={

                column: [
                    safe_float(x)
                    for x in df[column]
                ]

                for column in [

                    "SMA20",
                    "EMA20",
                    "RSI14",
                    "MACD",
                    "MACDSignal",
                    "BBUpper",
                    "BBLower",
                ]
            }
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# STOCK SEARCH / AUTOCOMPLETE
# ============================================================

@app.route("/api/search")
@login_required
def search_stocks():

    query = (
        request.args
        .get("q", "")
        .strip()
    )

    if not query:

        return jsonify([])

    try:

        quotes = (
            yf.Search(query)
            .quotes
        )

        results = []

        for stock in quotes[:10]:

            symbol = stock.get(
                "symbol"
            )

            name = (
                stock.get("longname")
                or stock.get("shortname")
            )

            if (
                symbol
                and name
                and valid_ticker(symbol)
            ):

                results.append({

                    "symbol":
                        symbol,

                    "name":
                        name,
                })

        return jsonify(
            results
        )

    except Exception:

        return jsonify(
            error="Unable to search stocks"
        ), 500


# ============================================================
# MODEL COMPARISON API
# ============================================================

@app.route("/api/models")
@login_required
def api_models():

    try:

        ticker = (
            request.args
            .get("ticker", "AAPL")
            .strip()
            .upper()
        )

        period = request.args.get(
            "period",
            "1y"
        )

        if not valid_ticker(ticker):

            return jsonify(
                success=False,
                error="Invalid stock symbol."
            ), 400

        results = model_suite(
            fetch_data(
                ticker,
                period
            )
        )

        clean = {}

        for name, result in results.items():

            clean[name] = {

                key: round(
                    float(value),
                    4
                )

                for key, value
                in result.items()

                if key != "model"
            }

        return jsonify(

            success=True,

            ticker=ticker,

            models=clean,
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# BACKTEST API
# ============================================================

@app.route("/api/backtest")
@login_required
def api_backtest():

    try:

        ticker = (
            request.args
            .get("ticker", "AAPL")
            .strip()
            .upper()
        )

        period = request.args.get(
            "period",
            "2y"
        )

        model = request.args.get(
            "model",
            "Random Forest"
        )

        if not valid_ticker(ticker):

            return jsonify(
                success=False,
                error="Invalid stock symbol."
            ), 400

        if period not in PERIODS:

            period = "2y"

        if model not in {
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
        }:

            model = "Random Forest"

        df = fetch_data(
            ticker,
            period
        )

        result = backtest(
            df,
            model
        )

        return jsonify(

            success=True,

            ticker=ticker,

            model=model,

            **result
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# SAVE PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
@login_required
@limiter.limit(
    "50 per minute"
)
def api_predict():

    try:

        body = (
            request
            .get_json(
                silent=True
            )
            or {}
        )

        ticker = str(
            body.get(
                "ticker",
                "AAPL"
            )
        ).upper().strip()

        if not valid_ticker(ticker):

            return jsonify(
                success=False,
                error="Invalid stock symbol."
            ), 400

        period = str(
            body.get(
                "period",
                "1y"
            )
        )

        model_name = str(
            body.get(
                "model",
                "Linear Regression"
            )
        )

        df = fetch_data(
            ticker,
            period
        )

        results = model_suite(df)

        if model_name not in results:

            model_name = (
                "Linear Regression"
            )

        result = results[
            model_name
        ]

        actual = float(
            df["Close"].iloc[-1]
        )

        prediction = Prediction(

            user_id=
                current_user.id,

            ticker=
                ticker,

            predicted_price=
                result["next_price"],

            actual_price=
                actual,

            mae=
                result["mae"],

            rmse=
                result["rmse"],

            r2=
                result["r2"],

            model=
                model_name,

            period=
                period,
        )

        db.session.add(
            prediction
        )

        db.session.commit()

        return jsonify(

            success=True,

            ticker=ticker,

            predicted=round(
                result["next_price"],
                2
            ),

            model=model_name,
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# WATCHLIST API
# ============================================================

@app.route(
    "/api/watchlist",
    methods=[
        "GET",
        "POST",
        "DELETE"
    ]
)
@login_required
def api_watchlist():

    if request.method == "GET":

        items = (
            Watchlist.query
            .filter_by(
                user_id=current_user.id
            )
            .all()
        )

        return jsonify(

            items=[

                {
                    "id": item.id,
                    "ticker": item.ticker
                }

                for item in items
            ]
        )

    body = (
        request
        .get_json(
            silent=True
        )
        or {}
    )

    ticker = str(
        body.get(
            "ticker",
            ""
        )
    ).upper().strip()

    if not valid_ticker(ticker):

        return jsonify(
            success=False,
            error="Invalid ticker."
        ), 400

    if request.method == "POST":

        exists = (
            Watchlist.query
            .filter_by(
                user_id=current_user.id,
                ticker=ticker
            )
            .first()
        )

        if not exists:

            db.session.add(
                Watchlist(
                    user_id=
                        current_user.id,
                    ticker=
                        ticker
                )
            )

            db.session.commit()

        return jsonify(
            success=True
        )

    item = (
        Watchlist.query
        .filter_by(
            user_id=current_user.id,
            ticker=ticker
        )
        .first()
    )

    if item:

        db.session.delete(
            item
        )

        db.session.commit()

    return jsonify(
        success=True
    )


# ============================================================
# PORTFOLIO API
# ============================================================

@app.route(
    "/api/portfolio",
    methods=[
        "POST",
        "DELETE"
    ]
)
@login_required
def api_portfolio():

    body = (
        request
        .get_json(
            silent=True
        )
        or {}
    )

    ticker = str(
        body.get(
            "ticker",
            ""
        )
    ).upper().strip()

    if not valid_ticker(ticker):

        return jsonify(
            success=False,
            error="Invalid ticker."
        ), 400

    if request.method == "DELETE":

        item = (
            PortfolioHolding.query
            .filter_by(
                user_id=current_user.id,
                ticker=ticker
            )
            .first()
        )

        if item:

            db.session.delete(
                item
            )

            db.session.commit()

        return jsonify(
            success=True
        )

    try:

        quantity = float(
            body.get(
                "quantity",
                0
            )
        )

        avg_price = float(
            body.get(
                "avg_price",
                0
            )
        )

        if (
            quantity <= 0
            or avg_price < 0
        ):

            raise ValueError(
                "Quantity must be positive "
                "and average price cannot "
                "be negative."
            )

        item = (
            PortfolioHolding.query
            .filter_by(
                user_id=current_user.id,
                ticker=ticker
            )
            .first()
        )

        if item:

            total_qty = (
                item.quantity +
                quantity
            )

            item.avg_price = (

                (
                    item.quantity *
                    item.avg_price
                )

                +

                (
                    quantity *
                    avg_price
                )

            ) / total_qty

            item.quantity = (
                total_qty
            )

        else:

            db.session.add(
                PortfolioHolding(

                    user_id=
                        current_user.id,

                    ticker=
                        ticker,

                    quantity=
                        quantity,

                    avg_price=
                        avg_price,
                )
            )

        db.session.commit()

        return jsonify(
            success=True
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# PRICE ALERTS API
# ============================================================

@app.route(
    "/api/alerts",
    methods=[
        "POST",
        "DELETE"
    ]
)
@login_required
def api_alerts():

    body = (
        request
        .get_json(
            silent=True
        )
        or {}
    )

    if request.method == "DELETE":

        try:

            alert_id = int(
                body.get(
                    "id",
                    0
                )
            )

        except Exception:

            alert_id = 0

        alert = db.session.get(
            PriceAlert,
            alert_id
        )

        if (
            alert
            and alert.user_id ==
                current_user.id
        ):

            db.session.delete(
                alert
            )

            db.session.commit()

        return jsonify(
            success=True
        )

    ticker = str(
        body.get(
            "ticker",
            ""
        )
    ).upper().strip()

    try:

        target = float(
            body.get(
                "target_price",
                0
            )
        )

    except Exception:

        target = 0

    direction = str(
        body.get(
            "direction",
            "above"
        )
    )

    if (
        not valid_ticker(ticker)
        or target <= 0
        or direction not in {
            "above",
            "below"
        }
    ):

        return jsonify(

            success=False,

            error=(
                "Enter a valid ticker, "
                "target price and direction."
            )

        ), 400

    db.session.add(
        PriceAlert(

            user_id=
                current_user.id,

            ticker=
                ticker,

            target_price=
                target,

            direction=
                direction,
        )
    )

    db.session.commit()

    return jsonify(
        success=True
    )


# ============================================================
# ALERT CHECK API
# ============================================================

@app.route(
    "/api/alerts/check"
)
@login_required
def api_alerts_check():

    alerts = (
        PriceAlert.query
        .filter_by(
            user_id=current_user.id,
            active=True
        )
        .all()
    )

    triggered = []

    for alert in alerts:

        try:

            price = current_price(
                alert.ticker
            )

            if alert.direction == "above":

                hit = (
                    price >=
                    alert.target_price
                )

            else:

                hit = (
                    price <=
                    alert.target_price
                )

            if hit:

                triggered.append({

                    "id":
                        alert.id,

                    "ticker":
                        alert.ticker,

                    "price":
                        round(
                            price,
                            2
                        ),

                    "target":
                        alert.target_price,

                    "direction":
                        alert.direction,
                })

        except Exception:

            pass

    return jsonify(

        success=True,

        triggered=triggered
    )


# ============================================================
# NEWS API
# ============================================================

@app.route("/api/news")
@login_required
def api_news():

    ticker = (
        request.args
        .get(
            "ticker",
            "AAPL"
        )
        .upper()
        .strip()
    )

    if not valid_ticker(ticker):

        return jsonify(
            success=False,
            error="Invalid ticker."
        ), 400

    try:

        news = (
            yf.Ticker(ticker)
            .news
            or []
        )

        rows = []

        for item in news[:12]:

            content = (
                item.get(
                    "content",
                    item
                )
            )

            title = (
                content.get("title")
                or item.get("title")
            )

            canonical_url = (
                content.get(
                    "canonicalUrl",
                    {}
                )
            )

            link = (
                canonical_url.get("url")
                or item.get("link")
            )

            published = (
                content.get("pubDate")
                or item.get(
                    "providerPublishTime"
                )
            )

            if title:

                rows.append({

                    "title":
                        title,

                    "link":
                        link,

                    "published":
                        published,
                })

        return jsonify(

            success=True,

            ticker=ticker,

            news=rows
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# LSTM API
# ============================================================

@app.route("/api/lstm")
@login_required
@limiter.limit(
    "5 per minute"
)
def api_lstm():

    try:

        from tensorflow.keras.models import Sequential

        from tensorflow.keras.layers import (
            LSTM,
            Dense,
            Dropout,
        )

    except Exception:

        return jsonify(

            success=False,

            error=(
                "LSTM requires TensorFlow. "
                "Install it with: "
                "pip install tensorflow"
            )

        ), 503

    try:

        ticker = (
            request.args
            .get(
                "ticker",
                "AAPL"
            )
            .upper()
            .strip()
        )

        period = request.args.get(
            "period",
            "2y"
        )

        if not valid_ticker(ticker):

            return jsonify(
                success=False,
                error="Invalid stock symbol."
            ), 400

        df = fetch_data(
            ticker,
            period
        )

        prices = (
            df["Close"]
            .astype(float)
            .values
            .reshape(-1, 1)
        )

        if len(prices) < 100:

            raise ValueError(
                "LSTM needs at least "
                "100 trading sessions. "
                "Use 2y or 5y."
            )

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(
            prices
        )

        window = 30

        X = []
        y = []

        for i in range(
            window,
            len(scaled)
        ):

            X.append(
                scaled[
                    i - window:i
                ]
            )

            y.append(
                scaled[i]
            )

        X = np.asarray(X)
        y = np.asarray(y)

        split = int(
            len(X) * 0.8
        )

        model = Sequential([

            LSTM(
                48,
                return_sequences=True,
                input_shape=(
                    window,
                    1
                )
            ),

            Dropout(0.15),

            LSTM(24),

            Dense(1),
        ])

        model.compile(
            optimizer="adam",
            loss="mse"
        )

        model.fit(
            X[:split],
            y[:split],
            epochs=6,
            batch_size=16,
            verbose=0
        )

        test = model.predict(
            X[split:],
            verbose=0
        ).reshape(
            -1,
            1
        )

        actual = (
            scaler
            .inverse_transform(
                y[split:]
                .reshape(-1, 1)
            )
            .ravel()
        )

        prediction = (
            scaler
            .inverse_transform(
                test
            )
            .ravel()
        )

        next_scaled = (
            model.predict(
                scaled[-window:]
                .reshape(
                    1,
                    window,
                    1
                ),
                verbose=0
            )
        )

        next_price = float(

            scaler
            .inverse_transform(
                next_scaled
            )[0, 0]

        )

        return jsonify(

            success=True,

            ticker=ticker,

            next_price=round(
                next_price,
                2
            ),

            mae=round(
                float(
                    mean_absolute_error(
                        actual,
                        prediction
                    )
                ),
                2
            ),

            rmse=round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            prediction
                        )
                    )
                ),
                2
            ),

            r2=round(
                float(
                    r2_score(
                        actual,
                        prediction
                    )
                ),
                4
            )
        )

    except Exception as e:

        return jsonify(
            success=False,
            error=str(e)
        ), 400


# ============================================================
# CSV EXPORT
# ============================================================

@app.route("/export/csv")
@login_required
def export_csv():

    rows = (
        Prediction.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Prediction.created_at.desc()
        )
        .all()
    )

    data = [

        {

            "Date":
                row.created_at.strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "Ticker":
                row.ticker,

            "Model":
                row.model,

            "Period":
                row.period,

            "Predicted":
                row.predicted_price,

            "Actual":
                row.actual_price,

            "MAE":
                row.mae,

            "RMSE":
                row.rmse,

            "R2":
                row.r2,
        }

        for row in rows
    ]

    csv_data = (
        pd.DataFrame(data)
        .to_csv(index=False)
    )

    return Response(

        csv_data,

        mimetype="text/csv",

        headers={
            "Content-Disposition":
                "attachment; "
                "filename="
                "predictx_history.csv"
        }
    )


# ============================================================
# PDF EXPORT
# ============================================================

@app.route("/export/pdf")
@login_required
def export_pdf():

    from reportlab.lib.pagesizes import A4

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    from reportlab.lib import colors

    from reportlab.lib.styles import (
        getSampleStyleSheet
    )

    rows = (
        Prediction.query
        .filter_by(
            user_id=current_user.id
        )
        .order_by(
            Prediction.created_at.desc()
        )
        .all()
    )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=30,

        leftMargin=30,

        topMargin=30,

        bottomMargin=30,
    )

    styles = (
        getSampleStyleSheet()
    )

    story = [

        Paragraph(
            "PredictX — Prediction History",
            styles["Title"]
        ),

        Spacer(
            1,
            12
        ),

        Paragraph(
            f"Account: "
            f"{current_user.email}",
            styles["Normal"]
        ),

        Spacer(
            1,
            12
        ),
    ]

    table_data = [

        [
            "Date",
            "Ticker",
            "Model",
            "Predicted",
            "Actual",
            "MAE",
            "RMSE",
            "R²",
        ]
    ]

    for row in rows:

        table_data.append([

            row.created_at.strftime(
                "%d %b %Y"
            ),

            row.ticker,

            row.model,

            f"{row.predicted_price:.2f}",

            f"{row.actual_price:.2f}",

            f"{row.mae:.2f}",

            f"{row.rmse:.2f}",

            f"{row.r2:.2f}",
        ])

    if len(table_data) == 1:

        table_data.append([

            "No predictions saved",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ])

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#1b2330"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.grey
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
        ])
    )

    story.append(
        table
    )

    story.append(
        Spacer(
            1,
            16
        )
    )

    story.append(

        Paragraph(

            "Educational use only — "
            "historical-data models do not "
            "guarantee future market outcomes.",

            styles["Normal"]
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return Response(

        buffer.read(),

        mimetype="application/pdf",

        headers={
            "Content-Disposition":
                "attachment; "
                "filename="
                "predictx_history.pdf"
        }
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

with app.app_context():

    db.create_all()

    migrate_schema()


# ============================================================
# RUN APPLICATION
# ============================================================

@app.route("/api/ai-analysis", methods=["POST"])
@login_required
def ai_analysis():

    try:
        data = request.get_json() or {}

        ticker = str(data.get("ticker", "")).strip().upper()

        if not ticker:
            return jsonify({
                "error": "Please enter a stock symbol."
            }), 400

        if not valid_ticker(ticker):
            return jsonify({
                "error": "Invalid stock symbol."
            }), 400

        # --------------------------------------------------
        # Receive values from PredictX ML dashboard
        # --------------------------------------------------

        latest = float(data.get("latest", 0) or 0)
        predicted = float(data.get("predicted", 0) or 0)
        difference = float(data.get("difference", 0) or 0)

        mae = float(data.get("mae", 0) or 0)
        rmse = float(data.get("rmse", 0) or 0)
        r2 = float(data.get("r2", 0) or 0)

        model = str(
            data.get("model", "Machine Learning model")
        )

        rsi = data.get("rsi", "N/A")
        macd = data.get("macd", "N/A")
        sma = data.get("sma", "N/A")
        ema = data.get("ema", "N/A")

        # --------------------------------------------------
        # Calculate percentage difference
        # --------------------------------------------------

        if latest != 0:
            percentage_difference = (
                (predicted - latest) / latest
            ) * 100
        else:
            percentage_difference = 0

        # --------------------------------------------------
        # Determine prediction direction
        # --------------------------------------------------

        if predicted > latest:
            direction = "higher"
            direction_text = (
                "The model estimates a slightly higher "
                "next price compared with the latest close."
            )

        elif predicted < latest:
            direction = "lower"
            direction_text = (
                "The model estimates a slightly lower "
                "next price compared with the latest close."
            )

        else:
            direction = "similar"
            direction_text = (
                "The model estimate is approximately equal "
                "to the latest closing price."
            )

        # --------------------------------------------------
        # Interpret R²
        # --------------------------------------------------

        if r2 >= 0.75:
            r2_text = (
                "The validation R² indicates a relatively "
                "strong relationship between the model "
                "features and the observed prices."
            )

        elif r2 >= 0.50:
            r2_text = (
                "The validation R² indicates a moderate "
                "level of explanatory performance."
            )

        elif r2 >= 0:
            r2_text = (
                "The validation R² indicates that the model "
                "has limited explanatory performance."
            )

        else:
            r2_text = (
                "The validation R² is negative, indicating "
                "that the model performs poorly on the "
                "validation data."
            )

        # --------------------------------------------------
        # Interpret RSI
        # --------------------------------------------------

        try:
            rsi_value = float(rsi)

            if rsi_value >= 70:
                rsi_text = (
                    f"RSI is {rsi_value:.2f}, which is in a "
                    "high-momentum/overbought range."
                )

            elif rsi_value <= 30:
                rsi_text = (
                    f"RSI is {rsi_value:.2f}, which is in a "
                    "low-momentum/oversold range."
                )

            else:
                rsi_text = (
                    f"RSI is {rsi_value:.2f}, which is within "
                    "the commonly observed middle range."
                )

        except (ValueError, TypeError):
            rsi_text = "RSI data was not available."

        # --------------------------------------------------
        # Generate PredictX AI-style analysis
        # --------------------------------------------------

        analysis = f"""
🤖 PredictX AI Analyst

Stock: {ticker}

📊 Prediction Summary
The latest closing price is {latest:.2f}.
The {model} model estimates the next price at {predicted:.2f}.

The estimated difference is {difference:+.2f},
which represents approximately {percentage_difference:+.2f}%
relative to the latest closing price.

{direction_text}

📈 Model Performance
The model was evaluated using chronological validation.

MAE: {mae:.2f}
RMSE: {rmse:.2f}
R² Score: {r2:.2f}

{r2_text}

📉 Technical Indicators
{rsi_text}

MACD: {macd}
SMA(20): {sma}
EMA(20): {ema}

🧠 Interpretation
PredictX combines historical stock-price information,
machine-learning features and technical indicators to
produce an experimental next-price estimate.

The prediction should be interpreted together with the
model-error metrics and historical market behaviour.
Short-term stock prices can change because of news,
earnings, market sentiment, economic conditions and
other factors that are not completely captured by the
model.

⚠️ Important
This analysis is generated for educational and academic
purposes. It is not financial advice and does not
guarantee the future movement of the stock price.
Do not use this prediction alone to make investment
decisions.
"""

        return jsonify({
            "analysis": analysis.strip(),
            "ticker": ticker,
            "direction": direction
        })

    except Exception as e:

        print("AI ANALYST ERROR:", e)

        return jsonify({
            "error": "AI analysis could not be generated."
        }), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    