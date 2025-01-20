import sqlite3
import random
import customtkinter as ctk
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import json
import requests
import hashlib
import qrcode
from forex_python.converter import CurrencyRates
import yfinance as yf

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AIBankingSystem:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.fraud_detector = IsolationForest(contamination=0.1)
        self.scaler = StandardScaler()
        
    def analyze_transaction_sentiment(self, description):
        result = self.sentiment_analyzer(description)
        return result[0]['label'], result[0]['score']
    
    def detect_fraud(self, transaction_data):
        scaled_data = self.scaler.fit_transform(transaction_data)
        return self.fraud_detector.fit_predict(scaled_data)
    
    def generate_financial_advice(self, transaction_history, balance):
        # Analyze spending patterns
        spending_categories = self.categorize_transactions(transaction_history)
        
        advice = []
        if balance < 1000:
            advice.append("Consider building an emergency fund.")
        
        # Add category-specific advice
        for category, amount in spending_categories.items():
            if amount > balance * 0.3:
                advice.append(f"Your {category} spending seems high. Consider reducing it.")
                
        return advice
    
    def categorize_transactions(self, transactions):
        categories = {}
        for transaction in transactions:
            # Use NLP to categorize transaction descriptions
            tokens = word_tokenize(transaction[5].lower())
            if any(word in tokens for word in ['food', 'restaurant', 'grocery']):
                categories['Food'] = categories.get('Food', 0) + transaction[3]
            elif any(word in tokens for word in ['transport', 'uber', 'taxi']):
                categories['Transport'] = categories.get('Transport', 0) + transaction[3]
            # Add more categories as needed
        return categories

class ModernBankingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI-Powered Banking System")
        self.geometry("800x600")
        self.ai_system = AIBankingSystem()
        self.conn = sqlite3.connect('smart_bank.db')
        self.setup_database()
        self.current_account = None
        self.currency_converter = CurrencyRates()
        self.configure_theme()
        self.show_login_screen()

    def configure_theme(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

    def setup_database(self):
        cursor = self.conn.cursor()
        # Add new tables and columns for enhanced features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS investment_portfolio (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_number INTEGER,
                stock_symbol TEXT,
                quantity INTEGER,
                purchase_price REAL,
                current_price REAL,
                FOREIGN KEY (account_number) REFERENCES accounts (account_number)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_insights (
                insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_number INTEGER,
                insight_type TEXT,
                insight_text TEXT,
                timestamp TEXT,
                FOREIGN KEY (account_number) REFERENCES accounts (account_number)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget_goals (
                goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_number INTEGER,
                category TEXT,
                target_amount REAL,
                current_amount REAL,
                deadline TEXT,
                FOREIGN KEY (account_number) REFERENCES accounts (account_number)
            )
        ''')
        self.conn.commit()

    def show_smart_dashboard(self):
        self.clear_window()
        
        # Create main container
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header section
        header = ctk.CTkFrame(container)
        header.pack(fill="x", pady=10)
        
        welcome_label = ctk.CTkLabel(
            header, 
            text=f"Welcome back, {self.current_account[1]}", 
            font=("Arial", 24, "bold")
        )
        welcome_label.pack(side="left", padx=10)
        
        # Balance section with animated update
        self.balance_label = ctk.CTkLabel(
            header,
            text=f"${self.current_account[6]:,.2f}",
            font=("Arial", 20)
        )
        self.balance_label.pack(side="right", padx=10)
        
        # Quick Actions
        actions_frame = ctk.CTkFrame(container)
        actions_frame.pack(fill="x", pady=10)
        
        quick_actions = [
            ("Send Money", self.show_transfer_screen),
            ("Investments", self.show_investment_dashboard),
            ("Bill Pay", self.show_bill_pay),
            ("Goals", self.show_goals_screen)
        ]
        
        for text, command in quick_actions:
            btn = ctk.CTkButton(
                actions_frame,
                text=text,
                command=command,
                width=120
            )
            btn.pack(side="left", padx=5)

        # Transaction Analysis
        self.show_transaction_analytics(container)
        
        # AI Insights
        self.show_ai_insights(container)
        
        # Bottom Navigation
        nav_frame = ctk.CTkFrame(container)
        nav_frame.pack(fill="x", side="bottom", pady=10)
        
        nav_items = [
            ("Profile", self.show_profile),
            ("Settings", self.show_settings),
            ("Support", self.show_support),
            ("Logout", self.logout)
        ]
        
        for text, command in nav_items:
            btn = ctk.CTkButton(
                nav_frame,
                text=text,
                command=command,
                width=100
            )
            btn.pack(side="left", padx=5)

    def show_investment_dashboard(self):
        self.clear_window()
        
        # Fetch portfolio data
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT stock_symbol, quantity, purchase_price, current_price 
            FROM investment_portfolio 
            WHERE account_number = ?
        """, (self.current_account[0],))
        portfolio = cursor.fetchall()
        
        # Update current prices
        for stock in portfolio:
            ticker = yf.Ticker(stock[0])
            current_price = ticker.info.get('regularMarketPrice', 0)
            cursor.execute("""
                UPDATE investment_portfolio 
                SET current_price = ? 
                WHERE account_number = ? AND stock_symbol = ?
            """, (current_price, self.current_account[0], stock[0]))
        self.conn.commit()
        
        # Display portfolio summary
        summary_frame = ctk.CTkFrame(self)
        summary_frame.pack(fill="x", padx=20, pady=10)
        
        total_value = sum(stock[1] * stock[3] for stock in portfolio)
        total_cost = sum(stock[1] * stock[2] for stock in portfolio)
        total_gain = total_value - total_cost
        
        ctk.CTkLabel(
            summary_frame,
            text=f"Portfolio Value: ${total_value:,.2f}",
            font=("Arial", 18, "bold")
        ).pack(pady=5)
        
        gain_color = "green" if total_gain > 0 else "red"
        ctk.CTkLabel(
            summary_frame,
            text=f"Total Gain/Loss: ${total_gain:,.2f}",
            text_color=gain_color
        ).pack(pady=5)
        
        # Portfolio visualization
        self.show_portfolio_chart(portfolio)
        
        # Trading interface
        trade_frame = ctk.CTkFrame(self)
        trade_frame.pack(fill="x", padx=20, pady=10)
        
        self.stock_symbol = ctk.CTkEntry(
            trade_frame,
            placeholder_text="Stock Symbol"
        )
        self.stock_symbol.pack(pady=5)
        
        self.quantity = ctk.CTkEntry(
            trade_frame,
            placeholder_text="Quantity"
        )
        self.quantity.pack(pady=5)
        
        ctk.CTkButton(
            trade_frame,
            text="Buy",
            command=self.execute_trade
        ).pack(pady=5)
        
        # Back button
        ctk.CTkButton(
            self,
            text="Back to Dashboard",
            command=self.show_smart_dashboard
        ).pack(pady=10)

    def execute_trade(self):
        symbol = self.stock_symbol.get().upper()
        quantity = int(self.quantity.get())
        
        # Fetch current price
        ticker = yf.Ticker(symbol)
        current_price = ticker.info.get('regularMarketPrice', 0)
        
        total_cost = current_price * quantity
        
        if total_cost > self.current_account[6]:
            self.show_error("Insufficient funds for this trade")
            return
        
        # Execute trade
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO investment_portfolio 
            (account_number, stock_symbol, quantity, purchase_price, current_price)
            VALUES (?, ?, ?, ?, ?)
        """, (self.current_account[0], symbol, quantity, current_price, current_price))
        
        # Update account balance
        cursor.execute("""
            UPDATE accounts 
            SET balance = balance - ? 
            WHERE account_number = ?
        """, (total_cost, self.current_account[0]))
        
        self.conn.commit()
        self.show_success(f"Successfully purchased {quantity} shares of {symbol}")
        self.show_investment_dashboard()

    def show_portfolio_chart(self, portfolio):
        if not portfolio:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pie chart of portfolio allocation
        labels = [stock[0] for stock in portfolio]
        values = [stock[1] * stock[3] for stock in portfolio]
        ax1.pie(values, labels=labels, autopct='%1.1f%%')
        ax1.set_title("Portfolio Allocation")
        
        # Performance chart
        symbols = [stock[0] for stock in portfolio]
        gains = [(stock[3] - stock[2]) / stock[2] * 100 for stock in portfolio]
        ax2.bar(symbols, gains)
        ax2.set_title("Performance by Stock")
        ax2.set_ylabel("Gain/Loss %")
        
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def show_bill_pay(self):
        self.clear_window()
        
        # Bill payment interface
        bill_frame = ctk.CTkFrame(self)
        bill_frame.pack(fill="x", padx=20, pady=10)
        
        self.payee = ctk.CTkEntry(
            bill_frame,
            placeholder_text="Payee Name"
        )
        self.payee.pack(pady=5)
        
        self.bill_amount = ctk.CTkEntry(
            bill_frame,
            placeholder_text="Amount"
        )
        self.bill_amount.pack(pady=5)
        
        self.bill_date = ctk.CTkEntry(
            bill_frame,
            placeholder_text="Due Date (YYYY-MM-DD)"
        )
        self.bill_date.pack(pady=5)
        
        ctk.CTkButton(
            bill_frame,
            text="Pay Bill",
            command=self.process_bill_payment
        ).pack(pady=10)
        
        # Scheduled payments
        self.show_scheduled_payments()
        
        # Back button
        ctk.CTkButton(
            self,
            text="Back to Dashboard",
            command=self.show_smart_dashboard
        ).pack(pady=10)

    def process_bill_payment(self):
        amount = float(self.bill_amount.get())
        if amount > self.current_account[6]:
            self.show_error("Insufficient funds")
            return
            
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE accounts 
            SET balance = balance - ? 
            WHERE account_number = ?
        """, (amount, self.current_account[0]))
        
        cursor.execute("""
            INSERT INTO transactions 
            (account_number, transaction_type, amount, timestamp, remarks)
            VALUES (?, ?, ?, ?, ?)
        """, (self.current_account[0], "Bill Payment", amount, 
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              f"Bill payment to {self.payee.get()}"))
        
        self.conn.commit()
        self.show_success("Bill payment processed successfully")
        self.show_bill_pay()

    def show_goals_screen(self):
        self.clear_window()
        
        # Create new goal interface
        goal_frame = ctk.CTkFrame(self)
        goal_frame.pack(fill="x", padx=20, pady=10)
        
        self.goal_category = ctk.CTkEntry(
            goal_frame,
            placeholder_text="Goal Category"
        )
        self.goal_category.pack(pady=5)
        
        self.goal_amount = ctk.CTkEntry(
            goal_frame,
            placeholder_text="Target Amount"
        )
        self.goal_amount.pack(pady=5)
        
        self.goal_deadline = ctk.CTkEntry(
            goal_frame,
            placeholder_text="Deadline (YYYY-MM-DD)"
        )
        self.goal_deadline.pack(pady=5)
        
        ctk.CTkButton(
            goal_frame,
            text="Set Goal",
            command=self.create_goal
        ).pack(pady=10)
        
        # Display existing goals
        self.show_existing_goals()
        
        # Back button
        ctk.CTkButton(
            self,
            text="Back to Dashboard",
            command=self.show_smart_dashboard
        ).pack(pady=10)

    def create_goal(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO budget_goals 
            (account_number, category, target_amount, current_amount, deadline)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.current_account[0],
            self.goal_category.get(),
            float(self.goal_amount.get()),
            0,
            self.goal_deadline.get()
        ))
        self.conn.commit()
        self.show_success("Goal created successfully")
        self.show_goals_screen()

    def show_existing_goals(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT category, target_amount, current_amount, deadline 
            FROM budget_goals 
            WHERE account_number = ?
        """, (self.current_account[0],))
        goals = cursor.fetchall()
        
        if not goals:
            ctk.CTkLabel(
                self,
                text="No goals set yet"
            ).pack(pady=10)
            return
            
        for goal in goals:
            goal_frame = ctk.CTkFrame(self)
            goal_frame.pack(fill="x", padx=20, pady=5)
            
            progress = (goal[2] / goal[1]) * 100
            
            ctk.CTkLabel(
                goal_frame,
                text=f"{goal[0]}: ${goal[2]:,.2f} / ${goal[1]:,.2f}"
            ).pack(side="left", padx=10)
            
            ctk.CTkProgressBar(
                goal_frame,
                width=200,
                mode="determinate"
            ).pack(side="left", padx=10)
            
            days_left = (datetime.strptime(goal[3], "%Y-%m-%d") - datetime.now()).days
            ctk.CTkLabel(
                goal_frame,
                text=f"{days_left} days left"
            ).pack(side="right", padx=10)

    def show_transaction_analytics(self, container):
        analytics_frame = ctk.CTkFrame(container)
        analytics_frame.pack(fill="x", pady=10)
        
        # Fetch recent transactions
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM transactions 
            WHERE account_number = ? 
            ORDER BY timestamp DESC LIMIT 30
        """, (self.current_account[0],))
        transactions = cursor.fetchall()
        
        # Analyze spending patterns
        spending_data = self.ai_system.categorize_transactions(transactions)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Spending by category pie chart
        if spending_data:
            categories = list(spending_data.keys())
            amounts = list(spending_data.values())
            ax1.pie(amounts, labels=categories, autopct='%1.1f%%')
            ax1.set_title("Spending by Category")
        
        # Transaction timeline
        dates = [datetime.strptime(t[4], "%Y-%m-%d %H:%M:%S") for t in transactions]
        amounts = [t[3] for t in transactions]
        ax2.plot(dates, amounts, marker='o')
        ax2.set_title("Transaction Timeline")
        ax2.tick_params(axis='x', rotation=45)
        
        canvas = FigureCanvasTkAgg(fig, master=analytics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def show_ai_insights(self, container):
        insights_frame = ctk.CTkFrame(container)
        insights_frame.pack(fill="x", pady=10)
        
        # Generate AI insights
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM transactions 
            WHERE account_number = ? 
            ORDER BY timestamp DESC LIMIT 100
        """, (self.current_account[0],))
        transactions = cursor.fetchall()
        
        # Fraud detection
        transaction_data = np.array([[t[3], 
                                    datetime.strptime(t[4], "%Y-%m-%d %H:%M:%S").timestamp()] 
                                   for t in transactions])
        fraud_predictions = self.ai_system.detect_fraud(transaction_data)
        
        if -1 in fraud_predictions:
            ctk.CTkLabel(
                insights_frame,
                text="⚠️ Unusual transaction patterns detected",
                text_color="red"
            ).pack(pady=5)
        
        # Financial advice
        advice = self.ai_system.generate_financial_advice(transactions, self.current_account[6])
        for tip in advice:
            ctk.CTkLabel(
                insights_frame,
                text=f"💡 {tip}",
                text_color="yellow"
            ).pack(pady=2)

    def show_profile(self):
        self.clear_window()
        
        profile_frame = ctk.CTkFrame(self)
        profile_frame.pack(fill="x", padx=20, pady=10)
        
        # Generate QR code for account details
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        account_data = {
            "name": self.current_account[1],
            "account": self.current_account[0],
            "email": self.current_account[4]
        }
        qr.add_data(json.dumps(account_data))
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Display profile information
        ctk.CTkLabel(
            profile_frame,
            text="Profile Information",
            font=("Arial", 20, "bold")
        ).pack(pady=10)
        
        fields = [
            ("Name", self.current_account[1]),
            ("Account Number", self.current_account[0]),
            ("Email", self.current_account[4]),
            ("Phone", self.current_account[3]),
            ("Address", self.current_account[2])
        ]
        
        for label, value in fields:
            field_frame = ctk.CTkFrame(profile_frame)
            field_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(
                field_frame,
                text=label,
                font=("Arial", 14, "bold")
            ).pack(side="left", padx=10)
            
            ctk.CTkLabel(
                field_frame,
                text=str(value)
            ).pack(side="right", padx=10)
        
        # Security features
        security_frame = ctk.CTkFrame(self)
        security_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            security_frame,
            text="Change Password",
            command=self.show_change_password
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            security_frame,
            text="Enable 2FA",
            command=self.setup_2fa
        ).pack(side="left", padx=10)
        
        # Back button
        ctk.CTkButton(
            self,
            text="Back to Dashboard",
            command=self.show_smart_dashboard
        ).pack(pady=10)

    def setup_2fa(self):
        # Generate 2FA secret
        secret = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]
        
        # Save to database (in practice, use proper encryption)
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE accounts 
            SET two_factor_secret = ? 
            WHERE account_number = ?
        """, (secret, self.current_account[0]))
        self.conn.commit()
        
        # Show QR code for 2FA setup
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"otpauth://totp/AIBank:{self.current_account[1]}?secret={secret}&issuer=AIBank")
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Display setup instructions
        self.show_success("2FA has been enabled. Scan the QR code with your authenticator app.")

    def show_settings(self):
        self.clear_window()
        
        settings_frame = ctk.CTkFrame(self)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        # Appearance settings
        appearance_label = ctk.CTkLabel(
            settings_frame,
            text="Appearance",
            font=("Arial", 16, "bold")
        )
        appearance_label.pack(pady=5)
        
        theme_var = ctk.StringVar(value="dark")
        theme_switch = ctk.CTkSwitch(
            settings_frame,
            text="Dark Mode",
            variable=theme_var,
            command=lambda: ctk.set_appearance_mode(theme_var.get())
        )
        theme_switch.pack(pady=5)
        
        # Notification settings
        notification_label = ctk.CTkLabel(
            settings_frame,
            text="Notifications",
            font=("Arial", 16, "bold")
        )
        notification_label.pack(pady=5)
        
        notification_options = [
            "Transaction Alerts",
            "Security Alerts",
            "Investment Updates",
            "Bill Payment Reminders"
        ]
        
        for option in notification_options:
            ctk.CTkCheckBox(
                settings_frame,
                text=option
            ).pack(pady=2)
        
        # Currency settings
        currency_label = ctk.CTkLabel(
            settings_frame,
            text="Preferred Currency",
            font=("Arial", 16, "bold")
        )
        currency_label.pack(pady=5)
        
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD"]
        currency_var = ctk.StringVar(value="USD")
        
        for currency in currencies:
            ctk.CTkRadioButton(
                settings_frame,
                text=currency,
                variable=currency_var,
                value=currency
            ).pack(pady=2)
        
        # Save button
        ctk.CTkButton(
            settings_frame,
            text="Save Settings",
            command=self.save_settings
        ).pack(pady=10)
        
        # Back button
        ctk.CTkButton(
            self,
            text="Back to Dashboard",
            command=self.show_smart_dashboard
        ).pack(pady=10)

    def show_success(self, message):
        """Display success message"""
        success_window = ctk.CTkToplevel(self)
        success_window.title("Success")
        success_window.geometry("300x100")
        
        ctk.CTkLabel(
            success_window,
            text=message,
            text_color="green"
        ).pack(pady=20)
        
        ctk.CTkButton(
            success_window,
            text="OK",
            command=success_window.destroy
        ).pack()

    def show_error(self, message):
        """Display error message"""
        error_window = ctk.CTkToplevel(self)
        error_window.title("Error")
        error_window.geometry("300x100")
        
        ctk.CTkLabel(
            error_window,
            text=message,
            text_color="red"
        ).pack(pady=20)
        
        ctk.CTkButton(
            error_window,
            text="OK",
            command=error_window.destroy
        ).pack()

    def logout(self):
        """Logout and return to login screen"""
        self.current_account = None
        self.show_login_screen()

# Run the application
if __name__ == "__main__":
    app = ModernBankingApp()
    app.mainloop()
