import argparse
import os
import sys
import json
import re
import random
from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import tensorflow as tf
import sentencepiece as spm


def create_comprehensive_upi_transactions() -> List[Tuple[str, int]]:
    """Create comprehensive UPI transaction messages reflecting real Indian usage"""
    
    # Common UPI apps and their formats
    upi_apps = {
        "PhonePe": ["phonepe", "pp", "PHONEPE"],
        "Google Pay": ["gpay", "googlepay", "GPay", "Google Pay"],
        "Paytm": ["paytm", "PAYTM"],
        "BHIM": ["bhim", "BHIM", "upi"],
        "Amazon Pay": ["amazonpay", "amzpay", "Amazon Pay"],
        "MobiKwik": ["mobikwik", "mkwik", "MobiKwik"],
        "Freecharge": ["freecharge", "fcharge", "Freecharge"],
        "Cred": ["cred", "CRED"],
        "Razorpay": ["razorpay", "rzpay", "Razorpay"]
    }
    
    # Common merchant categories and names
    merchants = {
        "Food & Dining": ["Swiggy", "Zomato", "Domino's", "Pizza Hut", "McDonald's", "KFC", "Subway", "Starbucks", "Cafe Coffee Day"],
        "Shopping": ["Amazon", "Flipkart", "Myntra", "Nykaa", "Ajio", "Snapdeal", "ShopClues", "Paytm Mall"],
        "Transport": ["Uber", "Ola", "Rapido", "BlaBlaCar", "RedBus", "IRCTC", "MakeMyTrip", "Goibibo"],
        "Entertainment": ["Netflix", "Amazon Prime", "Hotstar", "Disney+", "Spotify", "Wynk", "JioSaavn", "BookMyShow"],
        "Utilities": ["Electricity Bill", "Gas Bill", "Water Bill", "Mobile Recharge", "DTH Recharge", "Broadband Bill"],
        "Education": ["Coursera", "Udemy", "Byju's", "Unacademy", "Vedantu", "Toppr"],
        "Healthcare": ["Pharmeasy", "1mg", "Netmeds", "MedPlus", "Apollo Pharmacy"],
        "Grocery": ["BigBasket", "Grofers", "Blinkit", "Zepto", "Dunzo", "Reliance Fresh", "DMart"]
    }
    
    # Realistic UPI ID formats
    upi_id_formats = [
        "{name}@{bank}",
        "{name}{number}@{bank}",
        "{name}.{bank}@okicici",
        "{name}@paytm",
        "{name}@phonepe",
        "{name}@gpay",
        "{name}{number}@ybl",
        "{name}@axisbank",
        "{name}@hdfcbank",
        "{name}@sbi"
    ]
    
    # UPI transaction message templates (realistic Indian format)
    upi_templates = [
        "UPI transaction of Rs.{amount} to {merchant} ({upi_id}) successful. Ref: {ref_id}",
        "Rs.{amount} paid to {merchant} via UPI. UPI ID: {upi_id}. Ref: {ref_id}",
        "UPI payment of Rs.{amount} to {merchant} successful. {upi_id}. Ref: {ref_id}",
        "Transaction: Rs.{amount} paid to {merchant} via UPI. ID: {upi_id}. Ref: {ref_id}",
        "UPI: Rs.{amount} sent to {merchant} ({upi_id}). Ref: {ref_id}",
        "Payment of Rs.{amount} to {merchant} via UPI successful. {upi_id}. Ref: {ref_id}",
        "UPI transaction Rs.{amount} to {merchant} completed. {upi_id}. Ref: {ref_id}",
        "Rs.{amount} transferred to {merchant} via UPI. UPI ID: {upi_id}. Ref: {ref_id}",
        "UPI payment Rs.{amount} to {merchant} successful. {upi_id}. Ref: {ref_id}",
        "Transaction successful: Rs.{amount} paid to {merchant} via UPI. {upi_id}. Ref: {ref_id}"
    ]
    
    # Failed UPI transaction templates
    failed_upi_templates = [
        "UPI transaction of Rs.{amount} to {merchant} ({upi_id}) failed. Insufficient balance.",
        "Rs.{amount} payment to {merchant} via UPI failed. Transaction declined.",
        "UPI payment of Rs.{amount} to {merchant} unsuccessful. {upi_id}",
        "Transaction failed: Rs.{amount} to {merchant} via UPI. {upi_id}",
        "UPI: Rs.{amount} payment to {merchant} failed. {upi_id}",
        "Payment of Rs.{amount} to {merchant} via UPI failed. Transaction timeout.",
        "UPI transaction Rs.{amount} to {merchant} failed. {upi_id}",
        "Rs.{amount} transfer to {merchant} via UPI failed. {upi_id}",
        "UPI payment Rs.{amount} to {merchant} failed. Network error. {upi_id}",
        "Transaction failed: Rs.{amount} to {merchant} via UPI. {upi_id}"
    ]
    
    # Bank transfer via UPI templates
    bank_transfer_templates = [
        "UPI transfer of Rs.{amount} to {receiver} ({upi_id}) successful. Ref: {ref_id}",
        "Rs.{amount} transferred to {receiver} via UPI. {upi_id}. Ref: {ref_id}",
        "UPI payment of Rs.{amount} to {receiver} successful. {upi_id}. Ref: {ref_id}",
        "Transfer: Rs.{amount} sent to {receiver} via UPI. {upi_id}. Ref: {ref_id}",
        "UPI: Rs.{amount} transferred to {receiver} ({upi_id}). Ref: {ref_id}",
        "Payment of Rs.{amount} to {receiver} via UPI successful. {upi_id}. Ref: {ref_id}",
        "UPI transfer Rs.{amount} to {receiver} completed. {upi_id}. Ref: {ref_id}",
        "Rs.{amount} sent to {receiver} via UPI. {upi_id}. Ref: {ref_id}",
        "UPI payment Rs.{amount} to {receiver} successful. {upi_id}. Ref: {ref_id}",
        "Transfer successful: Rs.{amount} to {receiver} via UPI. {upi_id}. Ref: {ref_id}"
    ]
    
    # Common Indian names for UPI IDs
    indian_names = [
        "rahul", "priya", "amit", "neha", "rajesh", "kavita", "suresh", "anita", "mohan", "sunita",
        "ramesh", "puja", "dinesh", "meera", "vijay", "rekha", "sanjay", "kavya", "arun", "divya",
        "manish", "shweta", "prakash", "anjali", "nitin", "pooja", "rajiv", "deepa", "sachin", "ritu",
        "vikas", "priyanka", "ajay", "monika", "sandeep", "nisha", "rohit", "tanvi", "ankit", "shivani"
    ]
    
    # Bank codes for UPI IDs
    bank_codes = ["okicici", "ybl", "axisbank", "hdfcbank", "sbi", "paytm", "phonepe", "gpay", "bhim"]
    
    transactions = []
    
    # Generate successful UPI merchant payments
    for _ in range(600):
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        amount = random.randint(10, 5000)
        upi_id = f"{merchant.lower().replace(' ', '')}@paytm"
        ref_id = f"TXN{random.randint(100000000, 999999999)}"
        
        template = random.choice(upi_templates)
        message = template.format(
            amount=amount, merchant=merchant, upi_id=upi_id, ref_id=ref_id
        )
        transactions.append((message, 1))  # 1 = transactional
    
    # Generate failed UPI merchant payments
    for _ in range(150):
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        amount = random.randint(10, 5000)
        upi_id = f"{merchant.lower().replace(' ', '')}@paytm"
        
        template = random.choice(failed_upi_templates)
        message = template.format(
            amount=amount, merchant=merchant, upi_id=upi_id
        )
        transactions.append((message, 1))  # 1 = transactional
    
    # Generate successful UPI bank transfers
    for _ in range(400):
        sender = random.choice(indian_names)
        receiver = random.choice(indian_names)
        amount = random.randint(100, 10000)
        upi_id = random.choice(upi_id_formats).format(
            name=receiver, bank=random.choice(bank_codes), number=random.randint(100, 999)
        )
        ref_id = f"TXN{random.randint(100000000, 999999999)}"
        
        template = random.choice(bank_transfer_templates)
        message = template.format(
            amount=amount, receiver=receiver, upi_id=upi_id, ref_id=ref_id
        )
        transactions.append((message, 1))  # 1 = transactional
    
    # Generate UPI QR code payments
    qr_templates = [
        "UPI QR payment of Rs.{amount} to {merchant} successful. Ref: {ref_id}",
        "Rs.{amount} paid via UPI QR to {merchant}. Ref: {ref_id}",
        "QR payment Rs.{amount} to {merchant} via UPI successful. Ref: {ref_id}",
        "UPI QR: Rs.{amount} paid to {merchant}. Ref: {ref_id}",
        "QR transaction Rs.{amount} to {merchant} via UPI completed. Ref: {ref_id}"
    ]
    
    for _ in range(200):
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        amount = random.randint(10, 2000)
        ref_id = f"TXN{random.randint(100000000, 999999999)}"
        
        template = random.choice(qr_templates)
        message = template.format(
            amount=amount, merchant=merchant, ref_id=ref_id
        )
        transactions.append((message, 1))  # 1 = transactional
    
    # Generate UPI collect requests
    collect_templates = [
        "UPI collect request of Rs.{amount} from {sender} ({upi_id}) received. Ref: {ref_id}",
        "Rs.{amount} collect request from {sender} via UPI. {upi_id}. Ref: {ref_id}",
        "UPI collect: Rs.{amount} requested by {sender}. {upi_id}. Ref: {ref_id}",
        "Collect request Rs.{amount} from {sender} via UPI. {upi_id}. Ref: {ref_id}",
        "UPI collect payment of Rs.{amount} from {sender} received. {upi_id}. Ref: {ref_id}"
    ]
    
    for _ in range(100):
        sender = random.choice(indian_names)
        amount = random.randint(100, 5000)
        upi_id = random.choice(upi_id_formats).format(
            name=sender, bank=random.choice(bank_codes), number=random.randint(100, 999)
        )
        ref_id = f"TXN{random.randint(100000000, 999999999)}"
        
        template = random.choice(collect_templates)
        message = template.format(
            amount=amount, sender=sender, upi_id=upi_id, ref_id=ref_id
        )
        transactions.append((message, 1))  # 1 = transactional
    
    return transactions


def create_indian_banking_transactions() -> List[Tuple[str, int]]:
    """Create realistic Indian banking transaction messages for top 10 banks"""
    
    # Top 10 Indian banks by market cap/assets
    banks = {
        "SBI": "State Bank of India",
        "HDFC": "HDFC Bank", 
        "ICICI": "ICICI Bank",
        "Axis": "Axis Bank",
        "Kotak": "Kotak Mahindra Bank",
        "PNB": "Punjab National Bank",
        "BOB": "Bank of Baroda",
        "Canara": "Canara Bank",
        "Union": "Union Bank of India",
        "IDBI": "IDBI Bank"
    }
    
    # Transaction types and amounts
    transaction_types = [
        "NEFT", "RTGS", "IMPS", "UPI", "ATM", "POS", "Online", "Mobile Banking",
        "Cheque", "Standing Instruction", "ECS", "Direct Debit", "Credit"
    ]
    
    # Common merchant names
    merchants = [
        "Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Ola", "Paytm", "PhonePe",
        "Google Pay", "BHIM", "Netflix", "Hotstar", "Prime Video", "Spotify",
        "IRCTC", "BookMyShow", "MakeMyTrip", "Goibibo", "OYO", "Treebo",
        "Reliance Digital", "Croma", "Vijay Sales", "Big Bazaar", "DMart",
        "Reliance Fresh", "BigBasket", "Grofers", "Blinkit", "Zepto"
    ]
    
    # Transaction message templates
    templates = [
        "Dear Customer, Rs.{amount} debited from A/c {acct} on {date} for {type} to {merchant}. Avl Bal: Rs.{balance}",
        "Rs.{amount} debited from A/c {acct} on {date} for {type} transaction. Avl Bal: Rs.{balance}",
        "Transaction Alert: Rs.{amount} debited from A/c {acct} for {type} to {merchant} on {date}",
        "Your A/c {acct} has been debited with Rs.{amount} for {type} on {date}. Avl Bal: Rs.{balance}",
        "Rs.{amount} debited from A/c {acct} for {type} transaction to {merchant} on {date}",
        "Transaction: Rs.{amount} debited from A/c {acct} for {type} on {date}. Balance: Rs.{balance}",
        "Your {bank} A/c {acct} debited Rs.{amount} for {type} to {merchant} on {date}",
        "Rs.{amount} debited from {bank} A/c {acct} for {type} transaction on {date}",
        "Transaction Alert: {bank} A/c {acct} debited Rs.{amount} for {type} to {merchant}",
        "Your {bank} account {acct} has been debited Rs.{amount} for {type} on {date}"
    ]
    
    # Credit transaction templates
    credit_templates = [
        "Dear Customer, Rs.{amount} credited to A/c {acct} on {date} for {type}. Avl Bal: Rs.{balance}",
        "Rs.{amount} credited to A/c {acct} on {date} for {type} transaction. Avl Bal: Rs.{balance}",
        "Transaction Alert: Rs.{amount} credited to A/c {acct} for {type} on {date}",
        "Your A/c {acct} has been credited with Rs.{amount} for {type} on {date}. Avl Bal: Rs.{balance}",
        "Rs.{amount} credited to A/c {acct} for {type} transaction on {date}",
        "Transaction: Rs.{amount} credited to A/c {acct} for {type} on {date}. Balance: Rs.{balance}",
        "Your {bank} A/c {acct} credited Rs.{amount} for {type} on {date}",
        "Rs.{amount} credited to {bank} A/c {acct} for {type} transaction on {date}",
        "Transaction Alert: {bank} A/c {acct} credited Rs.{amount} for {type}",
        "Your {bank} account {acct} has been credited Rs.{amount} for {type} on {date}"
    ]
    
    transactions = []
    
    # Generate debit transactions
    for _ in range(800):
        bank = random.choice(list(banks.keys()))
        bank_name = banks[bank]
        acct = f"XXXX{random.randint(1000, 9999)}"
        amount = random.randint(10, 50000)
        balance = random.randint(amount + 1000, 100000)
        date = f"{random.randint(1, 28)}/{random.randint(1, 12)}/2024"
        txn_type = random.choice(transaction_types)
        merchant = random.choice(merchants) if txn_type in ["UPI", "POS", "Online"] else "Transfer"
        
        template = random.choice(templates)
        message = template.format(
            amount=amount, acct=acct, date=date, type=txn_type, 
            merchant=merchant, balance=balance, bank=bank_name
        )
        transactions.append((message, 1))  # 1 = transactional
    
    # Generate credit transactions
    for _ in range(400):
        bank = random.choice(list(banks.keys()))
        bank_name = banks[bank]
        acct = f"XXXX{random.randint(1000, 9999)}"
        amount = random.randint(1000, 100000)
        balance = random.randint(amount + 1000, 200000)
        date = f"{random.randint(1, 28)}/{random.randint(1, 12)}/2024"
        txn_type = random.choice(["Salary", "Refund", "Interest", "Dividend", "Transfer"])
        
        template = random.choice(credit_templates)
        message = template.format(
            amount=amount, acct=acct, date=date, type=txn_type, 
            balance=balance, bank=bank_name
        )
        transactions.append((message, 1))  # 1 = transactional
    
    return transactions


def create_synthetic_negatives() -> List[Tuple[str, int]]:
    """Create hard negative examples for better training"""
    
    # OTP messages
    otp_templates = [
        "Your OTP for {service} is {otp}. Valid for 10 minutes. Do not share with anyone.",
        "OTP: {otp} for {service}. Valid till {time}. Do not share this OTP.",
        "Your {service} verification code is {otp}. Valid for 10 minutes.",
        "OTP for {service} login: {otp}. Do not share this code with anyone.",
        "Verification code: {otp} for {service}. Valid till {time}.",
        "Your {service} OTP is {otp}. Valid for 10 minutes. Do not share.",
        "Login OTP for {service}: {otp}. Valid till {time}.",
        "Your verification code is {otp} for {service}. Do not share.",
        "OTP: {otp} for {service} verification. Valid for 10 minutes.",
        "Your {service} login code: {otp}. Valid till {time}."
    ]
    
    # Promotional messages
    promo_templates = [
        "Get {discount}% off on {product} at {store}! Use code {code}. Limited time offer.",
        "Special offer: {discount}% discount on {product}. Use code {code}. Shop now!",
        "Flash sale! {discount}% off on {product} at {store}. Code: {code}. Hurry!",
        "Limited time: {discount}% off on {product}. Use {code} at checkout.",
        "Exclusive offer: {discount}% discount on {product}. Code: {code}. Shop now!",
        "Get {discount}% off on {product} at {store}. Use code {code}. Limited stock!",
        "Special deal: {discount}% off on {product}. Code: {code}. Don't miss out!",
        "Flash sale alert! {discount}% off on {product}. Use {code}. Limited time!",
        "Exclusive: {discount}% discount on {product} at {store}. Code: {code}.",
        "Limited offer: {discount}% off on {product}. Use code {code}. Shop now!"
    ]
    
    # Balance check messages
    balance_templates = [
        "Your {bank} account balance is Rs.{balance} as on {date}.",
        "Balance in A/c {acct} is Rs.{balance} as on {date}.",
        "Your {bank} A/c {acct} balance: Rs.{balance} as on {date}.",
        "Account balance: Rs.{balance} in A/c {acct} as on {date}.",
        "Your balance in {bank} A/c {acct} is Rs.{balance} as on {date}.",
        "Balance alert: Rs.{balance} in A/c {acct} as on {date}.",
        "Your {bank} account balance: Rs.{balance} as on {date}.",
        "A/c {acct} balance: Rs.{balance} as on {date}.",
        "Balance in your {bank} A/c {acct}: Rs.{balance} as on {date}.",
        "Your account balance is Rs.{balance} as on {date}."
    ]
    
    # Statement messages
    statement_templates = [
        "Your {bank} account statement for {month} {year} is ready. Download from {link}.",
        "Account statement for A/c {acct} for {month} {year} is available. {link}",
        "Your {bank} statement for {month} {year} is ready. Access at {link}.",
        "Statement for A/c {acct} for {month} {year} is available. {link}",
        "Your {bank} account statement for {month} {year} is ready. {link}",
        "Account statement for {month} {year} is available. A/c: {acct}. {link}",
        "Your {bank} statement for {month} {year} is ready. Download: {link}.",
        "Statement for A/c {acct} for {month} {year} is available. {link}",
        "Your {bank} account statement for {month} {year} is ready. {link}",
        "Account statement for {month} {year} is available. {link}"
    ]
    
    negatives = []
    
    # Generate OTP messages
    services = ["Net Banking", "Mobile Banking", "UPI", "Card Payment", "Login", "Verification"]
    for _ in range(300):
        service = random.choice(services)
        otp = f"{random.randint(100000, 999999)}"
        time = f"{random.randint(10, 23)}:{random.randint(10, 59)}"
        template = random.choice(otp_templates)
        message = template.format(service=service, otp=otp, time=time)
        negatives.append((message, 0))  # 0 = not transactional
    
    # Generate promotional messages
    products = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books", "Food", "Travel"]
    stores = ["Amazon", "Flipkart", "Myntra", "Nykaa", "Croma", "Reliance Digital", "BigBazaar"]
    for _ in range(400):
        product = random.choice(products)
        store = random.choice(stores)
        discount = random.randint(10, 70)
        code = f"SAVE{random.randint(10, 99)}"
        template = random.choice(promo_templates)
        message = template.format(product=product, store=store, discount=discount, code=code)
        negatives.append((message, 0))  # 0 = not transactional
    
    # Generate balance check messages
    banks = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB"]
    for _ in range(200):
        bank = random.choice(banks)
        acct = f"XXXX{random.randint(1000, 9999)}"
        balance = random.randint(1000, 100000)
        date = f"{random.randint(1, 28)}/{random.randint(1, 12)}/2024"
        template = random.choice(balance_templates)
        message = template.format(bank=bank, acct=acct, balance=balance, date=date)
        negatives.append((message, 0))  # 0 = not transactional
    
    # Generate statement messages
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for _ in range(100):
        bank = random.choice(banks)
        acct = f"XXXX{random.randint(1000, 9999)}"
        month = random.choice(months)
        year = random.choice([2023, 2024])
        link = "https://netbanking.bank.com"
        template = random.choice(statement_templates)
        message = template.format(bank=bank, acct=acct, month=month, year=year, link=link)
        negatives.append((message, 0))  # 0 = not transactional
    
    return negatives


def load_and_prepare_datasets() -> Tuple[List[str], List[int]]:
    """Load and combine all datasets"""
    
    print("Loading and preparing datasets...")
    
    # Load original dataset
    df_original = pd.read_csv("neatsmsdata.csv")
    original_data = []
    for _, row in df_original.iterrows():
        text = str(row['body']).strip()
        label = 1 if str(row['label']).strip().lower() == 'transactional' else 0
        if text and len(text) > 10:  # Filter out very short messages
            original_data.append((text, label))
    
    print(f"Loaded {len(original_data)} examples from original dataset")
    
    # Load UPI transactions dataset from CSV
    df_upi = pd.read_csv("transactions.csv")
    upi_csv_data = []
    for _, row in df_upi.iterrows():
        # Create realistic UPI transaction messages from CSV data
        sender = row['Sender Name']
        receiver = row['Receiver Name']
        amount = row['Amount (INR)']
        status = row['Status']
        
        # Create multiple message formats for each transaction
        messages = [
            f"UPI transaction of Rs.{amount} from {sender} to {receiver}. Status: {status}",
            f"Rs.{amount} transferred via UPI from {sender} to {receiver}. {status}",
            f"UPI payment of Rs.{amount} to {receiver} from {sender}. Status: {status}",
            f"Transaction: Rs.{amount} UPI transfer from {sender} to {receiver}. {status}",
            f"UPI: Rs.{amount} sent to {receiver} from {sender}. Status: {status}"
        ]
        
        for msg in messages:
            upi_csv_data.append((msg, 1))  # All UPI transactions are transactional
    
    print(f"Created {len(upi_csv_data)} UPI transaction messages from CSV")
    
    # Load banks.csv dataset
    df_banks = pd.read_csv("banks.csv")
    banks_data = []
    for _, row in df_banks.iterrows():
        text = str(row['body']).strip()
        if text and len(text) > 10:
            banks_data.append((text, 1))  # All bank messages are transactional
    
    print(f"Loaded {len(banks_data)} examples from banks dataset")
    
    # Load upi.csv dataset
    df_upi_detailed = pd.read_csv("upi.csv")
    upi_detailed_data = []
    for _, row in df_upi_detailed.iterrows():
        text = str(row['body']).strip()
        if text and len(text) > 10:
            upi_detailed_data.append((text, 1))  # All UPI messages are transactional
    
    print(f"Loaded {len(upi_detailed_data)} examples from detailed UPI dataset")
    
    # Generate comprehensive UPI transactions
    upi_data = create_comprehensive_upi_transactions()
    print(f"Generated {len(upi_data)} comprehensive UPI transaction messages")
    
    # Combine CSV and generated UPI data
    upi_data = upi_csv_data + upi_data
    
    # Generate Indian banking transactions
    banking_data = create_indian_banking_transactions()
    print(f"Generated {len(banking_data)} Indian banking transaction messages")
    
    # Generate synthetic negatives
    negative_data = create_synthetic_negatives()
    print(f"Generated {len(negative_data)} synthetic negative examples")
    
    # Combine all datasets
    all_data = original_data + upi_data + banks_data + upi_detailed_data + banking_data + negative_data
    random.shuffle(all_data)
    
    texts, labels = zip(*all_data)
    
    print(f"Total dataset size: {len(texts)} examples")
    print(f"Transactional: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Non-transactional: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    return list(texts), list(labels)


def train_sentencepiece(texts: List[str], vocab_size: int = 8000, model_prefix: str = "sms_tokenizer") -> str:
    """Train SentencePiece tokenizer"""
    
    print(f"Training SentencePiece tokenizer with vocab_size={vocab_size}...")
    
    # Write texts to temporary file
    temp_file = "temp_texts.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Train SentencePiece
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram",
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>"
    )
    
    # Clean up temp file
    os.remove(temp_file)
    
    model_path = f"{model_prefix}.model"
    print(f"Tokenizer saved to {model_path}")
    return model_path


def build_model(vocab_size: int = 8000, sequence_length: int = 200, embedding_dim: int = 64) -> tf.keras.Model:
    """Build CNN-based model for pure TFLite compatibility"""
    
    # Input layer for token IDs
    text_input = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
    
    # Embedding layer
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(text_input)
    
    # CNN layers for feature extraction
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout1")(x)
    
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same', name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout2")(x)
    
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout3")(x)
    
    # Global max pooling
    x = tf.keras.layers.GlobalMaxPooling1D(name="global_pool")(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu', name="dense1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout4")(x)
    
    x = tf.keras.layers.Dense(64, activation='relu', name="dense2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn5")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout5")(x)
    
    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)
    
    model = tf.keras.Model(inputs=text_input, outputs=output, name="sms_classifier")
    return model


def preprocess_text(text: str, sp_model: spm.SentencePieceProcessor, max_length: int = 200) -> List[int]:
    """Preprocess text using SentencePiece tokenizer"""
    # Standardize text
    text = text.lower()
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)
    
    # Tokenize
    tokens = sp_model.encode_as_ids(text)
    
    # Pad or truncate
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))  # PAD_ID = 0
    
    return tokens


def convert_to_tflite_int8(saved_model_dir: str, tflite_path: str) -> None:
    """Convert SavedModel to INT8 quantized TFLite"""
    
    print(f"Converting to INT8 quantized TFLite...")
    
    # Load the SavedModel
    model = tf.saved_model.load(saved_model_dir)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Set optimization flags for dynamic range quantization (INT8)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Ensure pure TFLite ops only
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"INT8 quantized TFLite model saved to {tflite_path}")


def main():
    parser = argparse.ArgumentParser(description="Train enhanced SMS classifier with SentencePiece and INT8 quantization")
    parser.add_argument("--output_dir", default="artifacts_enhanced", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=8000, help="SentencePiece vocabulary size")
    parser.add_argument("--sequence_length", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare datasets
    texts, labels = load_and_prepare_datasets()
    
    # Train SentencePiece tokenizer
    tokenizer_path = train_sentencepiece(texts, args.vocab_size)
    
    # Load tokenizer
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(tokenizer_path)
    
    # Preprocess all texts
    print("Preprocessing texts...")
    tokenized_texts = [preprocess_text(text, sp_model, args.sequence_length) for text in texts]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        tokenized_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"Training set: {len(X_train)} examples")
    print(f"Test set: {len(X_test)} examples")
    
    # Build model
    model = build_model(
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim
    )
    
    # Compile model with multi-task losses
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            'classification': 'binary_crossentropy',
            'merchant_ner': 'sparse_categorical_crossentropy',
            'amount_ner': 'sparse_categorical_crossentropy', 
            'type_ner': 'sparse_categorical_crossentropy',
            'direction': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'classification': 1.0,
            'merchant_ner': 0.5,
            'amount_ner': 0.5,
            'type_ner': 0.5,
            'direction': 0.5
        },
        metrics={
            'classification': ['accuracy', 'precision', 'recall'],
            'merchant_ner': ['accuracy'],
            'amount_ner': ['accuracy'],
            'type_ner': ['accuracy'],
            'direction': ['accuracy']
        }
    )
    
    print("Model summary:")
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
        ]
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['not_transactional', 'transactional']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate additional metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Save model artifacts
    saved_model_dir = os.path.join(args.output_dir, "saved_model")
    tflite_path = os.path.join(args.output_dir, "sms_classifier.tflite")
    tokenizer_dest = os.path.join(args.output_dir, "tokenizer.spm")
    labels_path = os.path.join(args.output_dir, "labels.json")
    
    # Save SavedModel
    print(f"Saving SavedModel to {saved_model_dir}...")
    model.export(saved_model_dir)
    
    # Save tokenizer
    import shutil
    shutil.copy(tokenizer_path, tokenizer_dest)
    
    # Save labels
    labels_dict = {"0": "not_transactional", "1": "transactional"}
    with open(labels_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    
    # Convert to INT8 quantized TFLite
    convert_to_tflite_int8(saved_model_dir, tflite_path)
    
    # Test inference
    print("\nTesting inference...")
    test_texts = [
        "Rs.5000 debited from A/c XXXX1234 for UPI transaction to Amazon on 15/12/2024",
        "Your OTP for Net Banking is 123456. Valid for 10 minutes.",
        "Get 50% off on Electronics at Amazon! Use code SAVE50. Limited time offer.",
        "Dear Customer, Rs.2500 credited to A/c XXXX5678 for Salary on 01/12/2024"
    ]
    
    for text in test_texts:
        tokens = preprocess_text(text, sp_model, args.sequence_length)
        prediction = model.predict(np.array([tokens]), verbose=0)[0][0]
        label = "transactional" if prediction > 0.5 else "not_transactional"
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {prediction:.4f} -> {label}\n")
    
    print("Training completed successfully!")
    print(f"Model artifacts saved to: {args.output_dir}")
    print(f"- TFLite model: {tflite_path}")
    print(f"- Tokenizer: {tokenizer_dest}")
    print(f"- Labels: {labels_path}")
    print(f"- SavedModel: {saved_model_dir}")


if __name__ == "__main__":
    main()
