import pandas as pd
import numpy as np

def load_data(data_dir: str):
    distributors = pd.read_csv(f"{data_dir}/assignment_distributors.csv")
    orders = pd.read_csv(f"{data_dir}/assignment_orders.csv")
    payments = pd.read_csv(f"{data_dir}/assignment_payments.csv")
    return distributors, orders, payments

def clean_dates_and_amounts(df_orders, df_payments, df_distributors):
    if not df_orders.empty:
        df_orders['order_date'] = pd.to_datetime(df_orders['order_date'], errors="coerce")
        df_orders['order_value_inr'] = pd.to_numeric(df_orders['order_value_inr'], errors="coerce").fillna(0)
    
    if not df_payments.empty:
        df_payments['invoice_date'] = pd.to_datetime(df_payments['invoice_date'], errors="coerce")
        df_payments['due_date'] = pd.to_datetime(df_payments['due_date'], errors="coerce")
        df_payments['payment_date'] = pd.to_datetime(df_payments['payment_date'], errors="coerce")
        
        df_payments["invoice_amount_inr"] = pd.to_numeric(df_payments["invoice_amount_inr"], errors="coerce").fillna(0)
        df_payments["amount_paid_inr"] = pd.to_numeric(df_payments["amount_paid_inr"], errors="coerce").fillna(0)

        df_payments.loc[df_payments["amount_paid_inr"] > df_payments["invoice_amount_inr"], "amount_paid_inr"] = df_payments["invoice_amount_inr"]
        df_payments['unpaid_amount'] = df_payments['invoice_amount_inr'] - df_payments['amount_paid_inr']
        
        # Calculate delay_days globally just like the notebook
        snapshot_date = df_payments[["due_date", "payment_date"]].max().max() if not df_payments.empty else pd.Timestamp.now()
        df_payments["delay_days"] = (df_payments["payment_date"].fillna(snapshot_date) - df_payments["due_date"]).dt.days
        
    if not df_distributors.empty and 'onboarded_date' in df_distributors.columns:
        df_distributors['onboarded_date'] = pd.to_datetime(df_distributors['onboarded_date'], errors="coerce")
        
    return df_orders, df_payments, df_distributors

def create_labels(df_payments: pd.DataFrame, t_cutoff: pd.Timestamp):
    DAYS_DUE = 20
    UNPAID_DUE_PERCENT = 0.15
    
    perf_payments = df_payments[df_payments['invoice_date'] >= t_cutoff].copy()
    
    if perf_payments.empty:
        return pd.DataFrame(columns=["distributor_id", "high_risk"])
        
    def calculate_target(group):
        total_invoiced = group['invoice_amount_inr'].sum()
        total_unpaid = group['unpaid_amount'].sum()
        unpaid_ratio = total_unpaid / (total_invoiced + 1e-6)
        
        has_overdue = (group['payment_status'] == 'overdue').any()
        bad_debt = (group['payment_status'] == 'written_off').any()
        avg_delay = group['delay_days'].mean()
        
        high_risk = int(has_overdue or bad_debt or (unpaid_ratio > UNPAID_DUE_PERCENT) or (avg_delay > DAYS_DUE))
        return pd.Series({'high_risk': high_risk})
        
    target_df = perf_payments.groupby('distributor_id').apply(calculate_target, include_groups=False).reset_index()
    return target_df

def create_features(df_distributors, df_orders, df_payments, t_cutoff: pd.Timestamp):
    """
    Creates features using events before t_cutoff.
    """
    obs_orders = df_orders[df_orders['order_date'] < t_cutoff].copy() if not df_orders.empty else df_orders.copy()
    obs_payments = df_payments[df_payments['invoice_date'] < t_cutoff].copy() if not df_payments.empty else df_payments.copy()
    
    if not obs_orders.empty:
        order_feats = obs_orders.groupby('distributor_id').agg(
            total_orders=('order_id', 'count'),
            total_order_value=('order_value_inr', 'sum'),
            avg_order_value=('order_value_inr', 'mean'),
            cancelled_orders=('order_status', lambda x: (x == 'cancelled').sum()),
            delivered_orders=('order_status', lambda x: (x == 'delivered').sum()),
        ).reset_index()
        
        order_feats['cancellation_rate'] = order_feats['cancelled_orders'] / (order_feats['total_orders'] + 1e-9)
    else:
        order_feats = pd.DataFrame(columns=['distributor_id', 'total_orders', 'total_order_value', 'avg_order_value', 'cancelled_orders', 'delivered_orders', 'cancellation_rate'])
        
    if not obs_payments.empty:
        payment_feats = obs_payments.groupby('distributor_id').agg(
            total_invoices=('invoice_id', 'count'),
            total_invoice_amt=('invoice_amount_inr', 'sum'),
            total_paid_amt=('amount_paid_inr', 'sum'),
            total_unpaid_amt=('unpaid_amount', 'sum'),
            avg_delay_days=('delay_days', 'mean'),
            max_delay_days=('delay_days', 'max'),
            on_time_payment_count=('delay_days', lambda x: (x <= 0).sum()),
            overdue_count=('payment_status', lambda x: (x == 'overdue').sum()),
            partial_count=('payment_status', lambda x: (x == 'partial').sum())
        ).reset_index()
        
        payment_feats['on_time_ratio'] = payment_feats['on_time_payment_count'] / (payment_feats['total_invoices'] + 1e-9)
        payment_feats['overdue_ratio'] = payment_feats['overdue_count'] / (payment_feats['total_invoices'] + 1e-9)
    else:
        payment_feats = pd.DataFrame(columns=['distributor_id', 'total_invoices', 'total_invoice_amt', 'total_paid_amt', 'total_unpaid_amt', 'avg_delay_days', 'max_delay_days', 'on_time_payment_count', 'overdue_count', 'partial_count', 'on_time_ratio', 'overdue_ratio'])
        
    df_master = df_distributors.copy()
    if not payment_feats.empty:
        df_master = df_master.merge(payment_feats, on='distributor_id', how='left')
    else:
        for c in payment_feats.columns:
            if c != 'distributor_id': df_master[c] = 0.0
            
    if not order_feats.empty:
        df_master = df_master.merge(order_feats, on='distributor_id', how='left')
    else:
        for c in order_feats.columns:
            if c != 'distributor_id': df_master[c] = 0.0
            
    df_master['credit_limit_inr'] = pd.to_numeric(df_master['credit_limit_inr'], errors='coerce').fillna(0)
    
    history_num_cols = [
        "total_invoices", "total_invoice_amt", "total_paid_amt", "total_unpaid_amt",
        "avg_delay_days", "max_delay_days", "on_time_payment_count", "overdue_count",
        "partial_count", "on_time_ratio", "overdue_ratio",
        "total_orders", "total_order_value", "avg_order_value",
        "cancelled_orders", "delivered_orders", "cancellation_rate"
    ]
    
    for c in history_num_cols:
        if c in df_master.columns:
            df_master[c] = df_master[c].fillna(0)
        else:
            df_master[c] = 0.0
            
    df_master['credit_utilization'] = (df_master['total_unpaid_amt'] / df_master['credit_limit_inr'].replace(0, 1e-6))
    
    numeric_cols = [
        "credit_limit_inr", "credit_terms_days", "credit_utilization",
        "total_invoices", "avg_delay_days", "max_delay_days", "on_time_ratio", "overdue_ratio",
        "total_orders", "avg_order_value", "cancellation_rate",
    ]
    categorical_cols = ['region', 'channel']
    
    for c in numeric_cols:
        if c in df_master.columns: df_master[c] = df_master[c].fillna(0)
        else: df_master[c] = 0
            
    for c in categorical_cols:
        if c in df_master.columns: df_master[c] = df_master[c].fillna("Unknown")
        else: df_master[c] = "Unknown"
            
    return df_master[numeric_cols + categorical_cols + ['distributor_id']]

def build_training_data(data_dir: str):
    df_distributors, df_orders, df_payments = load_data(data_dir)
    df_orders, df_payments, df_distributors = clean_dates_and_amounts(df_orders, df_payments, df_distributors)
    
    t_cutoff = pd.Timestamp("2025-10-01")
    
    target_df = create_labels(df_payments, t_cutoff)
    
    features_df = create_features(df_distributors, df_orders, df_payments, t_cutoff)
    
    df_master = features_df.merge(target_df, on='distributor_id', how='inner')
    
    numeric_cols = [
        "credit_limit_inr", "credit_terms_days", "credit_utilization",
        "total_invoices", "avg_delay_days", "max_delay_days", "on_time_ratio", "overdue_ratio",
        "total_orders", "avg_order_value", "cancellation_rate",
    ]
    categorical_cols = ['region', 'channel']
    
    X = df_master[numeric_cols + categorical_cols]
    y = df_master["high_risk"]
    
    return X, y, df_master, t_cutoff

def process_single_distributor(distributor_info: dict, order_history: list = None, payment_history: list = None, t_cutoff=None):
    if t_cutoff is None:
        t_cutoff = pd.Timestamp.now()
    else:
        t_cutoff = pd.to_datetime(t_cutoff)
        
    df_dist = pd.DataFrame([distributor_info])
    df_ord = pd.DataFrame(order_history) if order_history else pd.DataFrame()
    df_pay = pd.DataFrame(payment_history) if payment_history else pd.DataFrame()
    
    df_ord, df_pay, df_dist = clean_dates_and_amounts(df_ord, df_pay, df_dist)
    
    dist_id = df_dist["distributor_id"].iloc[0]
    if not df_ord.empty and "distributor_id" not in df_ord.columns:
        df_ord["distributor_id"] = dist_id
    if not df_pay.empty and "distributor_id" not in df_pay.columns:
        df_pay["distributor_id"] = dist_id
        
    features_df = create_features(df_dist, df_ord, df_pay, t_cutoff)
    
    numeric_cols = [
        "credit_limit_inr", "credit_terms_days", "credit_utilization",
        "total_invoices", "avg_delay_days", "max_delay_days", "on_time_ratio", "overdue_ratio",
        "total_orders", "avg_order_value", "cancellation_rate",
    ]
    categorical_cols = ['region', 'channel']
    
    return features_df[numeric_cols + categorical_cols]