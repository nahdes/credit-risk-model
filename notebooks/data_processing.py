"""
Feature Engineering Pipeline for Credit Scoring Model
Bati Bank - Buy-Now-Pay-Later Service

This module implements a robust, automated, and reproducible data processing
pipeline that transforms raw transaction data into model-ready features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')


# ============================================================================
# Custom Transformers
# ============================================================================

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data to customer-level features.
    Creates RFMS (Recency, Frequency, Monetary, Standard Deviation) features.
    """
    
    def __init__(self, customer_id_col='CustomerId', 
                 amount_col='Amount',
                 date_col='TransactionStartTime'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.date_col = date_col
        self.reference_date = None
        
    def fit(self, X, y=None):
        """Learn the reference date for recency calculation."""
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        self.reference_date = X[self.date_col].max()
        return self
    
    def transform(self, X):
        """Aggregate transactions to customer level with RFMS features."""
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Aggregate features by customer
        agg_features = X.groupby(self.customer_id_col).agg({
            # Recency: Days since last transaction
            self.date_col: lambda x: (self.reference_date - x.max()).days,
            
            # Monetary: Transaction amounts
            self.amount_col: [
                'sum',      # Total transaction amount
                'mean',     # Average transaction amount
                'median',   # Median transaction amount
                'min',      # Minimum transaction amount
                'max',      # Maximum transaction amount
                'std',      # Standard deviation (volatility)
                lambda x: x.quantile(0.25),  # Q1
                lambda x: x.quantile(0.75),  # Q3
            ],
            
            # Additional columns for counting
            'TransactionId': 'count'  # Frequency: Transaction count
        }).reset_index()
        
        # Flatten multi-level columns
        agg_features.columns = [
            self.customer_id_col,
            'Recency',
            'Total_Transaction_Amount',
            'Average_Transaction_Amount',
            'Median_Transaction_Amount',
            'Min_Transaction_Amount',
            'Max_Transaction_Amount',
            'Std_Transaction_Amount',
            'Q1_Transaction_Amount',
            'Q3_Transaction_Amount',
            'Transaction_Count'
        ]
        
        # Additional derived features
        agg_features['Transaction_Range'] = (
            agg_features['Max_Transaction_Amount'] - 
            agg_features['Min_Transaction_Amount']
        )
        
        # Coefficient of Variation (CV) - relative volatility
        agg_features['CV_Transaction_Amount'] = (
            agg_features['Std_Transaction_Amount'] / 
            (agg_features['Average_Transaction_Amount'] + 1e-5)
        )
        
        # IQR
        agg_features['IQR_Transaction_Amount'] = (
            agg_features['Q3_Transaction_Amount'] - 
            agg_features['Q1_Transaction_Amount']
        )
        
        # Average transaction frequency (transactions per day)
        X_temp = X.copy()
        customer_tenure = X_temp.groupby(self.customer_id_col)[self.date_col].agg(
            lambda x: (x.max() - x.min()).days + 1
        ).reset_index()
        customer_tenure.columns = [self.customer_id_col, 'Customer_Tenure_Days']
        
        agg_features = agg_features.merge(customer_tenure, on=self.customer_id_col)
        agg_features['Avg_Transactions_Per_Day'] = (
            agg_features['Transaction_Count'] / 
            (agg_features['Customer_Tenure_Days'] + 1)
        )
        
        return agg_features


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from transaction timestamps.
    """
    
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract time-based features."""
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Extract temporal components
        X['Transaction_Hour'] = X[self.date_col].dt.hour
        X['Transaction_Day'] = X[self.date_col].dt.day
        X['Transaction_Month'] = X[self.date_col].dt.month
        X['Transaction_Year'] = X[self.date_col].dt.year
        X['Transaction_DayOfWeek'] = X[self.date_col].dt.dayofweek
        X['Transaction_Quarter'] = X[self.date_col].dt.quarter
        X['Transaction_WeekOfYear'] = X[self.date_col].dt.isocalendar().week
        
        # Derived temporal features
        X['Is_Weekend'] = X['Transaction_DayOfWeek'].isin([5, 6]).astype(int)
        X['Is_MonthStart'] = X[self.date_col].dt.is_month_start.astype(int)
        X['Is_MonthEnd'] = X[self.date_col].dt.is_month_end.astype(int)
        
        # Time of day categories
        X['Time_Of_Day'] = pd.cut(
            X['Transaction_Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        return X


class AdditionalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates additional behavioral and categorical features.
    """
    
    def __init__(self, customer_id_col='CustomerId'):
        self.customer_id_col = customer_id_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Engineer additional features at customer level."""
        X = X.copy()
        
        # Product diversity
        if 'ProductCategory' in X.columns:
            product_diversity = X.groupby(self.customer_id_col)['ProductCategory'].agg([
                ('Unique_Products', 'nunique'),
                ('Most_Common_Product', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
            ]).reset_index()
            X = X.merge(product_diversity, on=self.customer_id_col, how='left')
        
        # Channel behavior
        if 'ChannelId' in X.columns:
            channel_behavior = X.groupby(self.customer_id_col)['ChannelId'].agg([
                ('Unique_Channels', 'nunique'),
                ('Primary_Channel', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
            ]).reset_index()
            X = X.merge(channel_behavior, on=self.customer_id_col, how='left')
            X['Is_MultiChannel_User'] = (X['Unique_Channels'] > 1).astype(int)
        
        # Fraud indicators
        if 'FraudResult' in X.columns:
            fraud_stats = X.groupby(self.customer_id_col)['FraudResult'].agg([
                ('Fraud_Count', lambda x: (x == 1).sum() if pd.api.types.is_numeric_dtype(x) else 0),
                ('Total_Transactions_Fraud', 'count')
            ]).reset_index()
            fraud_stats['Fraud_Rate'] = (
                fraud_stats['Fraud_Count'] / fraud_stats['Total_Transactions_Fraud']
            )
            X = X.merge(fraud_stats[[self.customer_id_col, 'Fraud_Count', 'Fraud_Rate']], 
                       on=self.customer_id_col, how='left')
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables using specified strategy.
    """
    
    def __init__(self, strategy='onehot', top_n=10):
        """
        Parameters:
        -----------
        strategy : str
            'onehot' for one-hot encoding, 'label' for label encoding
        top_n : int
            Keep only top N categories, group others as 'Other'
        """
        self.strategy = strategy
        self.top_n = top_n
        self.encoders = {}
        self.top_categories = {}
        
    def fit(self, X, y=None):
        """Learn encoding mappings."""
        X = X.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Get top N categories
            top_cats = X[col].value_counts().head(self.top_n).index.tolist()
            self.top_categories[col] = top_cats
            
            if self.strategy == 'label':
                encoder = LabelEncoder()
                # Fit on top categories + 'Other'
                encoder.fit(top_cats + ['Other'])
                self.encoders[col] = encoder
                
        return self
    
    def transform(self, X):
        """Apply encoding."""
        X = X.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in self.top_categories:
                # Replace rare categories with 'Other'
                X[col] = X[col].apply(
                    lambda x: x if x in self.top_categories[col] else 'Other'
                )
                
                if self.strategy == 'label':
                    X[col] = self.encoders[col].transform(X[col])
                elif self.strategy == 'onehot':
                    # One-hot encode
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                    
        return X


class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) encoder for categorical and numerical features.
    
    WoE measures the strength of relationship between a feature and target:
    WoE = ln(% of Good / % of Bad)
    
    Information Value (IV) measures predictive power:
    IV = Σ (% of Good - % of Bad) * WoE
    """
    
    def __init__(self, categorical_cols=None, numerical_cols=None, n_bins=10):
        """
        Parameters:
        -----------
        categorical_cols : list
            Categorical columns to encode
        numerical_cols : list
            Numerical columns to bin and encode
        n_bins : int
            Number of bins for numerical features
        """
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.n_bins = n_bins
        self.woe_mappings = {}
        self.iv_values = {}
        self.bin_edges = {}
        
    def fit(self, X, y):
        """Calculate WoE and IV for each feature."""
        X = X.copy()
        
        if y is None:
            raise ValueError("Target variable y is required for WoE encoding")
        
        # Ensure y is binary
        if len(np.unique(y)) != 2:
            raise ValueError("Target must be binary for WoE encoding")
        
        # Encode categorical features
        for col in self.categorical_cols:
            if col in X.columns:
                woe_dict, iv = self._calculate_woe_categorical(X[col], y)
                self.woe_mappings[col] = woe_dict
                self.iv_values[col] = iv
        
        # Encode numerical features (after binning)
        for col in self.numerical_cols:
            if col in X.columns:
                woe_dict, iv, bins = self._calculate_woe_numerical(X[col], y)
                self.woe_mappings[col] = woe_dict
                self.iv_values[col] = iv
                self.bin_edges[col] = bins
                
        return self
    
    def transform(self, X):
        """Apply WoE transformation."""
        X = X.copy()
        
        # Transform categorical features
        for col in self.categorical_cols:
            if col in X.columns and col in self.woe_mappings:
                X[f'{col}_WoE'] = X[col].map(self.woe_mappings[col])
                # Fill missing mappings with 0 (neutral)
                X[f'{col}_WoE'].fillna(0, inplace=True)
        
        # Transform numerical features
        for col in self.numerical_cols:
            if col in X.columns and col in self.woe_mappings:
                # Bin the numerical feature
                X[f'{col}_Binned'] = pd.cut(
                    X[col], 
                    bins=self.bin_edges[col], 
                    include_lowest=True,
                    duplicates='drop'
                )
                # Map to WoE
                X[f'{col}_WoE'] = X[f'{col}_Binned'].astype(str).map(
                    self.woe_mappings[col]
                )
                X[f'{col}_WoE'].fillna(0, inplace=True)
                X.drop(f'{col}_Binned', axis=1, inplace=True)
                
        return X
    
    def _calculate_woe_categorical(self, feature, target):
        """Calculate WoE and IV for categorical feature."""
        df = pd.DataFrame({'feature': feature, 'target': target})
        
        # Calculate distributions
        grouped = df.groupby('feature')['target'].agg(['sum', 'count'])
        grouped.columns = ['n_bad', 'n_total']
        grouped['n_good'] = grouped['n_total'] - grouped['n_bad']
        
        # Total goods and bads
        total_good = grouped['n_good'].sum()
        total_bad = grouped['n_bad'].sum()
        
        # Calculate WoE
        grouped['pct_good'] = grouped['n_good'] / total_good
        grouped['pct_bad'] = grouped['n_bad'] / total_bad
        
        # Avoid division by zero
        grouped['pct_good'] = grouped['pct_good'].replace(0, 0.0001)
        grouped['pct_bad'] = grouped['pct_bad'].replace(0, 0.0001)
        
        grouped['WoE'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
        
        # Calculate IV
        grouped['IV'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['WoE']
        iv = grouped['IV'].sum()
        
        woe_dict = grouped['WoE'].to_dict()
        
        return woe_dict, iv
    
    def _calculate_woe_numerical(self, feature, target):
        """Calculate WoE and IV for numerical feature after binning."""
        # Create bins using quantiles
        _, bins = pd.qcut(feature, q=self.n_bins, retbins=True, duplicates='drop')
        
        # Bin the feature
        feature_binned = pd.cut(feature, bins=bins, include_lowest=True, duplicates='drop')
        
        # Calculate WoE for each bin
        df = pd.DataFrame({'feature': feature_binned.astype(str), 'target': target})
        
        woe_dict, iv = self._calculate_woe_categorical(df['feature'], target)
        
        return woe_dict, iv, bins
    
    def get_iv_summary(self):
        """Return Information Value summary."""
        iv_df = pd.DataFrame({
            'Feature': list(self.iv_values.keys()),
            'IV': list(self.iv_values.values())
        }).sort_values('IV', ascending=False)
        
        # Add predictive power interpretation
        def interpret_iv(iv):
            if iv < 0.02:
                return 'Not Predictive'
            elif iv < 0.1:
                return 'Weak'
            elif iv < 0.3:
                return 'Medium'
            elif iv < 0.5:
                return 'Strong'
            else:
                return 'Very Strong'
        
        iv_df['Predictive_Power'] = iv_df['IV'].apply(interpret_iv)
        
        return iv_df


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values using specified strategy.
    """
    
    def __init__(self, strategy='mean', threshold=0.5):
        """
        Parameters:
        -----------
        strategy : str
            'mean', 'median', 'mode', 'knn', or 'drop'
        threshold : float
            For 'drop' strategy: drop columns with > threshold missing
        """
        self.strategy = strategy
        self.threshold = threshold
        self.imputers = {}
        self.columns_to_drop = []
        
    def fit(self, X, y=None):
        """Learn imputation strategy."""
        X = X.copy()
        
        if self.strategy == 'drop':
            # Identify columns to drop
            missing_pct = X.isnull().sum() / len(X)
            self.columns_to_drop = missing_pct[missing_pct > self.threshold].index.tolist()
        else:
            # Fit imputers for numerical and categorical columns
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            if self.strategy == 'knn':
                self.imputers['numerical'] = KNNImputer(n_neighbors=5)
                self.imputers['numerical'].fit(X[numerical_cols])
            else:
                strategy_map = {
                    'mean': 'mean',
                    'median': 'median',
                    'mode': 'most_frequent'
                }
                
                if len(numerical_cols) > 0:
                    self.imputers['numerical'] = SimpleImputer(
                        strategy=strategy_map.get(self.strategy, 'mean')
                    )
                    self.imputers['numerical'].fit(X[numerical_cols])
                
                if len(categorical_cols) > 0:
                    self.imputers['categorical'] = SimpleImputer(
                        strategy='most_frequent'
                    )
                    self.imputers['categorical'].fit(X[categorical_cols])
        
        return self
    
    def transform(self, X):
        """Apply imputation or drop columns."""
        X = X.copy()
        
        if self.strategy == 'drop':
            X = X.drop(columns=self.columns_to_drop, errors='ignore')
        else:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if 'numerical' in self.imputers and len(numerical_cols) > 0:
                X[numerical_cols] = self.imputers['numerical'].transform(X[numerical_cols])
            
            if 'categorical' in self.imputers and len(categorical_cols) > 0:
                X[categorical_cols] = self.imputers['categorical'].transform(X[categorical_cols])
        
        return X


# ============================================================================
# Main Feature Engineering Pipeline
# ============================================================================

def create_feature_pipeline(
    use_woe=True,
    scaling_method='standard',
    encoding_strategy='label',
    imputation_strategy='median'
):
    """
    Create a complete feature engineering pipeline.
    
    Parameters:
    -----------
    use_woe : bool
        Whether to use WoE encoding (requires target variable)
    scaling_method : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    encoding_strategy : str
        'onehot' or 'label' for categorical encoding
    imputation_strategy : str
        'mean', 'median', 'mode', 'knn', or 'drop'
    
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
    """
    
    # Define the pipeline steps
    steps = [
        ('time_features', TimeFeatureExtractor()),
        ('additional_features', AdditionalFeatureEngineer()),
        ('missing_handler', MissingValueHandler(strategy=imputation_strategy)),
        ('categorical_encoder', CategoricalEncoder(strategy=encoding_strategy)),
    ]
    
    # Add scaler
    if scaling_method == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaling_method == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    
    pipeline = Pipeline(steps)
    
    return pipeline


def process_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Complete data processing workflow.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
    target_col : str, optional
        Name of target column for WoE encoding
    test_df : pd.DataFrame, optional
        Test dataset to transform with same pipeline
    
    Returns:
    --------
    train_processed : pd.DataFrame
        Processed training data
    test_processed : pd.DataFrame or None
        Processed test data
    woe_iv_summary : pd.DataFrame or None
        WoE/IV summary if target provided
    """
    
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Step 1: Aggregate to customer level
    print("\n[1/6] Aggregating transactions to customer level...")
    aggregator = CustomerAggregator()
    customer_df = aggregator.fit_transform(df)
    print(f"   Created {len(customer_df)} customer records with {len(customer_df.columns)} features")
    
    # Step 2: Add time features (on original transaction data if needed)
    print("\n[2/6] Extracting temporal features...")
    time_extractor = TimeFeatureExtractor()
    df_with_time = time_extractor.fit_transform(df)
    
    # Aggregate time features to customer level
    time_features = df_with_time.groupby('CustomerId').agg({
        'Transaction_Hour': ['mean', 'std'],
        'Is_Weekend': 'mean',
        'Transaction_DayOfWeek': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
    }).reset_index()
    time_features.columns = ['CustomerId', 'Avg_Transaction_Hour', 'Std_Transaction_Hour',
                             'Weekend_Transaction_Ratio', 'Most_Common_DayOfWeek']
    
    customer_df = customer_df.merge(time_features, on='CustomerId', how='left')
    print(f"   Added {len(time_features.columns)-1} temporal features")
    
    # Step 3: WoE encoding if target provided
    woe_iv_summary = None
    if target_col and target_col in customer_df.columns:
        print("\n[3/6] Calculating WoE and IV...")
        
        # Select features for WoE
        categorical_features = ['Most_Common_DayOfWeek']
        numerical_features = ['Recency', 'Average_Transaction_Amount', 'Transaction_Count',
                             'Std_Transaction_Amount', 'CV_Transaction_Amount']
        
        # Filter to existing columns
        categorical_features = [c for c in categorical_features if c in customer_df.columns]
        numerical_features = [c for c in numerical_features if c in customer_df.columns]
        
        woe_encoder = WoEEncoder(
            categorical_cols=categorical_features,
            numerical_cols=numerical_features,
            n_bins=5
        )
        
        y = customer_df[target_col]
        woe_encoder.fit(customer_df, y)
        customer_df = woe_encoder.transform(customer_df)
        
        woe_iv_summary = woe_encoder.get_iv_summary()
        print("\n   Information Value Summary:")
        print(woe_iv_summary.to_string(index=False))
    else:
        print("\n[3/6] Skipping WoE encoding (no target variable provided)")
    
    # Step 4: Handle missing values
    print("\n[4/6] Handling missing values...")
    missing_handler = MissingValueHandler(strategy='median')
    customer_df = missing_handler.fit_transform(customer_df)
    print(f"   Missing values handled")
    
    # Step 5: Encode categorical variables
    print("\n[5/6] Encoding categorical variables...")
    cat_encoder = CategoricalEncoder(strategy='label')
    customer_df = cat_encoder.fit_transform(customer_df)
    print(f"   Categorical encoding complete")
    
    # Step 6: Scale numerical features
    print("\n[6/6] Scaling numerical features...")
    numerical_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID and target columns from scaling
    exclude_cols = ['CustomerId', target_col] if target_col else ['CustomerId']
    numerical_cols = [c for c in numerical_cols if c not in exclude_cols]
    
    scaler = StandardScaler()
    customer_df[numerical_cols] = scaler.fit_transform(customer_df[numerical_cols])
    print(f"   Scaled {len(numerical_cols)} numerical features")
    
    # Process test data if provided
    test_processed = None
    if test_df is not None:
        print("\n" + "-"*80)
        print("Processing test dataset...")
        # Apply same transformations (fit already done on train)
        test_customer_df = aggregator.transform(test_df)
        # Continue with other transformations...
        test_processed = test_customer_df
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nFinal dataset shape: {customer_df.shape}")
    print(f"Total features: {len(customer_df.columns)}")
    
    return customer_df, test_processed, woe_iv_summary


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Pipeline for Credit Scoring Model")
    print("This module provides transformers and pipelines for data processing.")
    print("\nExample usage:")
    print("  from data_processing import process_data")
    print("  processed_df, _, woe_summary = process_data(raw_df, target_col='default')")