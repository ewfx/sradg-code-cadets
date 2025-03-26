import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from config import DATA_DIR, MODELS_DIR
from config import GL_IHUB_KEY_COLUMNS, GL_IHUB_CRITERIA_COLUMNS, GL_IHUB_DERIVED_COLUMNS
from config import CATALYST_IMPACT_KEY_COLUMNS, CATALYST_IMPACT_CRITERIA_COLUMNS
from sklearn.preprocessing import OneHotEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, recon_type: str = 'GL_IHUB'):
        self.recon_type = recon_type
        if recon_type == 'GL_IHUB':
            self.key_columns = GL_IHUB_KEY_COLUMNS
            self.criteria_columns = GL_IHUB_CRITERIA_COLUMNS
            self.derived_columns = GL_IHUB_DERIVED_COLUMNS
        elif recon_type == 'CATALYST_IMPACT':
            self.key_columns = CATALYST_IMPACT_KEY_COLUMNS
            self.criteria_columns = CATALYST_IMPACT_CRITERIA_COLUMNS
            self.derived_columns = ["Difference"] 
        else:
            raise ValueError(f"Unsupported reconciliation type: {recon_type}")
        
        self.scaler = None
        self.encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_excel(file_path)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                logger.warning(f"Could not convert column {col} to datetime")
        
        logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        
        if self.recon_type == 'GL_IHUB':
            # Calculate balance difference
            if 'GL Balance' in df_copy.columns and 'IHub Balance' in df_copy.columns:
                df_copy['Balance Difference'] = df_copy['GL Balance'] - df_copy['IHub Balance']
                df_copy['Balance Difference Abs'] = df_copy['Balance Difference'].abs()
                df_copy['Balance Difference Percentage'] = df_copy['Balance Difference'].abs() / df_copy['GL Balance'].abs() * 100
                df_copy['Balance Difference Percentage'] = df_copy['Balance Difference Percentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        elif self.recon_type == 'CATALYST_IMPACT':
            for col in self.criteria_columns:
                if f"Catalyst_{col}" in df_copy.columns and f"Impact_{col}" in df_copy.columns:
                    if df_copy[f"Catalyst_{col}"].dtype in [np.int64, np.float64] and df_copy[f"Impact_{col}"].dtype in [np.int64, np.float64]:
                        df_copy[f"{col}_Difference"] = df_copy[f"Catalyst_{col}"] - df_copy[f"Impact_{col}"]
                        df_copy[f"{col}_Difference_Abs"] = df_copy[f"{col}_Difference"].abs()
                    else:
                        df_copy[f"{col}_Match"] = (df_copy[f"Catalyst_{col}"] == df_copy[f"Impact_{col}"]).astype(int)
            df_copy["Has_Difference"] = 0
            for col in self.criteria_columns:
                if f"{col}_Difference" in df_copy.columns:
                    df_copy["Has_Difference"] = np.where(df_copy[f"{col}_Difference"] != 0, 1, df_copy["Has_Difference"])
                elif f"{col}_Match" in df_copy.columns:
                    df_copy["Has_Difference"] = np.where(df_copy[f"{col}_Match"] == 0, 1, df_copy["Has_Difference"])
        
        logger.info(f"Derived features calculated for {self.recon_type} reconciliation")
        return df_copy
    

    def add_historical_features(self, current_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
        result_df = current_df.copy()
        if historical_df is None or historical_df.empty:
            logger.warning("No historical data provided for feature calculation")
            return result_df

        date_col = None
        for col in result_df.columns:
            if 'date' in col.lower():
                date_col = col
                break
       
        if date_col is None:
            logger.warning("No date column found for historical analysis")
            return result_df

        for df in [result_df, historical_df]:
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col])
        
        groupby_columns = self.key_columns.copy()
        if date_col in groupby_columns:
            groupby_columns.remove(date_col)
        
        if not groupby_columns:
            logger.warning("No groupby columns available for historical analysis")
            return result_df
        
        if self.recon_type == 'GL_IHUB':
            numeric_cols = ['Balance Difference', 'GL Balance', 'IHub Balance']
            
            historical_stats = historical_df.groupby(groupby_columns).agg({
                'Balance Difference': ['mean', 'std', 'count', 'min', 'max'],
                'GL Balance': ['mean', 'std'],
                'IHub Balance': ['mean', 'std']
            })
            
            historical_stats.columns = ['_'.join(col).strip() for col in historical_stats.columns.values]
            historical_stats = historical_stats.reset_index()
            
            result_df = pd.merge(result_df, historical_stats, on=groupby_columns, how='left')

            if 'Balance Difference_mean' in result_df.columns and 'Balance Difference_std' in result_df.columns:
                result_df['Balance_Difference_ZScore'] = (result_df['Balance Difference'] - result_df['Balance Difference_mean']) / result_df['Balance Difference_std']
                result_df['Balance_Difference_ZScore'] = result_df['Balance_Difference_ZScore'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        elif self.recon_type == 'CATALYST_IMPACT':

            for col in self.criteria_columns:
                diff_col = f"{col}_Difference"
                if diff_col in result_df.columns:
                    if result_df[diff_col].dtype in [np.int64, np.float64]:
                        hist_stats = historical_df.groupby(groupby_columns)[diff_col].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()

                        hist_stats.columns = [col if i < len(groupby_columns) else f"{diff_col}_{col}" 
                                            for i, col in enumerate(hist_stats.columns)]

                        result_df = pd.merge(result_df, hist_stats, on=groupby_columns, how='left')

                        result_df[f"{diff_col}_ZScore"] = (result_df[diff_col] - result_df[f"{diff_col}_mean"]) / result_df[f"{diff_col}_std"]
                        result_df[f"{diff_col}_ZScore"] = result_df[f"{diff_col}_ZScore"].replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in result_df.columns:
            if '_mean' in col or '_std' in col or '_count' in col or '_min' in col or '_max' in col:
                result_df[col] = result_df[col].fillna(0)
        
        logger.info(f"Historical features added, resulting in {result_df.shape[1]} total columns")
        return result_df
    

    def prepare_features_for_model(self, df: pd.DataFrame, train: bool = False) -> Tuple[np.ndarray, List[str]]:
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in self.key_columns and 'ID' not in col]
        feature_cols = [col for col in feature_cols if 'date' not in col.lower() and 'time' not in col.lower()]
        exclude_cols = ['Anomaly', 'Is_Anomaly', 'Anomaly_Type', 'Anomaly_Category']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        X = df[feature_cols].copy()

        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        if train:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(self.scaler, os.path.join(MODELS_DIR, f"{self.recon_type}_scaler.pkl"))
        else:
            if self.scaler is None:
                scaler_path = os.path.join(MODELS_DIR, f"{self.recon_type}_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                else:
                    logger.warning("No scaler found, using StandardScaler without fitting")
                    self.scaler = StandardScaler()
                    X_scaled = X_imputed
            
            X_scaled = self.scaler.transform(X_imputed) if self.scaler else X_imputed
        
        logger.info(f"Prepared {X_scaled.shape[1]} features for modeling")
        return X_scaled, feature_cols
    

    def encode_categorical_variables(self, df: pd.DataFrame, cols: List[str], train: bool = False) -> pd.DataFrame:
        result_df = df.copy()
        
        for col in cols:
            if col in result_df.columns and result_df[col].dtype == 'object':
                if train:
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(result_df[[col]])
                    self.encoders[col] = encoder
                    
                    # Save the encoder
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    joblib.dump(encoder, os.path.join(MODELS_DIR, f"{self.recon_type}_{col}_encoder.pkl"))
                else:
                    if col not in self.encoders:
                        encoder_path = os.path.join(MODELS_DIR, f"{self.recon_type}_{col}_encoder.pkl")
                        if os.path.exists(encoder_path):
                            self.encoders[col] = joblib.load(encoder_path)
                        else:
                            logger.warning(f"No encoder found for {col}, skipping encoding")
                            continue
                    
                    encoder = self.encoders[col]
                    encoded = encoder.transform(result_df[[col]])

                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                for i, enc_col in enumerate(encoded_cols):
                    result_df[enc_col] = encoded[:, i]

                result_df = result_df.drop(col, axis=1)
        
        return result_df
    

    def process_data_for_training(self, current_data_path: str, historical_data_path: str = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:

        df = self.load_data(current_data_path)
        df = self.calculate_derived_features(df)
        if historical_data_path:
            historical_df = self.load_data(historical_data_path)
            historical_df = self.calculate_derived_features(historical_df)
            df = self.add_historical_features(df, historical_df)

        X, feature_names = self.prepare_features_for_model(df, train=True)
        return df, X, feature_names
    

    def process_data_for_inference(self, current_data_path: str, historical_data_path: str = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:

        df = self.load_data(current_data_path)
        df = self.calculate_derived_features(df)

        if historical_data_path:
            historical_df = self.load_data(historical_data_path)
            historical_df = self.calculate_derived_features(historical_df)
            df = self.add_historical_features(df, historical_df)
        
        X, feature_names = self.prepare_features_for_model(df, train=False)
        
        return df, X, feature_names