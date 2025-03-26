# Save this as main.py

import os
import pandas as pd
import numpy as np
import logging
from src.data_processor import DataProcessor
from src.anomaly_detector import AnomalyDetector
from config import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the reconciliation anomaly detection process
    """
    logger.info("Starting reconciliation anomaly detection")
    
    # Check if data directory exists and has files
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.warning(f"Data directory {DATA_DIR} created. Please add sample data files.")
        return
    
    # Find data files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx'))]
    if not files:
        logger.warning(f"No data files found in {DATA_DIR}. Please add sample CSV or Excel files.")
        return
    
    # Assume first file is current data and second is historical
    current_file = os.path.join(DATA_DIR, files[0])
    historical_file = os.path.join(DATA_DIR, files[1]) if len(files) > 1 else None
    
    # Initialize data processor for GL_IHUB reconciliation
    data_processor = DataProcessor(recon_type='GL_IHUB')
    
    # Process data
    if historical_file:
        logger.info(f"Processing data with historical context: {current_file}, {historical_file}")
        df, X, feature_names = data_processor.process_data_for_training(current_file, historical_file)
    else:
        logger.info(f"Processing data without historical context: {current_file}")
        df, X, feature_names = data_processor.process_data_for_training(current_file)
    
    # Initialize and train anomaly detector
    detector = AnomalyDetector(method='isolation_forest')
    detector.fit(X, feature_names)
    
    # Detect anomalies
    predictions = detector.predict(X)
    anomaly_indices = np.where(predictions == -1)[0].tolist()
    
    # Get anomaly scores
    scores = detector.predict_scores(X)
    
    # Add anomaly information to dataframe
    df['Is_Anomaly'] = (predictions == -1).astype(int)
    df['Anomaly_Score'] = scores
    
    # Print summary of results
    logger.info(f"Processed {len(df)} records")
    logger.info(f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(df)*100:.2f}%)")
    
    if anomaly_indices:
        logger.info("Top 5 anomalies:")
        # Sort anomalies by score
        top_anomalies = df.loc[df['Is_Anomaly'] == 1].sort_values('Anomaly_Score', ascending=False).head(5)
        for idx, row in top_anomalies.iterrows():
            logger.info(f"Record {idx} - Score: {row['Anomaly_Score']:.4f}")
            # Print key fields
            for col in data_processor.key_columns:
                if col in row:
                    logger.info(f"  {col}: {row[col]}")
            # Print main difference
            if 'Balance Difference' in row:
                logger.info(f"  Balance Difference: {row['Balance Difference']}")
    
    # Save results to file
    output_file = os.path.join(DATA_DIR, "anomaly_results.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()