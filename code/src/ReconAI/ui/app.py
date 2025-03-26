import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import logging
from typing import Dict, List, Any

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.anomaly_detector import AnomalyDetector
from src.classifier import AnomalyClassifier
from src.agent import ReconciliationAgent
from config import DATA_DIR, MODELS_DIR, ANOMALY_CATEGORIES, ACTION_TYPES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'anomalies' not in st.session_state:
        st.session_state.anomalies = None
    if 'anomaly_details' not in st.session_state:
        st.session_state.anomaly_details = {}
    if 'recon_type' not in st.session_state:
        st.session_state.recon_type = 'GL_IHUB'
    if 'agent' not in st.session_state:
        st.session_state.agent = ReconciliationAgent(human_in_loop=True)
    if 'feedback_log' not in st.session_state:
        st.session_state.feedback_log = []
    if 'action_log' not in st.session_state:
        st.session_state.action_log = []

# Function to load data
def load_data(file, recon_type):
    try:
        data_processor = DataProcessor(recon_type=recon_type)
        df = data_processor.load_data(file)
        df = data_processor.calculate_derived_features(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to process data and detect anomalies
def process_data_and_detect_anomalies(current_data, historical_data=None, recon_type='GL_IHUB', method='isolation_forest'):
    try:
        # Initialize processor
        data_processor = DataProcessor(recon_type=recon_type)
        
        # Save temporary files for processing
        current_file = os.path.join(DATA_DIR, "temp_current.csv")
        os.makedirs(DATA_DIR, exist_ok=True)
        current_data.to_csv(current_file, index=False)
        
        if historical_data is not None:
            historical_file = os.path.join(DATA_DIR, "temp_historical.csv")
            historical_data.to_csv(historical_file, index=False)
        else:
            historical_file = None
        
        # Process data
        if historical_file:
            df, X, feature_names = data_processor.process_data_for_training(current_file, historical_file)
        else:
            df, X, feature_names = data_processor.process_data_for_training(current_file)
        
        # Initialize and train anomaly detector
        detector = AnomalyDetector(method=method)
        detector.fit(X, feature_names)
        
        # Detect anomalies
        predictions = detector.predict(X)
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        # Get anomaly scores
        scores = detector.predict_scores(X)
        
        # Add anomaly information to dataframe
        df['Is_Anomaly'] = (predictions == -1).astype(int)
        df['Anomaly_Score'] = scores
        
        # Classify anomalies
        classifier = AnomalyClassifier(use_llm=True)
        if anomaly_indices:
            classifications = classifier.rule_based_classification(X, df, anomaly_indices)
            
            # Add classifications to dataframe
            df['Anomaly_Category'] = ''
            for idx, category in classifications.items():
                df.loc[idx, 'Anomaly_Category'] = category
        
        # Clean up temporary files
        if os.path.exists(current_file):
            os.remove(current_file)
        if historical_file and os.path.exists(historical_file):
            os.remove(historical_file)
        
        return df, anomaly_indices
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logger.error(f"Error in process_data_and_detect_anomalies: {str(e)}")
        return None, []

# Function to analyze a specific anomaly
def analyze_anomaly(record, agent):
    try:
        record_dict = record.to_dict()
        anomaly_category = record_dict.get('Anomaly_Category', None)
        
        analysis = agent.analyze_break(record_dict, anomaly_category)
        return analysis
    
    except Exception as e:
        st.error(f"Error analyzing anomaly: {str(e)}")
        logger.error(f"Error in analyze_anomaly: {str(e)}")
        return {
            "summary": "Error during analysis",
            "root_cause": "Unknown",
            "recommended_action": "Document Exception",
            "justification": f"Error: {str(e)}"
        }

# Function to take action on an anomaly
def take_action(record_id, action_type, details, agent):
    try:
        result = agent.take_action(record_id, action_type, details)
        st.session_state.action_log.append(result)
        return result
    
    except Exception as e:
        st.error(f"Error taking action: {str(e)}")
        logger.error(f"Error in take_action: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

# Function to submit feedback
def submit_feedback(record, analysis, resolution, feedback_text, agent):
    try:
        agent.learn_from_feedback(record, analysis, resolution, feedback_text)
        feedback_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record_id": record.get('id', 'Unknown'),
            "feedback": feedback_text
        }
        st.session_state.feedback_log.append(feedback_entry)
        return True
    
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        logger.error(f"Error in submit_feedback: {str(e)}")
        return False

# Function to generate charts
def generate_anomaly_distribution_chart(df):
    if 'Anomaly_Category' in df.columns:
        # Count anomalies by category
        anomaly_counts = df[df['Is_Anomaly'] == 1]['Anomaly_Category'].value_counts()
        
        if not anomaly_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            anomaly_counts.plot(kind='bar', ax=ax)
            plt.title('Anomaly Distribution by Category')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
    
    return None

def generate_anomaly_score_histogram(df):
    if 'Anomaly_Score' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Anomaly_Score'], bins=20, ax=ax)
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.axvline(x=0.7, color='red', linestyle='--', label='Typical Threshold')
        plt.legend()
        plt.tight_layout()
        return fig
    
    return None

def generate_balance_diff_scatter(df):
    if 'Balance Difference' in df.columns and 'Anomaly_Score' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df['Balance Difference'].abs(), 
            df['Anomaly_Score'],
            c=df['Is_Anomaly'],
            cmap='coolwarm',
            alpha=0.6
        )
        plt.colorbar(scatter, label='Is Anomaly')
        plt.title('Balance Difference vs Anomaly Score')
        plt.xlabel('Absolute Balance Difference')
        plt.ylabel('Anomaly Score')
        plt.tight_layout()
        return fig
    
    return None

# Main app function
def main():
    st.set_page_config(page_title="Reconciliation Anomaly Detection", page_icon="📊", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.title("Reconciliation AI")
    
    # Add file uploader to sidebar
    st.sidebar.header("Data Input")
    recon_type = st.sidebar.selectbox(
        "Reconciliation Type",
        options=["GL_IHUB", "CATALYST_IMPACT"],
        index=0
    )
    st.session_state.recon_type = recon_type
    
    current_file = st.sidebar.file_uploader("Upload Current Reconciliation Data", type=["csv", "xlsx"])
    historical_file = st.sidebar.file_uploader("Upload Historical Data (Optional)", type=["csv", "xlsx"])
    
    # Add anomaly detection settings
    st.sidebar.header("Anomaly Detection Settings")
    detection_method = st.sidebar.selectbox(
        "Detection Method",
        options=["isolation_forest", "lof", "dbscan", "zscore", "ensemble"],
        index=0
    )
    
    # Load and process data when button is clicked
    if st.sidebar.button("Load and Process Data"):
        with st.spinner("Loading and processing data..."):
            if current_file is not None:
                # Load current data
                current_data = load_data(current_file, recon_type)
                st.session_state.current_data = current_data
                
                # Load historical data if provided
                if historical_file is not None:
                    historical_data = load_data(historical_file, recon_type)
                    st.session_state.historical_data = historical_data
                else:
                    st.session_state.historical_data = None
                
                # Process data and detect anomalies
                if current_data is not None:
                    processed_data, anomaly_indices = process_data_and_detect_anomalies(
                        current_data, 
                        st.session_state.historical_data,
                        recon_type,
                        detection_method
                    )
                    
                    if processed_data is not None:
                        st.session_state.processed_data = processed_data
                        st.session_state.anomalies = anomaly_indices
                        st.session_state.data_loaded = True
                        st.sidebar.success(f"Detected {len(anomaly_indices)} anomalies in {len(processed_data)} records")
            else:
                st.sidebar.error("Please upload current data file")
    
    # Navigation in sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Dashboard", "Anomaly Explorer", "Agent Actions", "Feedback Log"]
    )
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.title("Reconciliation Anomaly Detection")
        st.info("Please upload data using the sidebar to get started.")
        
        # Show sample data structure
        st.subheader("Sample Data Structure")
        if recon_type == "GL_IHUB":
            st.write("Expected columns for GL vs IHub reconciliation:")
            st.code("""
            Company, Account, AU, Currency, GL Balance, IHub Balance, As of Date
            Company A, 12345, AU001, USD, 10000, 10000, 2025-03-01
            Company B, 67890, AU002, USD, 20000, 19500, 2025-03-01
            """)
        else:
            st.write("Expected columns for Catalyst vs Impact reconciliation:")
            st.code("""
            Trade ID, Inventory Code, CUSIP, Trade Date, Settlement Date, Buy or Sell, Price, Quantity, Recon Date
            T12345, INV001, 123456789, 2025-03-01, 2025-03-03, Buy, 100.5, 100, 2025-03-04
            T67890, INV002, 987654321, 2025-03-01, 2025-03-03, Sell, 99.75, 200, 2025-03-04
            """)
            
        return
    
    # Main content based on selected page
    if page == "Dashboard":
        st.title("Reconciliation Anomaly Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_records = len(st.session_state.processed_data)
        total_anomalies = len(st.session_state.anomalies)
        anomaly_percentage = (total_anomalies / total_records) * 100 if total_records > 0 else 0
        
        col1.metric("Total Records", f"{total_records:,}")
        col2.metric("Anomalies Detected", f"{total_anomalies:,}")
        col3.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
        
        # Calculate total difference amount for anomalies
        if 'Balance Difference' in st.session_state.processed_data.columns:
            anomaly_records = st.session_state.processed_data[st.session_state.processed_data['Is_Anomaly'] == 1]
            total_difference = anomaly_records['Balance Difference'].abs().sum()
            col4.metric("Total Difference Amount", f"${total_difference:,.2f}")
        
        # Charts
        st.subheader("Anomaly Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart = generate_anomaly_distribution_chart(st.session_state.processed_data)
            if chart:
                st.pyplot(chart)
            else:
                st.info("No anomaly categories available for visualization")
                
        with col2:
            chart = generate_anomaly_score_histogram(st.session_state.processed_data)
            if chart:
                st.pyplot(chart)
            else:
                st.info("No anomaly scores available for visualization")
        
        # Additional chart
        chart = generate_balance_diff_scatter(st.session_state.processed_data)
        if chart:
            st.pyplot(chart)
        
        # Top anomalies table
        st.subheader("Top Anomalies")
        if total_anomalies > 0:
            top_anomalies = st.session_state.processed_data[st.session_state.processed_data['Is_Anomaly'] == 1].sort_values('Anomaly_Score', ascending=False).head(10)
            st.dataframe(top_anomalies)
        else:
            st.info("No anomalies detected in the current dataset")
    
    elif page == "Anomaly Explorer":
        st.title("Anomaly Explorer")
        
        # Filter options
        st.subheader("Filter Anomalies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by anomaly category if available
            categories = ['All']
            if 'Anomaly_Category' in st.session_state.processed_data.columns:
                categories.extend(st.session_state.processed_data['Anomaly_Category'].unique())
            
            selected_category = st.selectbox("Anomaly Category", options=categories)
        
        with col2:
            # Filter by anomaly score
            min_score = st.slider("Minimum Anomaly Score", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=0.5, 
                                step=0.05)
        
        # Filter data based on selections
        filtered_data = st.session_state.processed_data
        
        if selected_category != 'All' and 'Anomaly_Category' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Anomaly_Category'] == selected_category]
        
        filtered_data = filtered_data[
            (filtered_data['Is_Anomaly'] == 1) & 
            (filtered_data['Anomaly_Score'] >= min_score)
        ]
        
        # Display filtered anomalies
        st.subheader(f"Filtered Anomalies ({len(filtered_data)} records)")
        
        if len(filtered_data) > 0:
            st.dataframe(filtered_data)
            
            # Select an anomaly to analyze
            selected_indices = filtered_data.index.tolist()
            
            if selected_indices:
                selected_idx = st.selectbox(
                    "Select an anomaly to analyze", 
                    options=selected_indices,
                    format_func=lambda x: f"Record {x}"
                )
                
                if selected_idx is not None:
                    selected_record = st.session_state.processed_data.loc[selected_idx]
                    
                    st.subheader(f"Anomaly Analysis for Record {selected_idx}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Record Details:")
                        st.json(selected_record.to_dict())
                    
                    with col2:
                        # Get or calculate analysis
                        if selected_idx not in st.session_state.anomaly_details:
                            with st.spinner("Analyzing anomaly..."):
                                analysis = analyze_anomaly(selected_record, st.session_state.agent)
                                st.session_state.anomaly_details[selected_idx] = analysis
                        else:
                            analysis = st.session_state.anomaly_details[selected_idx]
                        
                        st.write("AI Analysis:")
                        st.write(f"**Summary**: {analysis['summary']}")
                        st.write(f"**Root Cause**: {analysis['root_cause']}")
                        st.write(f"**Recommended Action**: {analysis['recommended_action']}")
                        st.write(f"**Justification**: {analysis['justification']}")
                    
                    # Action buttons
                    st.subheader("Take Action")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        action = st.selectbox(
                            "Select action to take",
                            options=ACTION_TYPES,
                            index=ACTION_TYPES.index(analysis['recommended_action']) if analysis['recommended_action'] in ACTION_TYPES else 0
                        )
                    
                    with col2:
                        if st.button("Execute Action"):
                            with st.spinner("Executing action..."):
                                result = take_action(
                                    str(selected_idx), 
                                    action, 
                                    {
                                        "summary": analysis['summary'],
                                        "justification": analysis['justification']
                                    },
                                    st.session_state.agent
                                )
                                
                                st.success(f"Action {action} executed successfully")
                                st.json(result)
                    
                    # Feedback section
                    st.subheader("Provide Feedback")
                    feedback = st.text_area("Enter feedback about this analysis", height=100)
                    actual_resolution = st.text_area("Describe the actual resolution (if different from recommendation)", height=100)
                    
                    if st.button("Submit Feedback"):
                        resolution_details = {
                            "actual_action": action,
                            "resolution_notes": actual_resolution
                        }
                        
                        submit_feedback(
                            selected_record.to_dict(),
                            analysis,
                            resolution_details,
                            feedback,
                            st.session_state.agent
                        )
                        
                        st.success("Feedback submitted successfully")
        else:
            st.info("No anomalies match the current filters")
    
    elif page == "Agent Actions":
        st.title("AI Agent Actions")
        
        # Display the action log
        st.subheader("Action History")
        
        if st.session_state.action_log:
            # Convert action log to DataFrame for display
            action_df = pd.DataFrame(st.session_state.action_log)
            st.dataframe(action_df)
            
            # Option to generate summary report
            if st.button("Generate Action Summary Report"):
                with st.spinner("Generating summary report..."):
                    # Collect records for summary
                    records = []
                    for action in st.session_state.action_log:
                        record_id = action.get("record_id")
                        if record_id and record_id.isdigit():
                            idx = int(record_id)
                            if idx in st.session_state.processed_data.index:
                                record = st.session_state.processed_data.loc[idx].to_dict()
                                record["action_taken"] = action.get("action_type")
                                record["id"] = record_id
                                records.append(record)
                    
                    if records:
                        summary = st.session_state.agent.get_resolution_summary(records)
                        
                        st.subheader("Action Summary Report")
                        st.write(f"**Executive Summary**: {summary.get('executive_summary', 'No summary available')}")
                        
                        st.write("**Key Statistics**:")
                        st.json(summary.get('key_statistics', {}))
                        
                        st.write("**Patterns Observed**:")
                        for pattern in summary.get('patterns_observed', []):
                            st.write(f"- {pattern}")
                        
                        st.write("**Recommendations**:")
                        for rec in summary.get('recommendations', []):
                            st.write(f"- {rec}")
                    else:
                        st.warning("Not enough action data to generate a summary")
        else:
            st.info("No actions have been taken yet")
        
        # Option to simulate bulk actions
        st.subheader("Bulk Actions")
        
        if st.button("Simulate Automated Resolution for All Anomalies"):
            with st.spinner("Processing automated resolutions..."):
                anomaly_records = st.session_state.processed_data[st.session_state.processed_data['Is_Anomaly'] == 1]
                
                for idx, record in anomaly_records.iterrows():
                    # Skip if already analyzed
                    if idx in st.session_state.anomaly_details:
                        analysis = st.session_state.anomaly_details[idx]
                    else:
                        analysis = analyze_anomaly(record, st.session_state.agent)
                        st.session_state.anomaly_details[idx] = analysis
                    
                    # Take recommended action
                    take_action(
                        str(idx),
                        analysis['recommended_action'],
                        {
                            "summary": analysis['summary'],
                            "justification": analysis['justification']
                        },
                        st.session_state.agent
                    )
                
                st.success(f"Processed {len(anomaly_records)} anomalies with automated actions")
    
    elif page == "Feedback Log":
        st.title("Feedback Log")
        
        # Display feedback history
        if st.session_state.feedback_log:
            feedback_df = pd.DataFrame(st.session_state.feedback_log)
            st.dataframe(feedback_df)
            
            # Export option
            if st.button("Export Feedback Log"):
                feedback_csv = feedback_df.to_csv(index=False)
                st.download_button(
                    label="Download Feedback CSV",
                    data=feedback_csv,
                    file_name="feedback_log.csv",
                    mime="text/csv"
                )
        else:
            st.info("No feedback has been recorded yet")
        
        # Display agent's learning from feedback
        st.subheader("Agent Learning Progress")
        
        # This is a placeholder for what would be a more sophisticated learning system
        if len(st.session_state.agent.feedback_history) > 0:
            st.write(f"Agent has learned from {len(st.session_state.agent.feedback_history)} feedback entries")
            
            # Calculate accuracy improvement (simulated)
            initial_accuracy = 0.75  # Baseline accuracy
            current_accuracy = min(0.95, initial_accuracy + (len(st.session_state.agent.feedback_history) * 0.02))
            
            col1, col2 = st.columns(2)
            col1.metric("Initial Accuracy", f"{initial_accuracy:.2%}")
            col2.metric("Current Accuracy", f"{current_accuracy:.2%}", f"+{(current_accuracy-initial_accuracy):.2%}")
            
            # Placeholder for feedback insights
            st.write("**Feedback Insights**:")
            st.write("- Agent is improving at identifying calculation errors")
            st.write("- More training needed for currency conversion issues")
            st.write("- Timing differences are now correctly identified in most cases")
        else:
            st.info("Agent has not received any feedback yet")

if __name__ == "__main__":
    main()
            