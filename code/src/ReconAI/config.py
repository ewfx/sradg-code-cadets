import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data Paths
DATA_DIR = "data"
MODELS_DIR = "models"

# Anomaly Detection Settings
ANOMALY_THRESHOLD = 0.05  # Default threshold for anomaly detection
HISTORICAL_WINDOW = 30    # Number of days to consider for historical patterns

# Classification Settings
CONFIDENCE_THRESHOLD = 0.7

# Reconciliation Specific
# GL vs IHub Reconciliation
GL_IHUB_KEY_COLUMNS = ["Company", "Account", "AU", "Currency"]
GL_IHUB_CRITERIA_COLUMNS = ["GL Balance", "IHub Balance"]
GL_IHUB_DERIVED_COLUMNS = ["Balance Difference"]
GL_IHUB_HISTORICAL_COLUMNS = ["Account", "Secondary Account", "Primary Account"]
GL_IHUB_DATE_COLUMNS = ["As of Date"]

# Catalyst VS Impact Reconciliation
CATALYST_IMPACT_KEY_COLUMNS = ["Trade ID"]
CATALYST_IMPACT_CRITERIA_COLUMNS = ["Inventory Code", "CUSIP", "Trade Date", 
                                    "Settlement Date", "Buy or Sell", "Price", "Quantity"]
CATALYST_IMPACT_HISTORICAL_COLUMNS = ["CUSIP", "Inventory Code"]
CATALYST_IMPACT_DATE_COLUMNS = ["Recon Date"]

# Anomaly Classification Categories
ANOMALY_CATEGORIES = [
    "Missing Data",
    "Timing Difference",
    "Calculation Error",
    "System Processing Error",
    "Currency Conversion Issue",
    "Duplicate Entry",
    "Incorrect Mapping",
    "Reconciliation Rule Issue",
    "Other"
]

# Action Types
ACTION_TYPES = [
    "Create JIRA Ticket",
    "Send Email",
    "Update Source System",
    "Document Exception",
    "Reprocess Reconciliation",
    "Escalate to Team Lead",
    "No Action Required"
]