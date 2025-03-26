import pandas as pd
import openai
from langchain.llms import HuggingFaceHub
from transformers import pipeline

llm = HuggingFaceHub(repo_id="google/flan-t5-large",huggingfacehub_api_token=API_TOKEN)

def explain_anomaly(new_entry, context=None):
    try:
        generator = pipeline("text-generation", model="gpt2")
        prompt = f"""
            Given the following data:
            - Match Status: {new_entry['Match Status']}
            - Anomaly: {new_entry['Anomaly']}
            - Balance Difference: {new_entry['Balance Difference']}
            Generate a meaningful insight explaining the situation.
            """
        output = generator(prompt, max_length=150, num_return_sequences=1)

        if output and len(output) > 0 and 'generated_text' in output[0]:
            explanation = output[0]['generated_text'].replace(prompt, "").strip()
            return f"{explanation}"
        else:
            return "Could not generate insights from the LLM."

    except Exception as e:
        return f"An error occurred while generating insights: {e}"


def load_data(file_path):
    df = pd.read_excel(file_path)
    return df


def main(file_path):
    df = load_data(file_path)
    new_entries = df[df['Insights'].isna()]
    new_comments = [explain_anomaly(row) for _, row in new_entries.iterrows()]
    df.loc[df['Insights'].isna(), 'Generated Insights'] = new_comments
    
    return df[['As Of Date', 'Company', 'Account', 'Generated Insights']]


file_path = "./../data/current.xlsx"
result_df = main(file_path)
print(result_df.head())
