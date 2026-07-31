from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b")

with open("blood_report_format.txt", "r") as f:
    blood_report = f.read()

extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL 
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
extraction_response = llm.invoke(extraction_prompt)
extracted_values = extraction_response.content

# diet prompt 

diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of. 
   Do not include any other sections in diet plan.

Blood Work Analysis:
{extracted_values}
"""

diet_response = llm.invoke(diet_prompt)

print("=== STAGE 2: HEALTH SUMMARY & DIET PLAN ===")

print(diet_response.content)