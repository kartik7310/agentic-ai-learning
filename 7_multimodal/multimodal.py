from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
import base64

load_dotenv()

with open("blood_work.png","rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

llm = ChatGroq(model="qwen/qwen3.6-27b")
message = HumanMessage(content=[
  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    {"type": "text",      "text": "This is a blood work report. Extract all test results and flag any values outside the normal range."}
])


# diet recomandation tool

@tool
def get_diet_recommendation(condition: str) -> dict:
    """Given a health condition, returns a diet plan. Condition must be one of: normal, high_cholesterol, high_sugar."""
    diet_plans = {
        "high_cholesterol": {
            "eat":        ["fruits", "vegetables", "whole grains", "lean protein"],
            "do_not_eat": ["red meat", "fried food", "full-fat dairy", "processed snacks"],
        },
        "high_sugar": {
            "eat":        ["vegetables", "whole grains", "legumes", "nuts"],
            "do_not_eat": ["white rice", "white sugar", "junk food", "sugary drinks"],
        },
        "normal": {
            "eat":        ["vegetables", "fruits", "whole grains", "lean protein"],
            "do_not_eat": ["excessive sugar", "processed food", "trans fats"],
        },
    }
    return diet_plans.get(condition)


SYSTEM_PROMPT = """
You are a helpful medical and nutrition assistant.
For the input blood work image, extract the numbers and the normal range, then categorize
the condition as one of: normal, high_cholesterol, high_sugar.
Then call the appropriate tool to retrieve and present the diet plan.
"""

diet_agent = create_agent(
    llm,
    tools=[get_diet_recommendation],
    system_prompt=SYSTEM_PROMPT,
)


result = diet_agent.invoke({
    "messages": [HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text",      "text": "Analyse this blood work report and suggest a diet plan."},
    ])]
})

print(result["messages"][-1].content)