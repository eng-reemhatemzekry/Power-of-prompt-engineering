import os
import time
import pandas as pd
from mistralai.client import Mistral

MODEL = "ministral-14b-latest"        #"mistral-small-2603"

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)


def ask_mistral(system_prompt, question):
    for attempt in range(3): 
        response = client.chat.complete(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content.strip().lower()

        if "customer" in text and "faq" not in text:
            return "customer"
        if "faq" in text and "customer" not in text:
            return "faq"

    raise ValueError(f"No valid answer for: {question}")


CSV_PATH = "ar-en-faq-customer-dataset.csv"

df = pd.read_csv(CSV_PATH)
print(df.head())

P1 = (
    """You are a routing classifier for a bank.
    Classify the user's question as one of two route types:
    - faq: general questions about the bank's services, products, fees or policies.
    - customer: questions about the customer's own account, card, loan or transactions.
    Answer with exactly one word: faq or customer. No explanation."""
)


answers = []
for question in df["question"]:
    answers.append(ask_mistral(P1, question))
    time.sleep(6)  

df["p1"] = answers


df.to_csv("results_p1.csv", index=False, encoding="utf-8-sig")
pivot = pd.crosstab(
    index=[df["language"], df["route"]],  # rows: language, then true route
    columns=df["p1"],                     # columns: model answer (faq / customer)
)
pivot = pivot.reindex(columns=["faq", "customer"], fill_value=0)

print("\nPivot table (rows = language and true route, columns = model answer):")
print(pivot)

accuracy = (df["p1"] == df["route"]).mean()
print("\nP1 accuracy:", round(accuracy, 3))