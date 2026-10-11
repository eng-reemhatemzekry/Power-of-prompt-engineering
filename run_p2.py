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

P2 = ("""
You are a banking intent classification model.

Your task is to classify the user's message into exactly one of these two categories.

# Categories

- faq:
  General questions about the bank, its services, products, fees, policies,
  procedures, eligibility, or other information that does not depend on
  the user's personal account.

- customer:
  Questions about the user's own banking information or activity, including
  their account, balance, card, loan, payment, transfer, or transaction.

# Decision Rule

Classify as "customer" when the user is asking about their own
account-specific information or activity.

Classify as "faq" when the question asks for general banking information
that could be answered without accessing the user's personal data.

#Examples:
question: What documents are required to apply for a personal loan?
answer: faq

# Output

Return exactly one label:
faq
or
customer

Do not provide an explanation.
Do not return punctuation.
Do not return any other text.
"""
)


answers = []
for question in df["question"]:
    answers.append(ask_mistral(P2, question))
    time.sleep(2)  

df["p2"] = answers


df.to_csv("results_p2.csv", index=False, encoding="utf-8-sig")
pivot = pd.crosstab(
    index=[df["language"], df["route"]],  # rows: language, then true route
    columns=df["p2"],                     # columns: model answer (faq / customer)
)
pivot = pivot.reindex(columns=["faq", "customer"], fill_value=0)

print("\nPivot table (rows = language and true route, columns = model answer):")
print(pivot)

accuracy = (df["p2"] == df["route"]).mean()
print("\nP1 accuracy:", round(accuracy, 3))