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

P3 = ("""
You are a banking intent classification model.

Your task is to classify the user's message into exactly one of these two categories.

# Categories

- faq:
  General questions about the bank, its services, products, fees, policies,
  procedures, eligibility, or how a banking service works.
  The question may use personal wording such as "my", "I", or "me" and can
  still be FAQ if it is asking about a general service, policy, procedure,
  eligibility, or capability rather than requesting information about a
  specific customer account, transaction, or financial record.

- customer:
  Questions that require information about the specific customer's account,
  balance, card, loan, payment, transfer, transaction, or other personal
  banking activity.

# Decision Process

First, determine whether the question can be answered using general banking
information, without accessing or examining the customer's specific records.

If it can be answered with general information about a service, product,
policy, fee, procedure, or eligibility, classify it as "faq".

If answering it requires accessing, checking, or investigating the customer's
specific account, card, loan, payment, transfer, transaction, balance, or
other personal banking records, classify it as "customer".

Do not classify a question as "customer" merely because it contains words
such as "my", "I", or "me".

# Examples

question: What documents are required to apply for a personal loan?
answer: faq

question: Can I transfer money from my account to another bank?
answer: faq

question: Why did my transfer to another bank fail?
answer: customer

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
    answers.append(ask_mistral(P3, question))
    time.sleep(2)  

df["p3"] = answers


df.to_csv("results_p3.csv", index=False, encoding="utf-8-sig")
pivot = pd.crosstab(
    index=[df["language"], df["route"]],  # rows: language, then true route
    columns=df["p3"],                     # columns: model answer (faq / customer)
)
pivot = pivot.reindex(columns=["faq", "customer"], fill_value=0)

print("\nPivot table (rows = language and true route, columns = model answer):")
print(pivot)

accuracy = (df["p3"] == df["route"]).mean()
print("\nP3 accuracy:", round(accuracy, 3))