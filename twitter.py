import pandas as pd
import re
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

data = {
    "date": [
        "2026-05-01", "2026-05-01",
        "2026-05-02", "2026-05-02",
        "2026-05-03", "2026-05-03"
    ],
    "tweet": [
        "Love the new iPhone 🔥 absolutely amazing!",
        "Worst experience ever... very disappointed 😡",
        "It's okay, nothing special",
        "Really happy with the service 😊 #satisfied",
        "Totally love this product!! #awesome",
        "Not good, I expected more..."
    ]
}

df = pd.DataFrame(data)

def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)      
    tweet = re.sub(r"@\w+", "", tweet)         
    tweet = re.sub(r"#", "", tweet)           
    tweet = re.sub(r"[^a-z\s]", "", tweet)    
    tweet = re.sub(r"\s+", " ", tweet).strip() 
    return tweet


df["clean_tweet"] = df["tweet"].apply(clean_tweet)
def get_score(text):
    return sia.polarity_scores(text)["compound"]

def get_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["sentiment_score"] = df["clean_tweet"].apply(get_score)
df["sentiment_label"] = df["sentiment_score"].apply(get_label)
df["date"] = pd.to_datetime(df["date"])

print("\nFinal Data with Sentiment:\n")
print(df)

daily_sentiment = df.groupby("date")["sentiment_score"].mean()

plt.figure(figsize=(10,5))
plt.plot(daily_sentiment.index, daily_sentiment.values, marker="o")
plt.title("Twitter Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.xticks(rotation=45)
plt.grid()
plt.show()
plt.figure(figsize=(6,6))
df["sentiment_label"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()