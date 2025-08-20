# 📌 Student Feedback Analysis - Jupyter Notebook (with chart saving)

# 1️⃣ Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# 2️⃣ Load your CSV file (with encoding fix for Windows files)
file_path = r"C:\Users\admin\Downloads\data_set_3\Student_Satisfaction_Survey.csv"

try:
    df = pd.read_csv(file_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="latin1")   # fallback for Windows CSVs

print("✅ Data Loaded Successfully!")
display(df.head())

# 3️⃣ Fix special characters globally
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace("’", "'", regex=False)
    df[col] = df[col].str.replace("‘", "'", regex=False)
    df[col] = df[col].str.replace("“", '"', regex=False)
    df[col] = df[col].str.replace("”", '"', regex=False)
    df[col] = df[col].str.replace("–", "-", regex=False)
    df[col] = df[col].str.replace("—", "-", regex=False)

plt.rcParams["font.family"] = "Arial"

# 4️⃣ Basic Cleaning
df = df.dropna(how="all")
if "Comments" in df.columns:
    df["Comments"] = df["Comments"].astype(str).str.strip()

# 5️⃣ Rating Analysis
if all(col in df.columns for col in ["Weightage 1","Weightage 2","Weightage 3","Weightage 4","Weightage 5"]):
    df["Total_Ratings"] = df[["Weightage 1","Weightage 2","Weightage 3","Weightage 4","Weightage 5"]].sum(axis=1)

    df["Average_Rating"] = (
        df["Weightage 1"]*1 + df["Weightage 2"]*2 +
        df["Weightage 3"]*3 + df["Weightage 4"]*4 + df["Weightage 5"]*5
    ) / df["Total_Ratings"]

    print("\n📊 Average Ratings by Question:")
    display(df[["Questions","Average_Rating"]])

    # Bar Chart
    plt.figure(figsize=(10,5))
    sns.barplot(x="Average_Rating", y="Questions", data=df, color="steelblue")
    plt.title("Average Rating per Question")
    plt.xlabel("Average Rating (1–5)")
    plt.ylabel("Questions")
    plt.savefig(r"C:\Users\admin\Downloads\data_set_3\average_ratings.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Pie Charts
    for i, row in df.iterrows():
        weights = [row["Weightage 1"], row["Weightage 2"], row["Weightage 3"], row["Weightage 4"], row["Weightage 5"]]
        labels = ["1★","2★","3★","4★","5★"]

        plt.figure(figsize=(6,6))
        plt.pie(weights, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f"Rating Distribution: {row['Questions']}")
        save_path = fr"C:\Users\admin\Downloads\data_set_3\piechart_question_{i+1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

# 6️⃣ Sentiment Analysis
if "Comments" in df.columns:
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        score = analyzer.polarity_scores(text)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Comments"].apply(get_sentiment)

    print("\n📝 Sentiment Distribution:")
    display(df["Sentiment"].value_counts())

    # Sentiment Pie Chart
    plt.figure(figsize=(6,6))
    df["Sentiment"].value_counts().plot.pie(
        autopct="%1.1f%%",
        colors=["green","red","grey"]
    )
    plt.title("Sentiment Distribution of Student Comments")
    plt.ylabel("")
    plt.savefig(r"C:\Users\admin\Downloads\data_set_3\sentiment_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

# 7️⃣ Word Cloud
if "Comments" in df.columns:
    text = " ".join(df["Comments"].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Student Feedback")
    plt.savefig(r"C:\Users\admin\Downloads\data_set_3\wordcloud.png", dpi=300, bbox_inches="tight")
    plt.show()

# 8️⃣ Save Processed Results
output_file = r"C:\Users\admin\Downloads\data_set_3\Processed_Student_Feedback.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Analysis Completed! Results & Charts saved in: {output_file}")
