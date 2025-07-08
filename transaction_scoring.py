import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

INPUT_FILE = "transactions_sample.xlsx"
OUTPUT_FILE = "transaction_scoring_output.xlsx"
CONTAMINATION_LEVEL = 0.05

# Loading data 
df = pd.read_excel(INPUT_FILE)

# Processing EXPENSESUM (sum all numbers)
def sum_expenses(x):
    nums = [float(s.replace(",", "").strip()) for s in str(x).split("~") if s.strip()]
    return sum(nums)

df["EXPENSE_TOTAL"] = df["EXPENSESUM"].apply(sum_expenses)

# Encoding categorical fields
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[["DOCTYPENAME_ENC", "ORGNAME_ENC"]] = encoder.fit_transform(df[["DOCTYPENAME", "ORGNAME"]])

# Extracting text features from PURPOSE ===
tfidf = TfidfVectorizer(max_features=50)
purpose_tfidf = tfidf.fit_transform(df["PURPOSE"].fillna("").astype(str))

svd = TruncatedSVD(n_components=5, random_state=42)
purpose_svd = svd.fit_transform(purpose_tfidf)

purpose_df = pd.DataFrame(purpose_svd, columns=[f"purpose_tfidf_{i}" for i in range(1, 6)])
df = pd.concat([df.reset_index(drop=True), purpose_df.reset_index(drop=True)], axis=1)

# Assembling final feature matrix
features = [
    "EXPENSE_TOTAL", "FILINPUT", "ORGINN",
    "DOCTYPENAME_ENC", "ORGNAME_ENC"
] + list(purpose_df.columns)

X = df[features].fillna(0)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training IsolationForest model
model = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION_LEVEL,
    random_state=42
)
model.fit(X_scaled)

# Making predictions 
scores = model.decision_function(X_scaled)
preds = model.predict(X_scaled)
is_valid = [1 if p == 1 else 0 for p in preds]

# Outputing results 
df["transaction_id"] = df.index + 1
df["score"] = scores
df["is_valid_prediction"] = is_valid

output = df[["transaction_id", "score", "is_valid_prediction"] + features]
output.to_excel(OUTPUT_FILE, index=False)

print(f"✅ Scoring completed. Results saved to '{OUTPUT_FILE}'")
