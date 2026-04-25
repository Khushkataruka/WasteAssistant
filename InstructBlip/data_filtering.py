import pandas as pd
import json

# Load CSV
df = pd.read_csv("vqa_generated.csv")

# Sort to ensure consistency
df = df.sort_values(by=["image_path", "question"])

# Get the global list of questions (first 15 should be enough)
questions = df["question"].drop_duplicates().tolist()

# Group by image_path and label
grouped = df.groupby(["image_path", "label"])

# Build structured data
data = []
for (image_path, label), group in grouped:
    answers = group.sort_values("question")["answer"].tolist()
    data.append({
    "image_path": image_path,
    "label": label,
    "answers": answers
    })

# Final JSON format
final_output = {
    "questions": questions,
    "data": data
}

# Save to file
with open("vqa_final.json", "w") as f:
    json.dump(final_output, f, indent=2)
