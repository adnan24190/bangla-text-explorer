import pandas as pd
import matplotlib.pyplot as plt

# 1. Ingestion: Load your cleanly filtered matrix back into RAM
df = pd.read_csv("processed_test_set.csv")

# 2. Visualization: Command pandas to draw a histogram with 20 bins (bars)
df['prompt_length'].plot(kind='hist', bins=20, title='Distribution of Bengali Prompt Lengths', color='blue')

# 3. Export: Save the drawing from temporary RAM to a permanent image file
plt.savefig('length_chart.png')
print("Visualization complete. Chart saved as length_chart.png!")

# 4. Data Cleaning: Drop any row containing 'svg' image code
clean_df = df[~df['prompt_bn'].str.contains('svg', na=False)]

print("Rows before cleaning:", len(df))
print("Rows after cleaning:", len(clean_df))

# 5. Final Export
clean_df.to_csv("final_clean_dataset.csv", index=False)
print("Final cleaned dataset saved as final_clean_dataset.csv!")