import os
import pandas as pd
from datasets import load_dataset
from langfuse import Langfuse

# Set Langfuse credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-0b384bde-303f-4df6-9066-9cb1e4e30e94"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-978aac65-5019-4092-9bfc-63bf75aaf98e"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# Initialize Langfuse client
langfuse = Langfuse()

# Load English-Spanish translation dataset from Hugging Face
print("Loading dataset from Hugging Face...")
dataset = load_dataset("opus_books", "en-es", split="train[:50]")
df = pd.DataFrame(dataset)

# Display sample data
print("\nFirst few rows of the dataset:")
for i, row in df.head().iterrows():
    print(f"English: {row['translation']['en']}")
    print(f"Spanish: {row['translation']['es']}")
    print("-" * 50)

# Define Langfuse dataset name
langfuse_dataset_name = "en-es-translation-benchmark"

# Create a dataset in Langfuse
print(f"\nCreating dataset '{langfuse_dataset_name}' in Langfuse...")
try:
    langfuse.create_dataset(
        name=langfuse_dataset_name,
        description="English-Spanish translation dataset from opus_books",
        metadata={
            "source": "opus_books",
            "language_pair": "en-es",
            "type": "benchmark"
        }
    )
    print("Dataset created successfully.")
except Exception as e:
    print(f"Note: {e}")
    print("Dataset might already exist, continuing with item creation...")

# Add dataset items from Hugging Face dataset to Langfuse
print("\nAdding items to Langfuse dataset...")
count = 0
for idx, row in df.iterrows():
    # Create dataset item with English as input and Spanish as expected output
    langfuse.create_dataset_item(
        dataset_name=langfuse_dataset_name,
        input={"text": row["translation"]["en"]},
        expected_output={"text": row["translation"]["es"]}
    )
    count += 1
    
    # Progress indicator
    if count % 10 == 0:
        print(f"Added {count} items...")
        
    if idx >= 49:  # For this example, we upload only the first 50 items
        break

# Ensure all data is sent to Langfuse
langfuse.flush()
print(f"\nSuccessfully uploaded {count} translation examples to Langfuse dataset '{langfuse_dataset_name}'")
print("You can now use this dataset for evaluating your translation agent.")