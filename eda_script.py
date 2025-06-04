import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from wordcloud import WordCloud # Optional, uncomment if you install wordcloud

# --- Configuration ---
DATA_FILE = 'data_science_jobs.csv'
LOCATION_FILTER = 'Philadelphia, PA'
OUTPUT_PLOT_FILE = 'top_skills_philadelphia.png'
OUTPUT_WORDCLOUD_FILE = 'wordcloud_philadelphia.png' # Optional

# --- Define Keywords/Skills to Search For ---
# These are common data science skills. You can expand this list!
SKILLS = [
    'Python', 'SQL', 'R', 'Machine Learning', 'Statistics', 'Tableau',
    'Power BI', 'Excel', 'Spark', 'AWS', 'Azure', 'GCP', 'Deep Learning',
    'NLP', 'Data Visualization', 'Big Data', 'A/B Testing', 'Regression',
    'Classification', 'Keras', 'TensorFlow', 'PyTorch', 'Scikit-learn',
    'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Git', 'Docker'
]

def perform_eda():
    """
    Performs a basic Exploratory Data Analysis on data science job listings.
    Filters by location, extracts key skills using regex, and visualizes them.
    """
    print(f"Starting EDA for Data Science Jobs in {LOCATION_FILTER}...")

    try:
        # Load the dataset
        df = pd.read_csv(DATA_FILE)
        print(f"Successfully loaded {len(df)} job listings from {DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please make sure the CSV file is in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    # Filter jobs for the specified location
    philadelphia_jobs = df[df['Location'].str.contains(LOCATION_FILTER, case=False, na=False)].copy()

    if philadelphia_jobs.empty:
        print(f"No job listings found for {LOCATION_FILTER}. Exiting.")
        return

    print(f"Found {len(philadelphia_jobs)} job listings in {LOCATION_FILTER}.")

    # Combine relevant text fields for skill extraction
    # Using .fillna('') to handle potential NaN values in descriptions
    philadelphia_jobs['Combined_Text'] = philadelphia_jobs['Job Title'].fillna('') + ' ' + \
                                         philadelphia_jobs['Job Description'].fillna('')

    # Extract skills using regex
    skill_counts = Counter()
    for text in philadelphia_jobs['Combined_Text']:
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        for skill in SKILLS:
            # Use regex to find whole words, ignoring case
            # \b ensures whole word match (e.g., 'R' won't match 'car')
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                skill_counts[skill] += 1

    # Convert skill counts to a Pandas Series for easier manipulation
    skills_series = pd.Series(skill_counts).sort_values(ascending=False)

    if skills_series.empty:
        print("No relevant skills found in job descriptions. Exiting.")
        return

    print("\nTop 10 Skills:")
    print(skills_series.head(10))

    # --- Visualization: Bar Chart of Top Skills ---
    plt.figure(figsize=(12, 7))
    skills_series.head(15).plot(kind='bar', color='skyblue') # Plot top 15 skills
    plt.title(f'Top Skills for Data Science Jobs in {LOCATION_FILTER}')
    plt.xlabel('Skills')
    plt.ylabel('Number of Mentions')
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"\nBar chart saved to {OUTPUT_PLOT_FILE}")
    # plt.show() # Uncomment to display plot immediately (might not work in all environments)

    # --- Optional: Word Cloud Generation ---
    # To use this, you need to install the 'wordcloud' library: pip install wordcloud
    # If you don't want a word cloud, you can comment out this entire section.
    try:
        all_job_descriptions = " ".join(philadelphia_jobs['Combined_Text'].dropna())
        if all_job_descriptions:
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  collocations=False, # Set to False to avoid phrases like "data science" as one word
                                  min_font_size=10).generate(all_job_descriptions)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off') # Hide axes
            plt.title(f'Word Cloud of Job Descriptions in {LOCATION_FILTER}')
            plt.savefig(OUTPUT_WORDCLOUD_FILE)
            print(f"Word cloud saved to {OUTPUT_WORDCLOUD_FILE}")
            # plt.show()
        else:
            print("No job descriptions available for word cloud generation.")
    except ImportError:
        print("\n'wordcloud' library not found. Skipping word cloud generation.")
        print("To enable word cloud, install it: pip install wordcloud")
    except Exception as e:
        print(f"An error occurred during word cloud generation: {e}")


if __name__ == "__main__":
    perform_eda()
