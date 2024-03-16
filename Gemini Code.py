import csv
import google.generativeai as genai
import re
import random
import time

genai.configure(api_key="You API key")  # Replace with your actual API key

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 5000,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings
)

# Function to clean tweet text
def clean_tweet(tweet):
    # Remove special characters like '#'
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet

# Function to clean response text
def clean_response(response):
    # Remove asterisks and other special characters
    response = re.sub(r'[*]', '', response)
    return response.strip()

# List of prompts
prompts = [
    "Evaluate the sentiment expressed in the tweet: '{tweet}'. Pick one of the sentiment labels: Positive, Extremely positive, Negative, Extremely Negative. Justify your selection with a brief explanation."
]

# Read the CSV file and write annotated data directly to the output CSV file
with open("Corona_NLP_test.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = ["No.", "tweet", "Prompt", "generated annotations", "explanation"]

    with open("corona_NLP_test_annotated.csv", "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader, start=1):
            tweet = row["OriginalTweet"]
            # Clean tweet text
            tweet = clean_tweet(tweet)

            # Check for empty tweets
            if not tweet:
                label = "Error: Empty tweet"
                reason = "Error: Empty tweet"
                writer.writerow({
                    "No.": i,
                    "tweet": row["OriginalTweet"],
                    "Prompt": "N/A",
                    "generated annotations": label,
                    "explanation": reason
                })
                continue  # Skip to the next iteration

            # Randomly select a prompt
            prompt = random.choice(prompts).format(tweet=tweet)

            max_retries = 3
            for retry in range(max_retries):
                try:
                    convo = model.start_chat(history=[])
                    convo.send_message(prompt)

                    if convo and convo.last:
                        response = convo.last.text

                        # Rest of your code for processing the response
                        # ...

                        break  # Break out of the retry loop if successful

                except Exception as e:
                    print(f"Error on attempt {retry + 1}/{max_retries}: {e}")
                    print("Prompt:", prompt)
                    print("Tweet:", tweet)
                    time.sleep(1)  # Add a short delay before retrying

            else:
                print("Max retries reached. Unable to analyze tweet.")
                label = "Error: Unable to analyze tweet"
                reason = "Max retries reached"

            writer.writerow({
                "No.": i,
                "tweet": row["OriginalTweet"],
                "Prompt": prompt,
                "generated annotations": label,
                "explanation": reason
            })
