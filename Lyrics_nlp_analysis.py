from collections import Counter

# Store some lyrics in a variable (example lyrics)
lyrics = """I see trees of green
Red roses too
I see them bloom
For me and you
And I think to myself
What a wonderful world"""

# Break down lyrics into words using split("\n")
lines = lyrics.split("\n")

# Create a list to hold all the words
words = []

# Split each line into words and extend the words list
for line in lines:
    words.extend(line.split())

# Count the frequency of every word in the text using Counter
word_count = Counter(words)

# Print the word frequency
for word, count in word_count.items():
    print(f"{word}: {count}")
