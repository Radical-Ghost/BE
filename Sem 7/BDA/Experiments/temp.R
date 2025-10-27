# Step 0: Load libraries
library(ggplot2)
library(syuzhet)
library(wordcloud)
library(dplyr)
library(RColorBrewer)

# Step 1: Read CSV
tweets_df <- read.csv("Tweets.csv", stringsAsFactors = FALSE)

# Step 2: Extract text column
tweets_text <- tweets_df$text

# Step 3: Sentiment analysis
sentiments <- get_nrc_sentiment(tweets_text)
sentiment_scores <- colSums(sentiments)

# Step 4: Bar plot of sentiment distribution
barplot(sentiment_scores,
        col = rainbow(10),
        las = 2,
        main = "Sentiment Distribution for Tweets Dataset")

# Step 5: Positive vs Negative Pie chart
pos_neg <- c(
  Positive = sum(sentiments$positive),
  Negative = sum(sentiments$negative)
)
pie(pos_neg, labels = names(pos_neg), col = c("green", "red"),
    main = "Positive vs Negative Tweets")

# Step 6: Word Cloud
words <- tolower(tweets_text)
wordcloud(words, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))
