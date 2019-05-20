rm(list=ls())
library(pacman)
pacman::p_load(tidyverse,
               tidytext,
               ggplot2,
               scales,
               data.table,
               hunspell,
               wordcloud,
               reshape2,
               topicmodels)

report <- data.table::fread("https://github.com/gadenbuie/mueller-report/raw/36fbb136a2a508c812db8773e9342b7a55204b20/mueller_report.csv",
                            data.table = FALSE)

content <- report %>%
            filter(page >= 9, !is.na(text)) %>%
            rowwise() %>%
            mutate(num_mispelled_words = length(hunspell(text)[[1]]),
                   num_words = length(str_split(text, " ")[[1]]),
                   perc_misspelled = num_mispelled_words/num_words) %>%
            filter(perc_misspelled <= 0.5) %>%
            select(-num_mispelled_words, -num_words)

content <- content %>% 
  unnest_tokens(text, text, token = "lines")

tidy_content <- content %>%
                  unnest_tokens(word, text) %>%
                  anti_join(stop_words)

tidy_content %>% 
  mutate(word = str_extract(word, "[a-z']+")) %>%
  filter(!is.na(word)) %>% 
  count(word, sort = TRUE) %>%
  filter(str_length(word) > 1, n > 400) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_segment(aes(x=word, xend=word, y=0, yend=n), color="skyblue", size = 1) +
  geom_point(color="blue", size = 4, alpha = 0.6) +
  coord_flip() +
  theme_bw() +
  theme(panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank(),
        legend.position="none") +
  labs(x = "",
       y = "Number of Occurences",
       title = "Most popular words from the Mueller Report",
       subtitle = "Words Occuring more than 400 times",
       caption = "Based on data from the Mueller Report")


# Sentiment analysis
afinn <- tidy_content %>%
          inner_join(get_sentiments("afinn")) %>%
          group_by(index = page %/% 20) %>%
          summarise(sentiment = sum(score)) %>%
          mutate(method = "AFINN")

bing_and_nrc <- bind_rows(tidy_content %>%
                            inner_join(get_sentiments("bing")) %>%
                            mutate(method = "Bing et al."),
                          tidy_content %>%
                            inner_join(get_sentiments("nrc") %>%
                                        filter(sentiment %in% c("positive", "negative"))) %>%
                            mutate(method = "NRC")) %>%
                count(method, index = page %/% 20, sentiment) %>%
                spread(sentiment, n, fill = 0) %>%
                mutate(sentiment = positive - negative)

bind_rows(afinn,
          bing_and_nrc) %>%
  ggplot(aes(index, sentiment, fill=method)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~method, ncol = 1, scales = "free_y") +
  theme_bw()

## Most common positive and negative words
bing_word_counts <- tidy_content %>%
                      inner_join(get_sentiments("bing")) %>%
                      count(word, sentiment, sort = TRUE) %>%
                      ungroup()

bing_word_counts %>% 
  group_by(sentiment) %>%
  top_n(15) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip() +
  theme_bw()

## Wordclouds
tidy_content %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

tidy_content %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#D95F02", "#1B9E77"),
                   max.words = 100)
  
## Beyond just words
# pnp_sentences <- content %>%
#                   unnest_tokens(sentence, text, token = "sentences")

bingnegative <- get_sentiments("bing") %>%
                  filter(sentiment == "negative")

wordcounts <- tidy_content %>%
              group_by(page) %>%
              summarise(words = n())

## Highest ratio of negative words that has >= 100 words on a page
tidy_content %>%
  semi_join(bingnegative) %>%
  group_by(page) %>%
  summarise(negativewords = n()) %>%
  left_join(wordcounts, by = c("page")) %>%
  mutate(ratio = negativewords/words) %>%
  ungroup() %>%
  filter(words >= 100) %>%
  arrange(desc(ratio)) 


# TFIDF (Term Frequency Inverse Document Frequency)
tidy_content %>%
  count(word, page,  sort = TRUE) %>%
  bind_tf_idf(word, page, n) %>%
  arrange(desc(tf_idf)) %>%
  top_n(15) %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(word, tf_idf)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  coord_flip() +
  theme_bw()


# bigrams
bigram_tf_idf <- content %>%
                  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
                  separate(bigram, c("word1", "word2"), sep = " ") %>%
                  filter(!word1 %in% stop_words$word) %>%
                  filter(!word2 %in% stop_words$word) %>%
                  unite(bigram, word1, word2, sep = " ") %>%
                  count(page, bigram) %>%
                  bind_tf_idf(bigram, page, n) %>%
                  arrange(desc(tf_idf))

# Top bigram tf-idf
bigram_tf_idf %>%
  top_n(15) %>%
  mutate(bigram = reorder(bigram, tf_idf)) %>%
ggplot(aes(x = bigram, y = tf_idf)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf of bigram") +
  coord_flip() +
  theme_bw()


# Topic modeling
dtm_content <- tidy_content %>%
  count(word, page, sort = TRUE) %>%
  rename(count = n) %>% 
  cast_dtm(page, word, count)


lda_model <- LDA(dtm_content, k = 2, control = list(seed = 1234))

lda_topics <- tidy(lda_model, matrix = "beta")

lda_top_terms <- lda_topics %>%
                  group_by(topic) %>%
                  top_n(10, beta) %>%
                  ungroup() %>%
                  arrange(topic, -beta)

# 2 topics
lda_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  theme_bw()

# 2 topics based on the difference b/w topic 1 and topic 2
beta_spread <- lda_topics %>%
                mutate(topic = paste0("topic", topic)) %>%
                spread(topic, beta) %>%
                filter(topic1 > .001 | topic2 > .001) %>%
                mutate(log_ratio = log2(topic2/topic1)) %>%
                mutate(direction_ind = ifelse(log_ratio > 0, 1, 0))

# Top 15 words
rbind(beta_spread %>%
        filter(direction_ind == 1) %>%
        arrange(desc(log_ratio)) %>%
        head(15),
      beta_spread %>%
        filter(direction_ind == 0) %>%
        arrange(log_ratio) %>%
        head(15)) %>% 
     mutate(term = reorder(term, log_ratio)) %>%
ggplot(aes(x = term, y = log_ratio)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  theme_bw() +
  labs(x = "Log2 ratio of beta in topic 2 / topic 1")
  
  
  
