import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp.vocab['wolf'].vector
word2 = nlp.vocab['dog'].vector
word3 = nlp.vocab['cat'].vector
from scipy import spatial
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
new_vector = word1 - word2 + word3
computed_similarities = []
for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
print([w[0].text for w in computed_similarities[:10]])
def vector_math(a,b,c):
    new_vector = nlp.vocab[a].vector - nlp.vocab[b].vector + nlp.vocab[c].vector
    computed_similarities = []

    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity = cosine_similarity(new_vector, word.vector)
                    computed_similarities.append((word, similarity))

    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

    return [w[0].text for w in computed_similarities[:10]]
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
review = input("Enter the review about the lecture ")
sid.polarity_scores(review)
def review_rating(string):
    scores = sid.polarity_scores(string)
    if scores['compound'] == 0:
        return 'Neutral'
    elif scores['compound'] > 0:
        return 'Positive'
    else:
        return 'Negative'
print("the review is classified as")
print(review_rating(review))
