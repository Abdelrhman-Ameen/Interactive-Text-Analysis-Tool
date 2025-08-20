import streamlit as st
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import language_tool_python
import subprocess  # For checking Java
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")


# Check if Java is installed
def check_java():
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        version_line = result.stderr.splitlines()[0]
        version = float(version_line.split()[2].replace('"', '').split('.')[0])
        return version >= 17
    except:
        return False


# Text Cleaning Function
def clean_text(text):
    """
    Cleans the input text by lowercasing, removing punctuation, tokenizing,
    and removing stop words.

    Args:
        text (str): The input string to clean.

    Returns:
        list: A list of cleaned and filtered tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


# Grammar and Style Analysis Function
def check_grammar_and_style(text):
    if not text.strip() or len(text.split()) < 2:
        return ["Text is too short or empty. Please provide a longer text for analysis."]

    if not check_java():
        return ["Error: Java 17 or higher is required. Install it from https://www.oracle.com/java/ and try again."]

    try:
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)
        if not matches:
            return ["No grammar or style issues found. Your text is clean!"]
        return [{"error": match.message, "suggestion": match.replacements, "context": match.context} for match in
                matches]
    except Exception as e:
        return [f"Error initializing LanguageTool: {str(e)}. Check Java, permissions, or reinstall the library."]


# NLP Processing Functions
def get_wordnet_pos(nltk_tag):
    """
    Maps NLTK POS tags to WordNet POS tags for lemmatization.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def process_text_nlp(tokens):
    """
    Performs POS tagging and lemmatization on a list of tokens.

    Args:
        tokens (list): A list of cleaned tokens.

    Returns:
        list: A list of tuples with original token, POS tag, and lemmatized form.
    """
    lemmatizer = WordNetLemmatizer()
    pos_tagged_tokens = nltk.pos_tag(tokens)
    processed_tokens = []
    for token, tag in pos_tagged_tokens:
        wordnet_pos = get_wordnet_pos(tag)
        lemmatized_token = lemmatizer.lemmatize(token, pos=wordnet_pos)
        processed_tokens.append((token, tag, lemmatized_token))
    return processed_tokens


# Sentiment and Tone Analysis Function
def analyze_sentiment_and_tone(text):
    """
    Analyzes sentiment and tone using TextBlob and VADER, with adjustment suggestions.

    Args:
        text (str): The input string to analyze.

 exchanges:
        dict: Polarity (with description), subjectivity, tone, VADER description, and suggestions.
    """
    # TextBlob for polarity and subjectivity
    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity

    # Convert polarity to description
    if polarity > 0.5:
        polarity_desc = "Very Positive"
    elif polarity > 0.1:
        polarity_desc = "Positive"
    elif polarity > -0.1:
        polarity_desc = "Neutral"
    elif polarity > -0.5:
        polarity_desc = "Negative"
    else:
        polarity_desc = "Very Negative"

    # VADER for tone analysis
    sid = SentimentIntensityAnalyzer()
    vader_scores = sid.polarity_scores(text)
    compound = vader_scores['compound']

    # Convert VADER compound score to a clear description
    if compound > 0.5:
        vader_desc = "The text feels positive and upbeat!"
    elif compound > -0.1:
        vader_desc = "The text is mostly neutral, with a balanced tone."
    else:
        vader_desc = "The text feels negative or critical."

    # Suggestions based on sentiment and tone
    suggestions = []
    if polarity < 0:
        suggestions.append("Try using positive words like 'great' or 'wonderful' to improve the tone.")
    if subjectivity > 0.6:
        suggestions.append("Consider adding factual statements to make the text more objective.")
    if len(text.split()) > 20 and textstat.flesch_reading_ease(text) < 60:
        suggestions.append("Shorten sentences or use simpler words to improve readability.")

    return {
        "polarity": polarity,
        "polarity_desc": polarity_desc,
        "subjectivity": subjectivity,
        "tone": "Positive" if compound > 0.5 else "Negative" if compound < -0.5 else "Neutral",
        "vader_desc": vader_desc,  # New field for user-friendly description
        "vader_scores": vader_scores,  # Keep raw scores for internal use
        "suggestions": suggestions
    }


# Readability Scores Function
def calculate_readability(text):
    """
    Calculates readability scores and provides simplification suggestions.

    Args:
        text (str): The input string to analyze.

    Returns:
        dict: Readability scores and simplification suggestions.
    """
    scores = {
        "ease_of_reading_score": textstat.flesch_reading_ease(text),
        # Flesch Reading Ease score (0-100, higher is easier)
        "required_education_level": textstat.flesch_kincaid_grade(text),
        # Education level required (e.g., grade 5 or 8) per Flesch-Kincaid
        "text_complexity_level": textstat.gunning_fog(text),  # Gunning Fog Index (measures text complexity)
        "reading_difficulty_rating": textstat.smog_index(text),  # SMOG Index (assesses reading difficulty)
        "simplicity_score": textstat.coleman_liau_index(text),  # Coleman-Liau Readability score
        "easy_to_read_score": textstat.automated_readability_index(text),
        # Automated Readability Index (ARI) score
        "clarity_score": textstat.dale_chall_readability_score(text),  # Dale-Chall Readability score
        "number_of_hard_words": textstat.difficult_words(text),  # Number of difficult words in the text
        "writing_simplicity_score": textstat.linsear_write_formula(text),  # Linsear Write formula (evaluates writing ease)
        "text_difficulty_level": textstat.text_standard(text),  # Overall text level assessment
        "total_syllable_number": textstat.syllable_count(text),  # Total number of syllables in the text
        "number_of_unique_words": textstat.lexicon_count(text),  # Number of unique words
        "total_number_of_sentences": textstat.sentence_count(text),  # Total number of sentences
        "total_character_count": textstat.char_count(text),  # Total number of characters (including spaces)
        "total_letter_count": textstat.letter_count(text),  # Total number of letters (excluding spaces)
        "number_of_long_words": textstat.polysyllabcount(text)
        # Number of polysyllabic words (words with multiple syllables)
    }

    # Simplification suggestions
    simplification_suggestions = []
    if scores["ease_of_reading_score"] < 60:
        simplification_suggestions.append(
            "The text is hard to read. Try dividing long sentences or using simpler words, like replacing 'cellulose' with 'material'.")
    if scores["total_number_of_sentences"] > 0 and scores["number_of_unique_words"] / scores["total_number_of_sentences"] > 20:
        simplification_suggestions.append("The sentences are too long. Try splitting them into shorter ones for better understanding.")
    if scores["number_of_hard_words"] > 5:
        simplification_suggestions.append("The text has too many difficult words. Use simpler alternatives to help readers.")

    return {"scores": scores, "simplification_suggestions": simplification_suggestions}


# Text Simplification Function
def simplify_text(text):
    """
    Simplifies text by splitting long sentences and suggesting simpler synonyms.

    Args:
        text (str): The input string to simplify.

    Returns:
        str: Simplified text.
    """
    sentences = sent_tokenize(text)
    simplified_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > 20:
            mid = len(sentence.split()) // 2
            words = sentence.split()
            simplified_sentences.append(" ".join(words[:mid]) + ".")
            simplified_sentences.append(" ".join(words[mid:]) + ".")
        else:
            simplified_sentences.append(sentence)

    simplified_text = " ".join(simplified_sentences)
    complex_words = {"cellulose": "material", "fragile": "weak", "imagine": "think"}
    for complex_word, simple_word in complex_words.items():
        simplified_text = simplified_text.replace(complex_word, simple_word)

    return simplified_text


# Paraphrasing Function
def paraphrase_text(text, model_name="t5-small", num_beams=4, num_return_sequences=1):
    """
    Paraphrases the input text using a pre-trained sequence-to-sequence model.

    Args:
        text (str): The input string to paraphrase.
        model_name (str): The name of the pre-trained model to use.
        num_beams (int): Number of beams for beam search.
        num_return_sequences (int): Number of paraphrased sequences to return.

    Returns:
        list: A list of paraphrased strings.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=512,
            early_stopping=True
        )
        paraphrased_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return paraphrased_texts
    except Exception as e:
        return [f"Error in paraphrasing: {str(e)}"]


# Plagiarism Check Function
def check_plagiarism(input_text, corpus, threshold=0.7):
    """
    Checks for plagiarism by comparing the input text against a corpus using sentence embeddings.

    Args:
        input_text (str): The text to check for plagiarism.
        corpus (list): A list of strings to compare against.
        threshold (float): The cosine similarity score threshold for considering a match.

    Returns:
        list: A list of dictionaries with matching texts and similarity scores.
    """
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        input_embedding = model.encode(input_text, convert_to_numpy=True)
        corpus_embeddings = model.encode(corpus, convert_to_numpy=True)
        similarity_scores = cosine_similarity(input_embedding.reshape(1, -1), corpus_embeddings)
        similarity_scores = similarity_scores[0]
        potential_matches = []
        for i, score in enumerate(similarity_scores):
            if score >= threshold:
                potential_matches.append({
                    "corpus_text": corpus[i],
                    "similarity_score": score,
                    "citation_suggestion": f"Cite as: Unknown Author. (2025). Text from corpus. Retrieved from internal database."
                })
        return potential_matches
    except Exception as e:
        return [f"Error in plagiarism check: {str(e)}"]


# Citation Generation Function
def generate_citation(source_text, author="Unknown Author", title="Text from Corpus", year="2025"):
    """
    Generates a basic APA-style citation for a source text.

    Args:
        source_text (str): The source text to cite.
        author (str): The author name.
        title (str): The title of the source.
        year (str): The publication year.

    Returns:
        str: APA-style citation.
    """
    return f"{author}. ({year}). {title}. [Text]. Retrieved from internal database."


# Streamlit App
st.title("Interactive Text Analysis and Content Improvement Tool")

st.write("Enter your text below to analyze its quality, readability, style, and originality.")

user_text = st.text_area("Enter the text you want to analyze:", height=200)

if st.button("Analyze"):
    if user_text:
        st.header("Analysis Results")

        # Grammar and Style Analysis
        st.subheader("Grammar and Style Analysis")
        grammar_results = check_grammar_and_style(user_text)
        if isinstance(grammar_results, list) and grammar_results and isinstance(grammar_results[0], dict):
            for result in grammar_results:
                st.write(f"- **Error**: {result['error']}")
                st.write(f"  **Context**: {result['context']}")
                st.write(f"  **Suggestion**: {result['suggestion']}")
        else:
            st.write(grammar_results[0])  # Show message (e.g., "No issues" or error)

        # Text Cleaning
        st.subheader("Cleaned Tokens")
        cleaned_tokens = clean_text(user_text)
        st.write(cleaned_tokens)

        # NLP Processing
        st.subheader("NLP Processing Results")
        processed_nlp = process_text_nlp(cleaned_tokens)
        df = pd.DataFrame(processed_nlp, columns=["Original Token", "POS Tag", "Lemmatized Token"])
        st.dataframe(df)

        # Sentiment and Tone Analysis
        st.subheader("Sentiment and Tone Analysis")
        sentiment_results = analyze_sentiment_and_tone(user_text)
        st.write(f"**Polarity**: {sentiment_results['polarity_desc']} (Score: {sentiment_results['polarity']:.4f})")
        st.write(f"**Subjectivity**: {sentiment_results['subjectivity']:.4f} (0 = Factual, 1 = Opinion)")
        st.write(f"**Tone**: {sentiment_results['tone']}")
        st.write(f"**Overall Feeling**: {sentiment_results['vader_desc']}")
        # Optionally, show raw VADER scores for advanced users
        # st.write(f"**VADER Scores**: {sentiment_results['vader_scores']}")
        st.write("**Suggestions**:")
        for suggestion in sentiment_results['suggestions']:
            st.write(f"- {suggestion}")

        # Readability Scores
        st.subheader("Readability Scores")
        readability_results = calculate_readability(user_text)
        st.write("**Scores**:")
        st.json(readability_results["scores"])
        st.write("**Simplification Suggestions**:")
        for suggestion in readability_results["simplification_suggestions"]:
            st.write(f"- {suggestion}")

        # Simplified Text
        st.subheader("Simplified Text")
        simplified_text = simplify_text(user_text)
        st.write(simplified_text)

        # Paraphrasing
        st.subheader("Paraphrasing Results")
        paraphrased_texts = paraphrase_text(user_text)
        for i, para_text in enumerate(paraphrased_texts):
            st.write(f"**Paraphrase {i + 1}**: {para_text}")

        # Plagiarism Check
        st.subheader("Plagiarism Check Results")
        sample_corpus = [
            "Paper is manufactured from trees, which are processed into pulp.",
            "Natural language processing is a field of artificial intelligence.",
            "The quick brown fox jumps over the lazy dog.",
            "Paper is a soft and fragile material made from plant fibers.",
            "Everyone knows that paper comes from trees and is used widely.",
            "Machine learning helps computers understand human language.",
            "Trees are the primary source of paper production.",
            "This sentence is simple and easy to understand."
        ]
        potential_plagiarism = check_plagiarism(user_text, sample_corpus)
        if isinstance(potential_plagiarism, list) and potential_plagiarism and isinstance(potential_plagiarism[0],
                                                                                          dict):
            for match in potential_plagiarism:
                st.write(f"- **Corpus Text**: {match['corpus_text']}")
                st.write(f"  **Similarity Score**: {match['similarity_score']:.4f}")
                st.write(f"  **Citation Suggestion**: {match['citation_suggestion']}")
        else:
            st.write("No potential plagiarism found above the threshold (0.7) or an error occurred.")

    else:
        st.warning("Please enter text to analyze.")