from flask import Flask, render_template, request
from gramformer import Gramformer
from transformers import pipeline
from spellchecker import SpellChecker
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

app = Flask(__name__)

# Initialize Gramformer (grammar correction)
gf = Gramformer(models=1)

# Initialize SpellChecker
spell = SpellChecker()

# Initialize paraphrasing model
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def spell_correct(text):
    corrected_words = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word if corrected_word else word)
    return " ".join(corrected_words)

def correct_grammar(text):
    corrected_sentences = gf.correct(text)
    if corrected_sentences:
        return list(corrected_sentences)[0]
    return text

def paraphrase(text):
    result = paraphraser(f"paraphrase: {text}", max_length=256, num_return_sequences=1, do_sample=True)[0]['generated_text']
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    spell_corrected = ""
    grammar_corrected = ""
    paraphrased = ""

    if request.method == "POST":
        original_text = request.form["input_text"]
        spell_corrected = spell_correct(original_text)
        grammar_corrected = correct_grammar(spell_corrected)
        paraphrased = paraphrase(grammar_corrected)

    return render_template("index.html",
                           original_text=original_text,
                           spell_corrected=spell_corrected,
                           grammar_corrected=grammar_corrected,
                           paraphrased=paraphrased)

if __name__ == "__main__":
    app.run(debug=True)
