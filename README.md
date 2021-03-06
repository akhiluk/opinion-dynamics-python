# Sentiment Dynamics Visualizer built with Python and Flask

### About

Accepts e-books and links to YouTube videos as input from the user. Produces a chart showing change in sentiment and opinions over time. Uses the DeepSpeech speech-to-text library built by Mozilla.

### Quickstart

1. Clone the repo.

2. Run `$ python init.py` and wait for the DeepSpeech model to be downloaded and extracted.

3. Run `$ pip install -r requirements.txt` to install all the required libraries.

4. Run `$ python app.py` and navigate to [localhost:5000](http://127.0.0.1:5000) to view the app.

5. Use the provided 'sample_ebook.epub' or any e-book (in EPUB format) of your choice for e-book analysis.

### Closing words

A rudimentary web app built with Flask as a weekend project. Avenues for improvement include porting the project to Streamlit and making the charts responsive to mouse hovers.