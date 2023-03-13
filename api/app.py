from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import json

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return {'status': 'forbidden','message': 'Why are you here?'}, 403

@app.route('/getAllMovies', methods=['POST'])
def getAllMovies():
    if request.method == 'POST':
        try:
            df = pd.read_csv('allmovies.csv')
            movie_list = df['Movie'].tolist()
            return {
                'status': 'success',
                'data': json.dumps(movie_list)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500
    else:
        return {
            'status': 'error',
            'message': 'Invalid Request'
        }, 400

@app.route('/recommendMovie', methods=['POST'])
def recommendMovie():
    if request.method == 'POST':
        try:
            movie = request.json['movie']
            df = pd.read_csv('finalDataSet.csv')
            features = ['genres', 'keywords', 'actor_1', 'actor_2', 'actor_3', 'director', 'original_language', 'tagline', 'original_title']
            for feature in features:
                df[feature] = df[feature].fillna('')

            combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['actor_1'] + ' ' + df['actor_2'] + ' ' + df['actor_3'] + ' ' + df['director'] + ' ' + df['original_language'] + ' ' + df['tagline'] + ' ' + df['original_title']
            
            cv = TfidfVectorizer()
            feature_vector = cv.fit_transform(combined_features)
            movie_index = df[df['original_title'] == movie].index[0]
            similarity_scores = cosine_similarity(feature_vector[movie_index], feature_vector)
            similar_movies = list(enumerate(similarity_scores[0]))
            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
            top_10_similar_movies = sorted_similar_movies[1:11]
            movie_names = []
            for movie in top_10_similar_movies:
                movieData = {
                    'title': df['original_title'][movie[0]],
                    'overview': df['overview'][movie[0]]
                }
                movie_names.append(movieData)
            return {
                'status': 'success',
                'data': movie_names
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500
    else:
        return {
            'status': 'error',
            'message': 'Invalid Request'
        }, 400

@app.route('/getMovieDetails', methods=['POST'])
def getMovieDetails():
    if request.method == 'POST':
        try:
            movie = request.json['movie']
            df = pd.read_csv('finalDataSet.csv')
            movie_details = df[df['original_title'] == movie]
            # remove nan
            movie_details = movie_details.replace(np.nan, '', regex=True)
            movie_details = movie_details.to_dict()
            return {
                'status': 'success',
                'data': movie_details
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500
    else:
        return {
            'status': 'error',
            'message': 'Invalid Request'
        }, 400

@app.route('/getMovieDetailsByGenre', methods=['POST'])
def getMovieRecommendationsByGenre():
    if request.method == 'POST':
        try:
            genre = request.json['genres']
            language = request.json['languages']
            minYear = request.json['minYear']
            maxYear = request.json['maxYear']
            df = pd.read_csv('finalDataSet.csv')
            features = ['genresLs', 'original_language']
            for feature in features:
                df[feature] = df[feature].fillna('')
            
            myStr = ' '.join(genre)
            movieData = myStr

            cv = TfidfVectorizer()
            feature_vector = cv.fit_transform(df['genresLs'])
            myFeatureVector = cv.transform([movieData])

            similarity_scores = cosine_similarity(myFeatureVector, feature_vector)
            similar_movies = list(enumerate(similarity_scores[0]))
            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
            top_10_similar_movies = []
            for movie in sorted_similar_movies:
                releaseDate = df['release_date'][movie[0]]
                lang = df['original_language'][movie[0]]
                if releaseDate != 'nan' and lang != 'nan':
                    try:
                        releaseYear = int(releaseDate.split('-')[0])
                        if (minYear == None or (releaseYear >= minYear and releaseYear <= maxYear)) and (lang == None or lang in language):
                            top_10_similar_movies.append(movie)
                            if len(top_10_similar_movies) == 10:
                                break
                    except:
                        pass
                    
            movie_names = []
            for movie in top_10_similar_movies:
                movieData = {
                    'title': df['original_title'][movie[0]],
                    'overview': df['overview'][movie[0]]
                }
                movie_names.append(movieData)

            return {
                'status': 'success',
                'data': movie_names
            }, 200
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 500
    else:
        return {
            'status': 'error',
            'message': 'Invalid Request'
        }, 400
    
@app.errorhandler(404)
def not_found(e):
    return {'status': 'Not found','message': 'Looks like you are lost'}, 404

@app.errorhandler(500)
def internal_error(e):
    return {'status': 'Internal Server Error','message': 'Something went wrong'}, 500
if __name__ == '__main__':

    app.run(debug=True)