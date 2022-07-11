import numpy as np
from math import asin, atan, cos, radians, sin, sqrt, tan
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
import pandas as pd
import sparse_dot_topn.sparse_dot_topn as ct
from typing import Union, Dict

# ///////////////////////////////////////////////////////////////////////////
# //////// NUMERICAL DISTANCE MATCHING ALGORITHMS ///////////////////////////
# ///////////////////////////////////////////////////////////////////////////

class DistanceMatcher:
    """Base class for distance matching.
    """

    def distance(self, a: np.ndarray, b: np.ndarray):
        return np.linalg.norm(a - b)

    def distance_matrix(self, a: np.ndarray, b: np.ndarray):
        result = []
        for i in a:
            row = []
            for j in b:
                row.append(self.distance(i, j))
            result.append(row)
        return result


class GeoMatcher(DistanceMatcher):
    def distance(self, a: np.ndarray, b: np.ndarray):
        """Calculate great circle distance between two points in a sphere,
        given longitudes and latitudes https://en.wikipedia.org/wiki/Haversine_formula

        We know that the globe is "sort of" spherical, so a path between two points
        isn't exactly a straight line. We need to account for the Earth's curvature
        when calculating distance from point A to B. This effect is negligible for
        small distances but adds up as distance increases. The Haversine method treats
        the earth as a sphere which allows us to "project" the two points A and B
        onto the surface of that sphere and approximate the spherical distance between
        them. Since the Earth is not a perfect sphere, other methods which model the
        Earth's ellipsoidal nature are more accurate but a quick and modifiable
        computation like Haversine can be handy for shorter range distances.

        Args:
            a (np.ndarray): A numpy array with a[0] = latitude and a[1] = longtidue.
            b (np.ndarray): A numpy array with a[0] = latitude and a[1] = longtidue.
        
        Returns:
            geographical distance between two points in metres
            
        """
        # CONSTANTS per WGS84 https://en.wikipedia.org/wiki/World_Geodetic_System
        # Distance in metres(m)
        AXIS_A = 6378137.0
        AXIS_B = 6356752.314245
        RADIUS = 6378137
        # Equation parameters
        # Equation https://en.wikipedia.org/wiki/Haversine_formula#Formulation
        lat1, lon1, lat2, lon2 = a[0], a[1], b[0], b[1]
        flattening = (AXIS_A - AXIS_B) / AXIS_A
        phi_1 = atan((1 - flattening) * tan(radians(lat1)))
        phi_2 = atan((1 - flattening) * tan(radians(lat2)))
        lambda_1 = radians(lon1)
        lambda_2 = radians(lon2)
        # Equation
        sin_sq_phi = sin((phi_2 - phi_1) / 2)
        sin_sq_lambda = sin((lambda_2 - lambda_1) / 2)
        # Square both values
        sin_sq_phi *= sin_sq_phi
        sin_sq_lambda *= sin_sq_lambda
        h_value = sqrt(sin_sq_phi + (cos(phi_1) * cos(phi_2) * sin_sq_lambda))
        return 2 * RADIUS * asin(h_value)


# ///////////////////////////////////////////////////////////////////////////
# //////// FUZZYMATCHING ALGORITHMS /////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////

class StringMatcherXL():
    """A method to fuzzy match strings for XL datasets.

    Uses cosine similairty and trigram tokenisation of characters
    to reduce the fuzzy matching caculations to an optimised matrix 
    calculation.
    """
    
    def __init__(self, source_names, target_names):
        self.source_names = source_names
        self.target_names = target_names
        self.ct_vect      = None
        self.tfidf_vect   = None
        self.vocab        = None
        self.sprse_mtx    = None
        
        
    def tokenize(self, analyzer='char_wb', n=3):

        """Tokenizes the list of strings, based on the selected analyzer

        Args:
            analyzer (str)/: Type of analyzer ('char_wb', 'word'). Default is trigram
            n (int): If using n-gram analyzer, the gram length. Default is 3.
        """
        self.ct_vect = CountVectorizer(analyzer=analyzer, ngram_range=(n, n))
        self.vocab   = self.ct_vect.fit(self.source_names + self.target_names).vocabulary_
        self.tfidf_vect  = TfidfVectorizer(vocabulary=self.vocab, analyzer=analyzer, ngram_range=(n, n))
        
        
    def match(self, ntop=1, lower_bound=0, output_fmt='df') -> Union[Dict, pd.DataFrame]:
        """Main match function. Default settings return only the top candidate for every source string.

        Args:
            ntop (int, optional): The number of top-n candidates that should be returned. Defaults to 1.
            lower_bound (int, optional): The lower-bound threshold for keeping a candidate, between 0-1. Defaults to 0.
            output_fmt (str, optional): The output format. Either dataframe ('df') or dict ('dict'). Defaults to 'df'.

        Returns:
            Union[Dict, pd.DataFrame]: The resulting matches in the output format.
        """
        self._awesome_cossim_top(ntop, lower_bound)
        if output_fmt == 'df':
            match_output = self._make_matchdf()
        elif output_fmt == 'dict':
            match_output = self._make_matchdict()
        return match_output
        
        
    def _awesome_cossim_top(self, ntop, lower_bound):
        ''' https://gist.github.com/ymwdalex/5c363ddc1af447a9ff0b58ba14828fd6#file-awesome_sparse_dot_top-py '''
        # To CSR Matrix, if needed
        A = self.tfidf_vect.fit_transform(self.source_names).tocsr()
        B = self.tfidf_vect.fit_transform(self.target_names).transpose().tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M * ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        self.sprse_mtx = csr_matrix((data,indices,indptr), shape=(M,N))
    
    
    def _make_matchdf(self):
        ''' Build dataframe for result return '''
        cx = self.sprse_mtx.tocoo()
        match_list = []
        for row,col,val in zip(cx.row, cx.col, cx.data):
            match_list.append((row, self.source_names[row], col, self.target_names[col], val))

        colnames = ['Row Idx', 'Title', 'Candidate Idx', 'Candidate Title', 'Score']
        match_df = pd.DataFrame(match_list, columns=colnames)

        return match_df

    
    def _make_matchdict(self):
        ''' Build dictionary for result return '''
        cx = self.sprse_mtx.tocoo()
        match_dict = {}
        for row,col,val in zip(cx.row, cx.col, cx.data):
            if match_dict.get(row):
                match_dict[row].append((col,val))
            else:
                match_dict[row] = [(col, val)]

        return match_dict   