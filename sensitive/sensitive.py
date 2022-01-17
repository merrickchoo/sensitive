##
## A sense based sentiment analyzer
##
##
## borrows a lot from Vader
##
import sys
import math
import yaml
from importlib.resources import open_text
#import sensitive.wsd
from sensitive.wsd import pos2wn, disambiguate
import wn
from wn.morphy import Morphy

en = wn.Wordnet('omw-en:1.4')
morphy = Morphy(wn)

###
### Changes to the scores based on
### * punctuation  DONE
### * capitalization  DONE
### * intensification
### * negation
### * conjunctions (contrastive)
### * comparatives, superlatives    # DONE, FIXME INCREMENTS
### * morphology based antonyms (hard)
###


###
### Constants
###

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for booster words)
COMP_INCR = 0.1
SUPR_INCR = 0.2

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

### highlighters
### have a score for before and after
### e.g. (but,      0.5, 1.5) #VADER, SOCAL 1, 2
###      (although, 1.0, 0.5) #SOCAL
# yet
# nevertheless
# nonetheless
# even so
# however
# still
# notwithstanding
# despite that
# in spite of that
# for all that
# all the same
# just the same
# at the same time
# be that as it may
# though
# although
# still and all
negators = {'i18282', 
            'i18284', 
            'i18436', #no
            'i2944', # neither
            'i18262', 
            'i18263', # never
            'i12559', 
            'i18285', # none
            'i18280', # not
            'i18281', # nothing ## NOT 'i109082',
            'i18289', # nowhere   # FCB does this help?
            'i18348', # rarely, seldom
            }
# missing contracted auxiliaries and without, nope, uh-uh, uhuh, nor
neg_dict = dict.fromkeys(negators, "-0.74")
boosters = {'i18320', #very
                'i18185', 
                'i18187', 
                'i18188', 
                'i18194', 
                'i18197', 
                'i18198', 
                'i18199', 
                'i18202', 
                'i18207', 
                'i18225', 
                'i18249', 
                'i18250', 
                'i18251', 
                'i18252', 
                'i18259',
                'i18260', 
                'i18270', 
                'i18287', 
                'i18320', 
                'i18321', 
                'i18324', 
                'i18325', 
                'i18344', 
                'i18345', 
                'i18356', 
                'i18357', 
                'i18359', 
                'i18361', 
                'i18368', 
                'i18370', 
                'i18390', 
                'i18410', 
                'i18413', 
                'i18414', 
                'i18424', 
                'i18465', 
                'i18466', 
                'i18474', 
                'i18475', 
                'i18480', 
                'i18481', 
                'i18493', 
                'i18494', 
                'i18495', 
                'i18533', 
                'i18578', 
                'i18629', 
                'i18648', 
                'i18654', 
                'i18656', 
                'i18689', 
                'i18690', 
                'i18691', 
                'i18748', 
                'i18760', 
                'i18762', 
                'i18776', 
                'i18815', 
                'i18846', 
                'i18849', 
                'i18888', 
                'i18889', 
                'i18890', 
                'i18894', 
                'i18907', 
                'i18911', 
                'i19062', 
                'i19066', 
                'i19112', 
                'i19124', 
                'i19125', 
                'i19126', 
                'i19127', 
                'i19128', 
                'i19129', 
                'i19142', 
                'i19207', 
                'i19322', 
                'i19332', 
                'i19334', 
                'i19335', 
                'i19351', 
                'i19359', 
                'i19362', 
                'i19363', 
                'i19364', 
                'i19386', 
                'i19401', 
                'i19433', 
                'i19453', 
                'i19476', 
                'i19486', 
                'i19495', 
                'i19535', 
                'i19587', 
                'i19588', 
                'i19612', 
                'i19619', 
                'i19671', 
                'i19683', 
                'i19685', 
                'i19688', 
                'i19689', 
                'i19690', 
                'i19692', 
                'i19827', 
                'i19843', 
                'i19856', 
                'i19857', 
                'i20139', 
                'i20140', 
                'i20206', 
                'i20266', 
                'i20267', 
                'i20268', 
                'i20361', 
                'i20362', 
                'i20446', 
                'i20602', 
                'i20603', 
                'i20604', 
                'i20605', 
                'i20659', 
                'i20673', 
                'i20733', 
                'i20740', 
                'i20795', 
                'i20807', 
                'i20968', 
                'i21036', 
                'i21086', 
                'i21281', 
                'i21647', 
                'i21657', 
                'i21666', 
                'i21667', 
                'i21670', 
                'i21714', 
                'i21720',
                'i18163', 
               'i18581', #barely
               'i78', 
               'i1206', 
               'i1265', 
               'i1845', 
               'i2472', 
               'i3300', 
               'i3912', 
               'i4698', 
               'i4840', 
               'i4955', 
               'i5422', 
               'i5850', 
               'i6188', 
               'i6595', 
               'i6635', 
               'i6653', 
               'i6989', 
               'i6991', 
               'i7021', 
               'i7579', 
               'i7599', 
               'i7722', 
               'i7819', 
               'i7963', 
               'i7964', 
               'i8033', 
               'i8056', 
               'i8058', 
               'i8060', 
               'i8062', 
               'i8064', 
               'i8067', 
               'i8068', 
               'i8071', 
               'i8208', 
               'i8218', 
               'i8284', 
               'i8287', 
               'i8410', 
               'i8418', 
               'i8524', 
               'i8526', 
               'i8527', 
               'i8531', 
               'i8534', 
               'i8535', 
               'i8536', 
               'i8537', 
               'i8540', 
               'i8541', 
               'i8542', 
               'i8544', 
               'i9030', 
               'i9195', 
               'i9442', 
               'i10123', 
               'i10228', 
               'i10278', 
               'i10380', 
               'i10381', 
               'i10676', 
               'i10739', 
               'i11517', 
               'i11891', 
               'i11893', 
               'i12239', 
               'i12344', 
               'i12553', 
               'i12941', 
               'i12963', 
               'i13000', 
               'i13214', 
               'i13296', 
               'i14223', 
               'i18163', 
               'i18165', 
               'i18177', 
               'i18181', 
               'i18187', 
               'i18193', 
               'i18195', 
               'i18198', 
               'i18208', 
               'i18209', 
               'i18210', 
               'i18248', 
               'i18250', 
               'i18264', 
               'i18302', 
               'i18332', 
               'i18333', 
               'i18350', 
               'i18352', 
               'i18477', 
               'i18578', 
               'i18581', 
               'i18582', 
               'i18694', 
               'i18757', 
               'i18760', 
               'i18761', 
               'i18762', 
               'i18763', 
               'i18764', 
               'i18812', 
               'i18873', 
               'i19191', 
               'i19229', 
               'i19623', 
               'i19691', 
               'i19749', 
               'i20135', 
               'i20838', 
               'i20839', 
               'i21309', 
               'i21658', 
               'i21674'
}
boost_dict = dict.fromkeys(boosters, "0.293")
#@staticmethod
def increment(valence, increment):
    """
    increment in the same direction as the valence
    """
    if valence == 0.0:
        return valence
    elif valence > 0:
        return valence + increment
    else: # valence < 0
        return valence - increment

#@staticmethod
def stretch(valence, increment):
    """
    stretch the valence
    """
    return valence * increment   

def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score

def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different

def _sift_sentiment_scores(sentiments):
    # want separate positive versus negative sentiment scores
    pos_sum = 0.0
    neg_sum = 0.0
    neu_count = 0
    for sentiment_score in sentiments:
        if sentiment_score > 0:
            pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
        elif sentiment_score < 0:
            neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
        else: #  sentiment_score == 0:
            neu_count += 1
    return pos_sum, neg_sum, neu_count

def score_valence(sentiments, punct_emph_amplifier):
    if sentiments:
        sum_s = float(sum(sentiments))
        # compute and add emphasis from punctuation in text
        sum_s = increment(sum_s, punct_emph_amplifier)

        compound = normalize(sum_s)
        # discriminate between positive, negative and neutral sentiment scores
        pos_sum, neg_sum, neu_count = _sift_sentiment_scores(sentiments)

        if pos_sum > math.fabs(neg_sum):
            pos_sum += punct_emph_amplifier
        elif pos_sum < math.fabs(neg_sum):
            neg_sum -= punct_emph_amplifier

        total = pos_sum + math.fabs(neg_sum) + neu_count
        pos = math.fabs(pos_sum / total)
        neg = math.fabs(neg_sum / total)
        neu = math.fabs(neu_count / total)

    else:
        compound = 0.0
        pos = 0.0
        neg = 0.0
        neu = 0.0

    sentiment_dict = \
        {"neg": round(neg, 3),
         "neu": round(neu, 3),
         "pos": round(pos, 3),
         "compound": round(compound, 4)}

    return sentiment_dict

    
def _amplify_ep(text):
    """
    check for added emphasis resulting from exclamation points (up to 4 of them)
    """
    ep_count = text.count("!")
    if ep_count > 4:
        ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
    ep_amplifier = ep_count * 0.292
    return ep_amplifier

def _amplify_qm(text):
    """
    check for added emphasis resulting from question marks (2 or 3+)
    """
    qm_count = text.count("?")
    qm_amplifier = 0
    if qm_count > 1:
        if qm_count <= 3:
            # (empirically derived mean sentiment intensity rating increase for
            # question marks)
            qm_amplifier = qm_count * 0.18
        else:
            qm_amplifier = 0.96
    return qm_amplifier

def punctuation_emphasis(text):
    # add emphasis from exclamation points and question marks
    ep_amplifier = _amplify_ep(text)
    qm_amplifier = _amplify_qm(text)
    punct_emph_amplifier = ep_amplifier + qm_amplifier
    return punct_emph_amplifier


class SentimentAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """
    def __init__(self, model="en_sense"):
        modpath = f"{__package__}.models.{model}"
        datapath = f"{__package__}.data"
        
        self.meta = self.read_meta(modpath, 'meta.yaml')

        ### Valence lexicons
        self.lexicon = dict()

        for lexfile in self.meta['lexicons']:
             self.lexicon.update(self.make_lex_dict(modpath, lexfile))

        print(f"loaded model {model}")
        ### 
 
    def read_meta(self, modpath, meta_file):
        """
        Read meta parameters for the model

        """
        with open_text(modpath, meta_file) as metafh:
            meta = yaml.safe_load(metafh)
        return meta
       

    def make_lex_dict(self, modpath, lexicon_file):
        """
        Convert lexicon file to a dictionary
        Expect a tab separated lexicon
        lemma	score	rest

        Allow comments with hashes
        """
        lex_dict = {}
        fh = open_text(modpath, lexicon_file)
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    
    def lexical_valence(self, w, p, l, t):
        """
        find the lexical valence 
        apply any morphological changes
        """
        if t in self.lexicon:
            valence = self.lexicon[t]
            
        if valence:
            if p == 'JJR':  # comparative
                valence = increment(valence, COMP_INCR)
            elif p == 'JJS':  # superlative
                valence = increment(valence, SUPR_INCR)
 
        return valence
                
    
    def sentiment_valence(self, i, senses, is_cap_diff):
        valence = 0.0
        (w, p, l, t) = senses[i]
        ### get the base valence, with morphological changes
        if t in self.lexicon:
            valence = self.lexical_valence(w, p, l, t)

        ### CAPITALIZATION
        if valence and is_cap_diff and \
           w.isupper() and not l.isupper():
            valence = increment(valence, C_INCR)
            
        return valence

    
    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """   
        senses = disambiguate(text, en, morphy)
        print(senses)
        is_cap_diff = allcap_differential([w for (w, p, l, t) in senses])
        ### pad with beginners?

        sentiments = list()

        for i, (w, p, l, t)  in enumerate(senses):          #position, word, pos, lemma, i-tag
            local = self.sentiment_valence(i, senses, is_cap_diff) 
            if i > 1 and (senses[i-1] in boost_dict):         #add -B_INCR to boosters itself
                local = increment(local, B_INCR)
            if i > 2 and (senses[i-2] in boost_dict): 
                local = increment(local, B_INCR*0.95) 
            if i > 3 and (senses[i-3] in boost_dict):
                local = increment(local, B_INCR*0.9) 
            if i > 1 and (senses[i-1] in neg_dict):
                local = stretch(local,N_SCALAR)
            if i > 2 and (senses[i-2] in neg_dict):
                local = stretch(local,N_SCALAR)
            if i > 3 and (senses[i-3] in neg_dict):
                local = stretch(local,N_SCALAR)
            sentiments.append(local)

        punct_score = punctuation_emphasis(text)

        valence_dict = score_valence(sentiments, punct_score)
        print(sentiments)

        return valence_dict
                

    
# if __name__ == '__main__':
#     sentences = ["VADER is smart, handsome, and funny.",
#                  "We have some problems."]
#     analyzer = SentimentAnalyzer()
