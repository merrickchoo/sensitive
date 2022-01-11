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

def amplify_int():

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