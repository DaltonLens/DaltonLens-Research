import numpy as np

debug = False

class TermColors:
    END      = '\33[0m'
    BOLD     = '\33[1m'
    ITALIC   = '\33[3m'
    URL      = '\33[4m'
    BLINK    = '\33[5m'
    BLINK2   = '\33[6m'
    SELECTED = '\33[7m'

    BLACK  = '\33[30m'
    RED    = '\33[31m'
    GREEN  = '\33[32m'
    YELLOW = '\33[33m'
    BLUE   = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE  = '\33[36m'
    WHITE  = '\33[37m'

    BLACKBG  = '\33[40m'
    REDBG    = '\33[41m'
    GREENBG  = '\33[42m'
    YELLOWBG = '\33[43m'
    BLUEBG   = '\33[44m'
    VIOLETBG = '\33[45m'
    BEIGEBG  = '\33[46m'
    WHITEBG  = '\33[47m'

    GREY    = '\33[90m'
    RED2    = '\33[91m'
    GREEN2  = '\33[92m'
    YELLOW2 = '\33[93m'
    BLUE2   = '\33[94m'
    VIOLET2 = '\33[95m'
    BEIGE2  = '\33[96m'
    WHITE2  = '\33[97m'

    GREYBG    = '\33[100m'
    REDBG2    = '\33[101m'
    GREENBG2  = '\33[102m'
    YELLOWBG2 = '\33[103m'
    BLUEBG2   = '\33[104m'
    VIOLETBG2 = '\33[105m'
    BEIGEBG2  = '\33[106m'
    WHITEBG2  = '\33[107m'

def in_range(im: np.ndarray, r: int, c: int):
    return r >= 0 and r < im.shape[0] and c >= 0 and c < im.shape[1]

def swap_rb(im):
    return im[:,:,[2,1,0]]

def bool_image_to_uint8(im):
    return im.astype(np.uint8)*255

def saturation(rgb):
    V = np.max(rgb)
    C = V - np.min(rgb)
    return C / V if V > 1e-8 else 0.0
    

class InfiniteIterator:
    def __init__(self, sequence):
        self.sequence = sequence
        self.it = iter(self.sequence)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            e = next(self.it)
        except StopIteration:
            self.it = iter(self.sequence)
            e = next(self.it)
        return e
