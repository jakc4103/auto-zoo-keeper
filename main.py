import cv2
import mss
import numpy as np
from scipy import signal

"""
0 elephant
1 frog
2 giraffe
3 hippo
4 lion
5 monkey
6 panda
7 rabbit
"""
animals = {
    2: 'elephant',
    3: 'frog',
    5: 'giraffe',
    8: 'hippo',
    12: 'lion',
    17: 'monkey',
    23: 'panda'
}

def parse_screen(img, targ, thres=0.8, scale=1):
    """
    parse all animals on board
    """
    img_tmp = cv2.resize(img, (0, 0), fx=scale, fy=scale)#, interpolation=cv2.INTER_NEAREST)
 
    res = cv2.matchTemplate(img_tmp, targ, cv2.TM_CCOEFF_NORMED)
    
    loc = np.where( res >= thres)
    loc = list(zip(*loc[::-1]))
    
    while len(loc) == 0 and np.sum(np.array(img_tmp.shape) > np.array(targ.shape) + 20) == 2:
        scale *= 0.99
        img_tmp = cv2.resize(img_tmp, (0, 0), fx=scale, fy=scale)#, interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(img_tmp, targ, cv2.TM_CCOEFF_NORMED)

        loc = np.where( res >= thres)
        loc = list(zip(*loc[::-1]))

    return loc, scale, img_tmp

def locate_board_animals(img, target_dict, thres=0.8, scale=1):
    """
    locate the position of board at first time
    """
    min_h = min_w = 300000
    max_h = max_w = 0
    for idx in range(1, 8):
        target = target_dict[idx]
        _, w, h = target.shape[::-1]

        img_tmp = cv2.resize(img, (0, 0), fx=scale, fy=scale)#, interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(img_tmp, target, cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= thres)
        loc = list(zip(*loc[::-1]))
        scale = 0.95
        while len(loc) == 0 and np.sum(np.array(img_tmp.shape) > np.array(target.shape) + 20) == 2:
            # scale *= 0.99
            img_tmp = cv2.resize(img_tmp, (0, 0), fx=scale, fy=scale)#, interpolation=cv2.INTER_NEAREST)
            res = cv2.matchTemplate(img_tmp, target, cv2.TM_CCOEFF_NORMED)

            loc = np.where( res >= thres)
            loc = list(zip(*loc[::-1]))
        
        for pt in loc:
            min_h = min(min_h, pt[1])
            min_w = min(min_w, pt[0])
            max_h = max(max_h, pt[1]+h)
            max_w = max(max_w, pt[0]+w)

        if ((max_h - min_h) // 28) >= 8 and ((max_w - min_w) // 28) >= 8:
            break
    
    return min_h, max_h, min_w, max_w, img_tmp.shape[:2][::-1]

def locate_board(img, board, thres=0.8):
    """
    locate the position of board at second time
    """
    min_h = min_w = 300000
    max_h = max_w = 0
    scale = 1.05
    res = cv2.matchTemplate(img, board, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= thres)
    loc = list(zip(*loc[::-1]))
    _, w, h = board.shape[::-1]
    while len(loc) == 0 and np.sum(np.array(img.shape) > np.array(board.shape) + 20) == 2:
        # scale *= 1.005
        board = cv2.resize(board, (0, 0), fx=scale, fy=scale)#, interpolation=cv2.INTER_NEAREST)
        _, w, h = board.shape[::-1]
        res = cv2.matchTemplate(img, board, cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= thres)
        loc = list(zip(*loc[::-1]))

    for pt in loc:
        min_h = min(min_h, pt[1])
        min_w = min(min_w, pt[0])
        max_h = max(max_h, pt[1]+h)
        max_w = max(max_w, pt[0]+w)
    
    return min_h, max_h, min_w, max_w

transformer = None

class Transformer:
    def __init__(self, shape, shape2, min_h, max_h, min_w, max_w):
        self.shape = shape
        self.shape2 = shape2
        self.min_h = min_h
        self.max_h = max_h
        self.min_w = min_w
        self.max_w = max_w


    def __call__(self, x):
        # return cv2.resize(cv2.resize(x, self.shape), self.shape2)[self.min_h:self.max_h, self.min_w:self.max_w, :]
        return x[self.min_h:self.max_h, self.min_w:self.max_w, :]


def get_arr(img, target_dict):
    global transformer
    arr = np.zeros((8, 8))

    if transformer is None:
        # print("no transformer")
        min_h, max_h, min_w, max_w, shape = locate_board_animals(img, target_dict)
        
        if max_h - min_h > 200 and max_w - min_w > 200:
            board = cv2.resize(img, shape)
            min_h, max_h, min_w, max_w, shape2 = locate_board_animals(board, target_dict)
            board = cv2.resize(board, shape2)
            board = board[min_h:max_h, min_w:max_w, :]
            min_h, max_h, min_w, max_w = locate_board(img, board)
            
            transformer = Transformer(shape, shape2, min_h, max_h, min_w, max_w)

    else:
        # print("transformer")
        img_show = transformer(img)
        cv2.imshow("transformer", img_show)
        cv2.waitKey(1)

    return arr

filters = {
    0: np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    1: np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0.5]]),
    2: np.array([0, 0, 0, 0, 0, 0.5, 0.5]).reshape((1, -1))
}


def get_move(arr=None):
    """
    0 for right, 1 for up, 2 for left, 3 for down
    """
    if arr is None:
        arr = np.array([[ 3.,  8., 12., 12.,  5., 17.,  8., 17.],
                        [ 8.,  5., 23., 23.,  5.,  5.,  2.,  3.],
                        [ 5., 12., 23.,  3., 12., 23., 17., 23.],
                        [ 8.,  0., 12.,  2., 23.,  3., 12.,  2.],
                        [ 5., 17., 23.,  5.,  8.,  3.,  5.,  8.],
                        [12.,  2.,  2., 17.,  2.,  5.,  2.,  2.],
                        [17., 17.,  5.,  3., 12.,  8., 17.,  3.],
                        [ 8., 12.,  5., 17., 12.,  2.,  5.,  3.]])

    moves = [] # (coord, dir) ex ((3, 4), 1) means move (3, 4) to right
    # detect 2 consecutive
    for key in filters:
        for rot in range(4):
            out = signal.correlate2d(arr, np.rot90(filters[key], rot), mode='same', fillvalue=100)
            # print(arr)
            # print(np.rot90(filters[key], rot))
            # print(out==arr)
            
            mask = (out==arr).astype(np.float)
            
            # from 2 consecutive, detect 3 consecutive

def main():
    scale = 1
    arr = np.zeros((8, 8))
    coord = {}
    min_h = min_w = 300000
    max_h = max_w = 0
    for idx in animals:
        # print("{}".format(animals[idx]))
        img = cv2.imread("ttt.png")
        # img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

        target = cv2.imread("icons/animals/{}.png".format(animals[idx]))
        # target = cv2.imread("crop.png")
        # target = cv2.resize(target, (0, 0), fx=0.5, fy=0.5)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        c, w, h = target.shape[::-1]

        loc, scale, img = parse_screen(img, target, 0.75, 1)
        img_show = img
        coord[idx] = loc
        
        for pt in loc:
            min_h = min(min_h, pt[1])
            min_w = min(min_w, pt[0])
            max_h = max(max_h, pt[1]+h)
            max_w = max(max_w, pt[0]+w)
            # cv2.rectangle(img_rgb, (int(round(pt[0]/scale)), int(round(pt[1]/scale))), (int(round((pt[0]+w) / scale)), int(round((pt[1]+h) / scale))), (0, 0, 255), 2)
            cv2.rectangle(img_show, pt, (int(pt[0]) + w, int(pt[1]) + h), (0, 0, 255), 2)

        cv2.imshow("res", img_show)
        cv2.waitKey(0)

    for key in coord:
        for pt in coord[key]:
            x = int(round((pt[0] - min_w)/28))
            y = int(round((pt[1] - min_h)/28))
            arr[y, x] = key
    print(animals)
    print(arr)

def capture():
    import time
    target_dict = {}
    for i in animals:
        target_dict[i] = cv2.imread("icons/animals/{}.png".format(animals[i]))
        
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 640, "mon": 0}
        # monitor = {"mon": 1}

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(sct.monitors[1]))

            arr = get_arr(img[:, :, :3], target_dict)
            print(arr)
            print("fps: {}".format(1 / (time.time() - last_time)))


if __name__ == '__main__':
    # capture()
    get_move()
    # main()