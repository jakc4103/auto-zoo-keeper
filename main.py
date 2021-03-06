import cv2
import mss
import numpy as np
from scipy import signal
import pyautogui
import time
import argparse
from multiprocessing import Queue, Process, Array
from queue import Empty
import ctypes

top_left_icon = None
top_left_icon_small = None
scale = None

animals = {
    2: 'elephant',
    3: 'frog',
    5: 'giraffe',
    9: 'hippo',
    12: 'lion',
    17: 'monkey',
    23: 'panda',
    30: 'boss'
}

filters = {
    0: np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
    1: np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0.5], [0, 0, 0.5]]),
    2: np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0.5]]),
    3: np.array([0, 0, 0, 0, 0, 0.5, 0.5]).reshape((1, -1))
}

dirs = {
    0: np.array([0, 1]),
    1: np.array([-1, 0]),
    2: np.array([0, -1]),
    3: np.array([1, 0])
}

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp", action="store_true", help="Whether to use multiprocessing for better icon moving")
    return parser.parse_args()

def locate_home_screen(img, screen, thres=0.8):
    min_h = min_w = 300000
    max_h = max_w = 0
    _, w, h = screen.shape[::-1]
    loc = []

    img_tmp = img
    scale = 1.0
    while len(loc) == 0 and np.sum(np.array(img_tmp.shape) > np.array(screen.shape) + 20) == 2:
        scale -= 0.01
        img_tmp = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        res = cv2.matchTemplate(img_tmp, screen, cv2.TM_CCOEFF_NORMED)

        loc = np.where( res >= thres)
        loc = list(zip(*loc[::-1]))
    
    for pt in loc:
        min_h = min(min_h, pt[1])
        min_w = min(min_w, pt[0])
        max_h = max(max_h, pt[1]+h)
        max_w = max(max_w, pt[0]+w)
    
    return min_h, max_h, min_w, max_w, img_tmp.shape


def is_pattern_found(img, pattern, thres=0.8):
    res = cv2.matchTemplate(img, pattern, cv2.TM_CCOEFF_NORMED)

    loc = np.where( res >= thres)
    loc = list(zip(*loc[::-1]))

    if len(loc) > 0:
        return True

    return False

def locate_animals(img, target_dict, thres=0.8, scale=1):
    """
    locate the position of board at first time
    """
    coord_dict = {}
    coord_list = []
    for idx in animals:
        target = target_dict[idx]
        w, h = target.shape[::-1]

        res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thres)
        loc = list(zip(*loc))
        boxes = [[int(ll[1]), int(ll[0]), w, h] for ll in loc]

        indices = cv2.dnn.NMSBoxes(boxes, [.8] * len(boxes), 0.5, 0.5)
        
        loc = [loc[i[0]] for i in indices]
        coord_dict[idx] = np.array(loc)
        if len(loc) != 0:
            coord_list.append(coord_dict[idx])

    return coord_dict, coord_list


def get_arr(img, target_dict):
    global top_left_icon, top_left_icon_small, scale
    arr = np.zeros((8, 8))
    coord_dict, coord_list = locate_animals(img, target_dict)

    if len(coord_list) != 0:
        if len(coord_list) > 1:
            res = np.concatenate(coord_list, 0)
        else:
            res = np.array(coord_list[0])
        
        if res.shape[0] > 50: # 50 animals detected
            if top_left_icon is None:
                indice = np.argmin(np.sum(res, 1))
                top_left_icon_small = res[indice]
                top_left_icon = np.round(res[indice] * scale).astype(np.int64)

            min_h, min_w = top_left_icon_small

            for key in coord_dict:
                if coord_dict[key].shape[0] == 0:
                    continue
                coord_dict[key][:, 0] -= min_h
                coord_dict[key][:, 1] -= min_w
                
                coord = (coord_dict[key] / target_dict[2].shape[0]).astype(np.int32)
                for co in range(coord.shape[0]):
                    arr[tuple(coord[co])] = key
        
    return arr


def get_move(arr=None):
    """
    use correlate(convolve without reverse kernel order) to get all possible moves
    """
    if arr is None or np.sum(arr!=0) < 55:
        return []
        
    moves = [] # (coord, dir) ex ((3, 4), 0) means move (3, 4) to right, 0 right, 1 up, 2 left, 3 down
    mask_moved = np.ones_like(arr)
    replace_value = 0
    # detect 2 consecutive
    for key in filters:
        for rot in [1, 3, 0, 2]:
            early_break = False
            out = signal.correlate2d(arr, np.rot90(filters[key], rot), mode='same', fillvalue=100)
            
            mask = (out==arr).astype(np.float)
            tmp = np.stack(np.where(mask), -1)
            # print(tmp)
            for idx in range(tmp.shape[0]):
                # if mask_moved[tuple(tmp[idx])] == 1:
                if mask_moved[tuple(tmp[idx])] == 1 and mask_moved[tuple(tmp[idx]+dirs[rot])] == 1:
                # if mask_moved[tuple(tmp[idx])] == 1 and mask_moved[tuple(tmp[idx]+dirs[rot])] == 1 and arr[tuple(tmp[idx]+dirs[rot])] != replace_value:
                    moves.append((tmp[idx], rot))
                    # mask_moved[tuple(tmp[idx])] = 0
                    mask_moved[tuple(tmp[idx]+dirs[rot])] = 0
                    arr[tuple(tmp[idx])], arr[tuple(tmp[idx]+dirs[rot])] = arr[tuple(tmp[idx]+dirs[rot])], arr[tuple(tmp[idx])]
                    arr[tuple(tmp[idx]+dirs[rot])] = replace_value
                    if key == 3:
                        mask_moved[tuple(tmp[idx]+dirs[rot]*2)] = 0
                        mask_moved[tuple(tmp[idx]+dirs[rot]*3)] = 0
                        arr[tuple(tmp[idx]+dirs[rot]*2)] = replace_value
                        arr[tuple(tmp[idx]+dirs[rot]*3)] = replace_value
                    elif key == 2:
                        mask_moved[tuple(tmp[idx]+dirs[rot]+dirs[(rot+1)%4])] = 0
                        mask_moved[tuple(tmp[idx]+dirs[rot]+dirs[(rot+3)%4])] = 0
                        arr[tuple(tmp[idx]+dirs[rot]+dirs[(rot+1)%4])] = replace_value
                        arr[tuple(tmp[idx]+dirs[rot]+dirs[(rot+3)%4])] = replace_value
                    elif key == 0:
                        mask_moved[tuple(tmp[idx]+dirs[rot]+dirs[(rot+1)%4])] = 0
                        mask_moved[tuple(tmp[idx]+dirs[rot]+2*dirs[(rot+1)%4])] = 0
                        arr[tuple(tmp[idx]+dirs[rot]+dirs[(rot+1)%4])] = replace_value
                        arr[tuple(tmp[idx]+dirs[rot]+2*dirs[(rot+1)%4])] = replace_value
                    else:
                        mask_moved[tuple(tmp[idx]+dirs[rot]+dirs[(rot+3)%4])] = 0
                        mask_moved[tuple(tmp[idx]+dirs[rot]+2*dirs[(rot+3)%4])] = 0
                        arr[tuple(tmp[idx]+dirs[rot]+dirs[(rot+3)%4])] = replace_value
                        arr[tuple(tmp[idx]+dirs[rot]+2*dirs[(rot+3)%4])] = replace_value
                    early_break = True
                    break
            if early_break:
                break
                    
        if len(moves) > 5: # early break to save computing resources
            break

    if len(moves) == 0:
        icon_other = np.stack(np.where(arr==0), -1)
        for idx in range(icon_other.shape[0]):
            moves.append((icon_other[idx], np.random.randint(0, 4)))

    return moves

def move_icon(share_arr, q, icon_shape, coord_base):
    battle_mode = False
    top_left_icon = None
    time_margin = 0.

    with share_arr.get_lock(): # synchronize access
        arr = np.frombuffer(share_arr.get_obj()).reshape(8, 8) # update arr

    try:
        while True:
            try:
                out = q.get(block=False)
                if type(out) == str:
                    if out == 'battle':
                        battle_mode = True
                    elif out == 'pause':
                        battle_mode = False

                elif type(out) == np.ndarray:
                    top_left_icon = out

                elif out is None:
                    break

            except Empty:
                pass

            if not battle_mode or top_left_icon is None:
                continue

            moved = False
            # detect 2 consecutive
            for key in filters:
                for rot in [1, 3, 0, 2]:
                    
                    moves = []
                    # early_break = False
                    with share_arr.get_lock():
                        arr_copy = arr.copy()
                    out = signal.correlate2d(arr_copy, np.rot90(filters[key], rot), mode='same', fillvalue=100)
                    
                    mask = (out==arr_copy).astype(np.float) * (out!=0).astype(np.float)
                    tmp = np.stack(np.where(mask), -1)
                    # print(tmp)
                    for idx in range(tmp.shape[0]):
                        moves.append((tmp[idx], rot))

                        break
                    
                    # move icons here
                    if len(moves) > 0:
                        prev_move = None
                        moved = True
                        counter = 0
                        start_time = time.time()
                        for mm in moves:
                            if counter > 3:
                                break
                            if prev_move is not None and np.sum(np.abs(mm[0] - prev_move[0])) < 3:
                                prev_move = mm
                                continue 
                            start, dd = mm #moves[-1]
                            
                            coord_start = (start * icon_shape + (icon_shape*2/3)  + top_left_icon + coord_base).astype(np.int32)
                            coord_dest = (coord_start + dirs[dd] * icon_shape).astype(np.int32)

                            pyautogui.moveTo(coord_start[1], coord_start[0])
                            pyautogui.dragTo(coord_dest[1], coord_dest[0], button='left')

                            prev_move = mm
                            counter += 1
                            
                            cur_time = time.time()
                            diff_time = cur_time - start_time
                            # print("diff_time: {}".format(diff_time))
                            if diff_time < time_margin:
                                time.sleep(time_margin - diff_time)

                            start_time = time.time()

            if not moved:
                icon_other = np.stack(np.where(arr==0), -1)
                moves = []
                if icon_other.shape[0] < 6:
                    for idx in range(icon_other.shape[0]):
                        moves.append((icon_other[idx], np.random.randint(0, 4)))

                    prev_move = None
                    moved = True
                    counter = 0
                    for mm in moves:
                        if counter > 3:
                            break
                        if prev_move is not None and np.sum(np.abs(mm[0] - prev_move[0])) < 3:
                            prev_move = mm
                            continue 
                        start, dd = mm #moves[-1]
                        
                        coord_start = (start * icon_shape + (icon_shape*2/3)  + top_left_icon + coord_base).astype(np.int32)
                        coord_dest = (coord_start + dirs[dd] * icon_shape).astype(np.int32)

                        pyautogui.moveTo(coord_start[1], coord_start[0])
                        pyautogui.dragTo(coord_dest[1], coord_dest[0], button='left')

                        prev_move = mm
                        counter += 1
    except KeyboardInterrupt:
        pass
                    
def main():
    global top_left_icon, scale
    args = get_argument()
    # read all pattern images
    target_dict = {}
    for i in animals:
        target_dict[i] = cv2.cvtColor(cv2.imread("icons/animals/{}.png".format(animals[i])), cv2.COLOR_BGR2GRAY)
    home_screen = cv2.imread("icons/home.png")
    round_start = cv2.cvtColor(cv2.imread("icons/round.png"), cv2.COLOR_BGR2GRAY)
    win = cv2.cvtColor(cv2.imread("icons/win.png"), cv2.COLOR_BGR2GRAY)
    battle = cv2.cvtColor(cv2.imread("icons/battle.png"), cv2.COLOR_BGR2GRAY)
    home_icons = cv2.cvtColor(cv2.imread("icons/home_icons.png"), cv2.COLOR_BGR2GRAY)

    game_mode = 0
    battle_start_time = None

    if args.mp:
        top_left_flag = True

    with mss.mss() as sct:
        min_h = min_w = 300000
        max_h = max_w = 0
        scale = 0
        print("Scan for home screen")
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(sct.monitors[1]))
            min_h, max_h, min_w, max_w, shape = locate_home_screen(img[:, :, :3], home_screen)
            if max_h != 0:
                scale = img.shape[0] / shape[0]
                break
        try:
            if scale != 0:
                print("Screen detected, standing by")
                game_mode = 1
                min_h = int(round(min_h * scale)) - 5
                max_h = int(round(max_h * scale)) + 5
                min_w = int(round(min_w * scale)) - 5
                max_w = int(round(max_w * scale)) + 5
                coord_base = np.array([min_h, min_w])

                monitor = {"top": min_h, "left": min_w, "width": max_w-min_w, "height": max_h-min_h}

                # get the correct size patterns
                # round_start = cv2.resize(round_start, (0, 0), fx=scale, fy=scale)
                # win = cv2.resize(win, (0, 0), fx=scale, fy=scale)
                # battle = cv2.resize(battle, (0, 0), fx=scale, fy=scale)
                # for i in target_dict:
                    # target_dict[i] = cv2.resize(target_dict[i], (0, 0), fx=scale, fy=scale)
                
                icon_shape = np.array(cv2.resize(target_dict[2], (0, 0), fx=scale, fy=scale).shape)

                # start second process if specified
                if args.mp:
                    print("Start moving process...")
                    q = Queue(5) 
                    shared_arr = Array(ctypes.c_double, 64)
                    np_arr = np.frombuffer(shared_arr.get_obj())
                    process_move = Process(target=move_icon, args=(shared_arr, q, icon_shape, coord_base, ))
                    process_move.start()

                while True:
                    img = np.array(sct.grab(monitor))[:, :, :3]
                    img = cv2.resize(img, (0, 0), fx=1/scale, fy=1/scale)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if (game_mode == 1 or game_mode == 3) and is_pattern_found(img_gray, round_start):
                        game_mode = 2
                        battle_start_time = time.time()
                        print("Start battle!!!")
                        if args.mp:
                            q.put("battle")

                    elif game_mode == 3 and is_pattern_found(img_gray, win):
                        game_mode = 1
                        print("End battle, standing by")
                        if args.mp:
                            q.put("pause")

                    elif game_mode == 2:
                        # detect, parse, and command board
                        arr = get_arr(img_gray, target_dict)
                        if args.mp:
                            if top_left_icon is not None and top_left_flag:
                                top_left_flag = False
                                q.put(top_left_icon)
                            with shared_arr.get_lock():
                                np_arr[:] = arr.reshape(-1)

                        else:
                            moves = get_move(arr)
                            if len(moves) != 0:
                                for idx in range(len(moves)):
                                    start, dd = moves[idx]
                                    
                                    coord_start = (start * icon_shape + (icon_shape*2/3)  + top_left_icon + coord_base).astype(np.int32)
                                    coord_dest = (coord_start + dirs[dd] * icon_shape).astype(np.int32)

                                    pyautogui.moveTo(coord_start[1], coord_start[0])
                                    pyautogui.dragTo(coord_dest[1], coord_dest[0], button='left')

                        # detect if cur round is over
                        if time.time()-battle_start_time > 29:
                            if is_pattern_found(img_gray, battle):
                                game_mode = 3
                                if args.mp:
                                    q.put('pause')
                                print("Pause battle")
                            elif is_pattern_found(img_gray, home_icons):
                                game_mode = 1
                                print("Home icons detected, standing by")

                    elif game_mode != 1 and is_pattern_found(img_gray, home_icons):
                        game_mode = 1

                    time.sleep(0.005)

        except KeyboardInterrupt:
            if args.mp:
                q.put(None)
                process_move.join()
                    

if __name__ == '__main__':
    main()