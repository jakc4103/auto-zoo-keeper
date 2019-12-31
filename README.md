# auto-zoo-keeper

## Disclaimer
This repo is for educational and research purpose.  
If you like the game, go download and enjoy it without this module!  
I am not responsible for any consequences of using this mod.  
Try at your own risk.


## What is this repo about
This repo aims to solve the popular game **Zoo Keeper Battle** ([google_play](https://play.google.com/store/apps/details?id=jp.kiteretsu.zookeeperbattle.google&hl=zh_TW), [app_store](https://apps.apple.com/tw/app/zookeeper-battle/id548270497)) using pure **computer vision** techniques.  
Ths goal is to automatically solve puzzles without human intervention.

![](image/demo_crop.gif?raw=true)

## The thoughts
At first, I was thinking to get the optimal icon-moving decisions at each round of battle. However, I found that these kinds of game is actually **NP-Hard** (in short: it takes time and luck to get optimal dicisions); thus I decided to use brute force solutions instead.  
  
  To use brute force solutions, the program must perceive the icons on the board fast and produce decisions then move icons with **low latency**. There are 4 main parts in the program:  
  `perceive -> info-transform -> decision-making -> do-action`  

    
See more about [Why this game is NP-Hard](https://www.isnphard.com/g/bejeweled/), [NP-Hard](https://en.wikipedia.org/wiki/NP-hardness)

## How this program works?
There are 2 versions: single process and multi-process.  
For single process version, the **latency** is much higher than multi-process version  
![](image/single.png?raw=true)    

I use Opencv to recognize animal icons on the screen. To be specific, I use [matchTemplate](https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html) to detect all icons.  
However, the **scale** of image is an issue since matchTemplate cannot handle images of different scale. Thus, I detect scale in the main login images.  
After that, we dont need to worry about scale later while using matchTemplate.  

With the images been converted to 8x8 numpy array, I use [cross-correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html) to find all possible moves on the board.  
This is convenient since I only need to design filters, instead of find 3-icons-in-the-row by hand-crafted rules.

While some possible moves extracted from numpy array, I use [pyautogui](https://pypi.org/project/PyAutoGUI/) to control mouse device and ignore some moves if there are too many moves in current round (in order to lower latency).  

Although I apply some early termination rules to single process version, the latency is still too high to get good scores.  
So, I develop multi-process version. I use [shared-memory](https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Array) to speed up the process since using [Queue](https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Queue) or [Manager](https://docs.python.org/2/library/multiprocessing.html#multiprocessing-managers) to communucate between process is kind of slow.  
Here is the flow of multi-process version  
![](image/mp.png?raw=true)

## Usage
You need to project your phone screen to computer screen. Personally I recommand [scrcpy](https://github.com/Genymobile/scrcpy).  
You can use android simulators, too.  
Start the program with the game in login screen (to locate game board and image scale), then enjoy it.
See [Demo](https://www.youtube.com/watch?v=zXbJ2C2au1c&feature=youtu.be) for step-by-step usage.

To run the program, run  
  `python main.py`    
or multiprocessing version  
  `python main.py --mp`

For iOS, I did not found any solutions to control the phone over projected screen and mouse; I'm afraid it does not support iOS phones.

## Dependencies
    python 3.6 (all python version should work)
    numpy  
    opencv  
    python-mss
    pyautogui  
    scipy  

## Acknowledgements
https://play.google.com/store/apps/details?id=jp.kiteretsu.zookeeperbattle.google&hl=zh_TW
