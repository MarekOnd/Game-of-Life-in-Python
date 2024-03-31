import numpy as np
from numba import jit, cuda
import time
import cv2 as cv
from gooey import Gooey
import argparse

SIZEX = 1200
SIZEY = 1920
COUNT_TO_LIVE = 2 # 2
COUNT_TO_REVIVE = 3 # 3
COUNT_TO_OVERCROWD = 4 # 4

board:np.ndarray = np.zeros((SIZEX,SIZEY),dtype=bool)

VECTORS = np.array([[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1],[-1,1],[1,-1]])

@Gooey(
    optional_cols=3,
    program_name='Game of Life in Python',
    menu=[{
        'name': 'File',
        'items': [{
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'Game of Life in Python',
                'description': 'Implementation of Game of Life in python using various approaches. Uses integration of Python with CUDA.',
                'version': '1.',
                'website': 'https://github.com/MarekOnd/Game-of-Life-in-Python',
                'developer': 'https://marekond.github.io/',
                'license': 'MIT'
            }, {
                'type': 'MessageDialog',
                'menuTitle': 'Information',
                'caption': 'Test Message',
                'message': 'I am trying Gooey'
            }]
        }
    ]
)
def main():
    """
    Arguments parser setup, initialization, main loop and termination of the program.
    """
    # INITIAL ARGUMENT PARSING USING GUI DECORATOR
    parser = argparse.ArgumentParser(prog="Game of life using GPU",
                                     description="Several methods of simulating Game of life on big arrays using NumPy, numba cuda and jit\nfor loop optimization",
                                     epilog="hope this works")

    parser.add_argument('method',type=str,help='Which method for updating should the program use gpu, jit, for, numpy',default='gpu')
    general_group = parser.add_argument_group(
        "Visual output options", 
        "Set "
    )
    general_group.add_argument('-frames', type=int,default=100)
    # STARTING LAYOUT
    general_group.add_argument('--random','-r',action='store_false',default=True)
    general_group.add_argument('-construct','-const',nargs='*',type=str,default=[])
    # SHOW
    show_group = parser.add_argument_group(
        "Visual output options", 
        "Set whether to show images in window, fullscreen and the max frames per second."
    )
    show_group.add_argument('--show',action='store_true',default=True)
    show_group.add_argument('--full', '--f',action='store_true',default=False)
    # EXPORT
    export_group = parser.add_argument_group(
        "Export options", 
        "Set whether the images should be saved into a video."
    )
    export_group.add_argument('--export',action='store_true',default=False)
    export_group.add_argument('-fpsexp',action='store',type=int,default=25)
    export_group.add_argument('-o',action='store',type=str,default='./output.avi')
    


    args = parser.parse_args()

    global board
    # SHOW
    
    if args.show:
        if args.full:
            cv.namedWindow('img',cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty("img",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow('img',cv.WINDOW_NORMAL)

    # EXPORT
    out:cv.VideoWriter # predeclares video output
    if args.export:
        out = cv.VideoWriter(args.o, cv.VideoWriter_fourcc(*'mp4v'), float(args.fpsexp), (SIZEX, SIZEY), True)

    # METHOD
    updatingFunction = set_updating_function(args.method)

    # STARTING LAYOUT
    if not args.random:
        random_layout()
    for structure in args.construct:
        create_structure(structure)

    i:int = 0 # iterating value
    averageFps:float = 0
    
    while i < args.frames:
        # FPS
        start = time.time()
        # SHOW
        if args.show:
            cv.imshow("img",board.astype(float))
            #cv.imshow("img",board.astype(float)/2) # does not damage display
            
        # EXPORT
        if args.export:
            frame = cv.cvtColor((np.floor(board.astype(float)*255).astype(dtype='uint8')), cv.COLOR_BGRA2BGR)
            out.write(frame)
            
            
        # UPDATE
        board = updatingFunction()
        # FPS
        averageFps = (averageFps*20 + 1/(time.time()-start))/21
        print('frame: ' + str(i) + '    '  + str(averageFps),end=' fps                    \r')
        # LOOP
        cv.waitKey(1)
        i+=1
    frame = cv.cvtColor((np.floor(board.astype(float)*255).astype(dtype='uint8')), cv.COLOR_BGRA2BGR)
    cv.imwrite('last_image.jpg', frame)
    if args.export:
        out.release()
    if args.show:
        cv.destroyAllWindows()

def set_updating_function(name:str):
    match name:
        case 'gpu' | 'cudajit' | 'cuda':
            return get_which_survive_gpu
        case 'jit':
            return get_which_survive_jit
        case 'numpy':
            return get_which_survive_ndarrays
        case 'for':
            return get_which_survive_for
        case _:
            ValueError('Option "' + name + '" does not exist')

# Starting layout of board
def random_layout():
    global board
    board = np.random.randint(0,2,size=(SIZEX,SIZEY),dtype=bool)

def create_structure(name):
    match name:
        case 'glider' | 'g': # GLIDER
            create_in_board(".O...OOOO",3)
        case 'infinite' | 'inf': # INFINITELY GROWING PATTERN
            create_in_board("......O.....O.OO....O.O.....O.....O.....O.O.....", 6)

def create_in_board(structure,lines):
    global board
    flattened = board[0:(0+lines),0:(0+len(structure)/lines)].flat
    for i in np.arange(0,len(structure)):
        if structure[i] == "O":
            flattened[i] = True

def get_next_gen(neighbourSumBoard:np.ndarray):
    """
    Returns bool array of living cells from the counted neighbouring living cells array.
    """
    enoughNeighboursBoard = neighbourSumBoard>=COUNT_TO_LIVE
    notTooManyNeigbours = neighbourSumBoard<COUNT_TO_OVERCROWD
    canBeRevived = neighbourSumBoard==COUNT_TO_REVIVE
    # some smort functions to declare what happens with only array counting
    willSurvive = (enoughNeighboursBoard.astype(int) + notTooManyNeigbours.astype(int) + board.astype(int)) == 3
    willBeRevived = canBeRevived.astype(int)*2 + board.astype(int) == 2
    willBeAlive = willSurvive.astype(int) + willBeRevived.astype(int) >= 1
    return willBeAlive

# 1) NUMPY NDARRAYS
def get_which_survive_ndarrays():
    """
    Creates a bigger array and add shiftes arrays by given values into it.
    """
    global board
    eB = np.zeros((SIZEX+2,SIZEY+2),dtype=bool) # EXTENDED BOARD
    eB[1:SIZEX+1,1:SIZEY+1] = board
    neighbourSumBoard = np.zeros((SIZEX,SIZEY),dtype=int)
    for i in np.arange(len(VECTORS)):
        neighbourSumBoard += np.array(eB[1+VECTORS[i,0]:SIZEX+1+VECTORS[i,0],1+VECTORS[i,1]:SIZEY+1+VECTORS[i,1]],dtype=int)
    return get_next_gen(neighbourSumBoard)

# 2) FOR
def get_which_survive_for():
    global board
    neighbourSumBoard = np.zeros((SIZEX,SIZEY),dtype=int)
    sum_blocks(board,neighbourSumBoard)
    return get_next_gen(neighbourSumBoard)

def sum_blocks(board:np.ndarray,resultBoard:np.ndarray):
    for x in np.arange(0,board.shape[0]):
        for y in np.arange(0,board.shape[1]):
            for z in np.arange(0,VECTORS.shape[0]):
                if (x + VECTORS[z,0] >= 0 and x + VECTORS[z,0] < board.shape[0] and 
                    y + VECTORS[z,1] >= 0 and y + VECTORS[z,1] < board.shape[1] and 
                    board[x + VECTORS[z,0],y +VECTORS[z,1]]):
                    resultBoard[x,y] += 1

# 3) JIT
def get_which_survive_jit():
    neighbourSumBoard = np.zeros((SIZEX,SIZEY),dtype=int)
    sum_blocks(board,neighbourSumBoard)
    return get_next_gen(neighbourSumBoard)

@jit(target_backend='cuda') 
def sum_blocks(board:np.ndarray,resultBoard:np.ndarray):
    for x in np.arange(0,board.shape[0]):
        for y in np.arange(0,board.shape[1]):
            for z in np.arange(0,VECTORS.shape[0]):
                if (x + VECTORS[z,0] >= 0 and x + VECTORS[z,0] < board.shape[0] and 
                    y + VECTORS[z,1] >= 0 and y + VECTORS[z,1] < board.shape[1] and 
                    board[x + VECTORS[z,0],y +VECTORS[z,1]]):
                    resultBoard[x,y] += 1


# 4) GPU
# optimalization, using this the array is transfered to the GPU only once at the start, and is only modified afterwards
neighbourSumBoardFlat = np.zeros((SIZEX*SIZEY,1),dtype=bool)
d_arr = cuda.to_device(neighbourSumBoardFlat)

def get_which_survive_gpu():
    global board
    # get necessary dimension for GPU blocks and grids
    blockdim = 32 # 32 beacause that is the maximum amount of threads that can be given to a core
    griddim = np.ceil(board.shape[0]*board.shape[1] / blockdim).astype(int)

    global d_arr
    # goes through the array as 1d and resizes it afterwards
    sum_blocks_gpu_1d[griddim,blockdim](board.flatten(),d_arr)
    return d_arr.copy_to_host().reshape((SIZEX,SIZEY))

@cuda.jit
def sum_blocks_gpu_1d(board:np.ndarray, resultBoard:np.ndarray):
    i = cuda.grid(1)
    if i < board.shape[0]:
        z = 0
        val = 0
        # go through all the neigbouring cells and add 1 for each life
        while z < VECTORS.shape[0]:
            index = VECTORS[z,0] + SIZEY*VECTORS[z,1]
            if (board[(i + index)%board.shape[0]]):# loops into the array indexes
                val += 1
            z+=1
        # different scenarios for cases with and without life
        if board[i]:
            # will it survive?
            if val >= COUNT_TO_LIVE and val < COUNT_TO_OVERCROWD:
                resultBoard[i] = True
            else:
                resultBoard[i] = False
        else:
            # will it revive
            if val == COUNT_TO_REVIVE:
                resultBoard[i] = True
            else:
                resultBoard[i] = False







if __name__ == "__main__":
    main()

    