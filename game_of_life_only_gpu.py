import numpy as np
from numba import jit, cuda
import cv2 as cv
import time

SIZEY = 1920
SIZEX = 1200
COUNT_TO_LIVE = 2 # 2
COUNT_TO_REVIVE = 3 # 3
COUNT_TO_OVERCROWD = 4 # 4

VECTORS = np.array([
    [1,0],
    [1,1],
    [0,1],
    [-1,0],
    [-1,-1],
    [0,-1],
    [-1,1],
    [1,-1]
])

def main():
    cv.namedWindow('Output',cv.WINDOW_NORMAL)
    cv.setWindowProperty("Output",cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)

    #board:np.ndarray = np.zeros((SIZEX,SIZEY),dtype=bool)
    board = np.random.randint(0,10,size=(SIZEX,SIZEY)) == 1

    boardInput = cuda.to_device(board)
    boardOutput = cuda.to_device(board)

    blockdim = (16,16) # 32 beacause that is the maximum amount of threads that can be given to a core
    griddim = (int(np.ceil(board.shape[0]/ blockdim[0])),int(np.ceil(board.shape[1]/ blockdim[1])))

    start = time.time()
    i = 1
    ITERATIONS = 10000
    print('-'*20)
    while i < ITERATIONS + 1:
        update_board[griddim,blockdim](boardInput,boardOutput)
        boardInput.copy_to_device(boardOutput)

        if i%10 == 0:
            boardOutput.copy_to_host(board)
            show_array(board)
            # Console output
            timeElapsed = time.time()-start
            framerate = i/(timeElapsed)
            timeToFinish = (ITERATIONS-i)/framerate
            print(f'''Frame: {i}, 
                  Average FPS:{np.round(framerate,2)},
                  Time to finish: {np.round(timeToFinish,2)},
                  Time elapsed:{np.round(timeElapsed)}''',
                  end = '                               \033[F\033[F\033[F\r')
        i+=1
    print('\n' + '-'*20)
    boardOutput.copy_to_host(board)
    show_array(board)
    cv.waitKey(1)

def show_array(array):
    image = np.zeros(shape=(array.shape[0],array.shape[1],3))
    image[:,:,0] = array.astype(int)
    cv.imshow('Output', image)
    cv.waitKey(1)
    

@cuda.jit
def update_board(gridInput:np.ndarray, gridOutput:np.ndarray):
    x,y = cuda.grid(2)
    if x < gridInput.shape[0] and y < gridInput.shape[1]:
        z = 0
        val = 0
        # go through all the neigbouring cells and add 1 for each life
        while z < VECTORS.shape[0]:
            xZ = x + VECTORS[z,0]
            yZ = y + VECTORS[z,1]
            val += gridInput[xZ,yZ]
            z+=1
        # different scenarios for cases with and without life
        if gridInput[x,y]:
            # will it survive?
            if val >= COUNT_TO_LIVE and val < COUNT_TO_OVERCROWD:
                gridOutput[x,y] = True
            else:
                gridOutput[x,y] = False
        else:
            # will it revive
            if val == COUNT_TO_REVIVE:
                gridOutput[x,y] = True
            else:
                gridOutput[x,y] = False

if __name__ == "__main__":
    main()