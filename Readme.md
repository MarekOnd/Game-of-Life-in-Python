# Game of Life on GPU

This is an implementation of Game of Life in python using various approaches. One approach uses integration of [Python with CUDA](https://developer.nvidia.com/cuda-python) for best performance.

Main script `game_of_life.py` uses [Gooey](https://pypi.org/project/Gooey/) to parse settings and has all the implementations.
It is also able to export videos and it exports a screenshot at the end of the program.

Secondary script `game_of_life_only_gpu.py` contains only an optimized implementation with CUDA.

> [!CAUTION]
> Leaving program on with screen output for long periods of time can <span style="color:red">damage the monitor display</span>! It is possible to change the pixel color to gray in the code:
> ```py
> if args.show:
>   cv.imshow("img",board.astype(float))
>   #cv.imshow("img",board.astype(float)/2) # does not damage display
> ```

![preview](/last_image.jpg)
