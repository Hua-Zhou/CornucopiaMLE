# CornucopiaMLE

To reproduce the MLE examples in the manuscript `A Cornucopia of Maximum Likelihood Algorithms` by Kenneth Lange, Xun-Jian Li, and Hua Zhou:  
1. Git clone this repository.  
2. Start Julia at the directory `CornucopiaMLE`.  
3. Run examples by `include("Cornucopia.jl")` at Julia REPL.


# Sinkhorn's method

Figure 1 illustrates the color transfer results using Sinkhorn’s method and linear programming algorithm. For Sinkhorn’s method, the white cloud in the target image is accurately transferred to green, while the iron tower is transformed into white. In contrast, the LP algorithm produces a blurred transferred image, making it difficult to establish a clear relationship between the target and the transferred images. Moreover, Sinkhorn’s method efficiently handles high-dimensional images. 

![Figure 1](images/sinkhorn1.pdf)

Figure 2 presents the color-transferred image for $m=n=128$ using Sinkhorn’s method, demonstrating a clear and accurate transfer where the white cloud appears green, and the iron tower is distinctly transformed into white. We exclude the linear programming algorithm for m=n=128, as it is computationally infeasible to handle such a high-dimensional case on a laptop.


![Figure 2](images/sinkhorn2.pdf)
