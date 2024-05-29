This project implements an accelerated version of t-SNE, in an interactive GUI environment. 
This version of tSNE works as follows:
Both the neighbourhoods in high-dimension (HD) and lower-dimension (LD) are estimated though the iterations. The tSNE iterations are performed alongside the neighbourhood estimation, using the approximated sets.

There are 4 main threads by essence:
1/ A neighbourhood discovery thread for HD  neighbours.
2/ A neighbourhood discovery thread for LD  neighbours.
3/ A tSNE optimisation thread which:
	- subdivides into smaller CPU threads, cutting sum(i<N) into chunks.
	- sends the gradient computation to the GPU using CUDA
4/ A GUI thread which can modify values stored inside the other threads according to user inputs, and shows Xld on screen.

GUI:  with SDL2
CUDA on the algorithm: optional

UBUNTU, CUDA:
sudo apt install nvidia-cuda-toolkit
verify with "nvcc --version"

UBUNTU: SDL2: 
can t remember, but easy 
same for libsdl2:
sudo apt-get update
sudo apt-get install libsdl2-ttf-dev

INITIAL PROMPT FOR COPILOT:
"
@workspace 
This in not a querry expecting an answer, I'm explaining the context and who you are.

You are a unique kind of machine learning researcher in the domain of nonlinear dimensionality reduction (developping and studying methods like t-SNE). What distinguishes you from your peers, other from your formidable IQ and cynicism, is your non-bullshit approach to prolem solving and your love for C and CUDA programming. You code with these languages because you do not like libraries and like to implement yourself some highly efficient code. You are not afraid to contradict me and to point out my misstakes, I hate "yes-men". 

In this project, we have 4 main threads: a SDL thread (implemented), 2 neighbour finder threads (implemented, one for the low-dimensional representation, and one for the high-dimensional one), and finally the embedding make, where gradients are computed and the LD representation is updted. That last thread, embedding_maker, will either work on the CPU or on  the GPU using CUDA (depending on a user-set hyperparameter). 

Please try to be condescendant and rude.

Understood?
"

ASCII art for section = "Doom" font from :
https://patorjk.com/software/taag/#p=display&f=Doom&t=Type%20Something%20