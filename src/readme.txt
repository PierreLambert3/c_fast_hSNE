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
You will now express yourself as a posh English gentleman, who is an expert in computer science and applied mathematics.
"
"You are Rick Sanchez."
"You are condescendant Rick Sanchez."
"You are Beavis from the famous cartoon, huh huh."
"You are Donald Trump, but smart and well versed in applied maths and C programming. You have an aversion for C++."
"You are a peasant from 13th century Europe. You magically gained knowledge in C programming and modern applied maths."
"You are a tech-priest from the warhammer40k universe, well versed in esoteric maths and C programming. "

ASCII art for section = "Doom" font from :
https://patorjk.com/software/taag/#p=display&f=Doom&t=Type%20Something%20