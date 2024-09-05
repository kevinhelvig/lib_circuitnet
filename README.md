# lib_circuitnet

Open-source replication attempt of CircuitNet: [CircuitNet: A Generic Neural Network to Realize Universal Circuit Motif Modeling](https://openreview.net/pdf?id=Fl9q5z40e3), Wang et al., in Pytorch.

# Content :mag:
This repository proposes an attempt to replicate the CircuitNet architecture for image classification only, as a personal project. A main script is available to run training on MNIST and CIFAR-10, following the hyperparameters set by the paper. Another script proposes a "raw" genetic selection approach to set the hyperparameters, aiming to converge to the article's proposed ones (why not grid search? For "fun" :) ).

NB: The base core prototype of the module script was developed with ChatGPT-4, then iteratively and manually corrected and refined. Several differences and ambiguous points remain between this proposal and the paper's implementation. There is a minor gap between the accuracy of this script and the paper's performance of about 0.2%.

# How to run it ? :rocket:

1. Clone the repository:
   ```
   git clone https://github.com/kevinhelvig/lib_circuitnet.git
   ```
3. Navigate to the repository folder:
   ```
   cd lib_circuitnet
   ```
5. Install the required dependencies:
   ```
   pip install torch torchvision
   ```   
7. Run the main script:
   ```
   python main.py
   ```

# References and hints :question:
The [CircuitNet approach](https://openreview.net/pdf?id=Fl9q5z40e3) proposes a model that mimic more accurately how the brain looks to work regarding the "networks and graph theory" angle to study the brain : the different regions of the brain look to be organized following a ["small worlds" topology](https://pubmed.ncbi.nlm.nih.gov/17079517/), where densely connected nodules processes local or mono-modal information, with very sparse and limited connections between these different areas. 

Several interesting papers linked (to be completed)
- [Building artificial neural circuits for domain-general cognition: a primer on brain-inspired systems-level architecture](https://arxiv.org/abs/2303.13651) , Achterberg et al., 2023


# Future works ? :construction:

This repo should be considered primarily as a personal project and isn't intended to be pursued or extended further. However, several improvements might be added in the longer term for fun or by other coders:
- Replication of experiments proposed in reinforcement learning and/or forecasting?
- Modifications to the script to reproduce the paper more accurately?
- Adapt CircuitNet to NLP for token prediction?
- Come back to spiking neurons instead ? (more biologically plausible ?)
