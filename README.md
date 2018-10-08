# MAS
Humans can learn in a continuous manner. Old rarely utilized knowledge can be overwritten by new incoming information while important, frequently used knowledge is prevented from being erased. In artificial learning systems,
lifelong learning so far has focused mainly on accumulating knowledge over tasks and overcoming catastrophic forgetting. In this paper, we argue that, given the limited model capacity and the unlimited new information to be learned, knowl-
edge has to be preserved or erased selectively. Inspired by neuroplasticity, we propose a novel approach for lifelong learning, coined Memory Aware Synapses(MAS). It computes the importance of the parameters of a neural network in an
unsupervised and online manner. Given a new sample which is fed to the network,MAS accumulates an importance measure for each parameter of the network,  based  on  how  sensitive  the  predicted  output  function  is  to  a  change  in
this parameter. When learning a new task, changes to important parameters can then be penalized, effectively preventing important knowledge related to previous tasks from being overwritten. Further, we show an interesting connection between
a local version of our method and Hebb’s rule, which is a model for the learning process  in  the  brain.  We  test  our  method  on  a  sequence  of  object  recognition tasks and on the challenging problem of learning an embedding for predicting
<subject, predicate, object> triplets. We show state-of-the-art performance and, for the first time, the ability to adapt the importance of the parameters based on unlabeled data towards what the network needs (not) to forget, which may vary
depending on test conditions.

![Global Model](https://raw.githubusercontent.com/rahafaljundi/MAS-Memory-Aware-Synapses/master/teaser_fig.png)

This directory contains a pytorch implementation of Memory Aware Synapses: Learning what not to forget method. A demo file that shows a learning scenario in mnist split set of tasks is included.

## Authors

Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach and Tinne Tuytelaars


For questions about the code, please contact me, Rahaf Aljundi (rahaf.aljundi@esat.kuleuven.be)
## Requirements
The code was built using pytorch version 0.3, python 3.5 and cuda 9.1
## Citation
Aljundi R., Babiloni F., Elhoseiny M., Rohrbach M., Tuytelaars T. (2018) Memory Aware Synapses: Learning What (not) to Forget. In:  Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science, vol 11207. Springer, Cham

## License

This software package is freely available for research purposes.
