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

This directory contains a pytorch implementation of Memory Aware Synapses: Learning what not to forget method. It allows a neural network to learn in a continual manner without catastrophic forgetting. Moreover, at deployment time while the system is applied to a set of images, the method learns the important parts of the tasks that shouldn't be forgotten allowing extra freedom for later tasks.A demo file that shows a learning scenario in mnist split set of tasks is included.

## Authors

Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach and Tinne Tuytelaars


For questions about the code, please contact me, Rahaf Aljundi (rahaf.aljundi@esat.kuleuven.be)

## Citation
```bibtex
@article{aljundi2017memory,
  title={Memory Aware Synapses: Learning what (not) to forget},
  author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
  journal={arXiv preprint arXiv:1711.09601},
  year={2017}
}
```

## License

This software package is freely available for research purposes. Please check the LICENSE file for details.

