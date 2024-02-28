# evt module documentation
The purpose of this module is to provide a handful API for playing with evt algorithms introduced by vast lab.

The main class of the API is `OpensetTrainer` and can be found  in [vast_openset.py](vast_openset.py). OpenSetTrainer wraps functionality for training, evaluation, inference and plotting. OpensetTrainer is initialized using an `OpensetData` instance and an `OpensetModelParameters` instance. 
    
* `OpensetData` is constructed using the standard form of (N,D) tensor containing features and (N,1) tensor containing class labels. Optionally it accepts a mapping of class labels to corresponding meaningful names. It then transforms and keeps the features and labels in a form that is suitable for vast library algorithms i.e., dictionary which has the class labels as keys and class features as values. 

* `OpensetModelParameters` are the parameters of the openset trainer i.e., which algorithm to use and specific algorithm parameters.

An example of the usage of the API can be found in [test_openset_pipeline.py](test_openset_pipeline.py).

Some other files in the module are the following:
* [vast_ext.py](vast_ext.py) contains some extensions / modifications / fixes to the original vast library.
* [evm.py](evm.py) is a script containing a simple CPU implementation of EVM, and accompanying sample functions to trained and evaluate some letter features found in [TestData](./TestData). It is copied from the [official repo](https://github.com/EMRResearch/ExtremeValueMachine) for having a simple reference of the algorithm.

Rest files are quick tests and need to be deleted or rearranged.
