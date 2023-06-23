Frequently Asked Questions
**************************

General Questions
===================

What is the difference between Bambi and PyMC?
----------------------------
    * Bambi is a regression library built on top of PyMC. It provides a simple 
        interface for specifying Bayesian models, and allows for easy inference using MCMC or 
        variational inference.
    * PyMC is a library for Bayesian modelling, and is the backend used by Bambi. It is a very
            powerful library, but can be challenging to use for beginners. Bambi provides a simple
            interface for specifying models, and allows for easy inference via MCMC or variational
            inference using PyMC.


Inference Questions
====================

What sampling methods are available?
----------------------------
The sampler used is automatically selected given the type of variables used in the model.
For inference, Bambi supports both MCMC and variational inference. MCMC is the default, but you can specify variational inference by passing `inference_method='vi'` to `Model.fit()`.
Bambi also supports multiple backends for MCMC, including NumPyro, and BlackJax
(see API for "fit" method for more details `here <https://bambinos.github.io/bambi/api_reference.html>`_).

Can inference in Bambi be sped up using GPUs/TPUs?
----------------------------
Yes, Bambi supports inference on GPUs and TPUs using the numpyro and blackjax backends. 
See the API for "fit" method for more details 
`here <https://bambinos.github.io/bambi/api_reference.html>`_.

Model Specification Questions
====================

My data has a non-normal distributions, can I still use Bambi?
----------------------------
Yes, Bambi supports a wide range of distributions which can be specified using the "family"
argument to the "Model". You can find examples of how to specify these distributions 
in the `Bambi examples <https://bambinos.github.io/bambi/examples.html>`_.

How do I find out what priors are available?
----------------------------
You can use any valid PyMC distribution as a prior. You can find a list of all the distributions available in PyMC `here <https://www.pymc.io/projects/docs/en/stable/api/distributions.html>`_.  You can also find examples of how to specify priors in the `Bambi examples <https://bambinos.github.io/bambi/examples.html>`_, and in the `Getting Started Guide <https://bambinos.github.io/bambi/notebooks/getting_started.html#Specifying-priors>`_.

Does bambi come with pre-specified regression models?
----------------------------
To allow building of bespoke models, Bambi does not come with pre-specified regression models.
However, you can find examples of how to specify models in the 
`Bambi examples <https://bambinos.github.io/bambi/examples.html>`_.