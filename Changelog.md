# Changelog

## 0.1.1 (December 11, 2017)
Minor release for bugfixes and minor improvements. Changes include:
* Bug that was causing an incorrect link function to be used in the PyMC3 backend when fitting logistic models.
* Fixed handling of missing values in categorical variables.
* Fixed bug in set_priors() when passing numerical values for scale.
* Improved internal handling of custom priors.
* Preliminary Sphinx docs (WIP; thanks to @ejolly).

## 0.1.0 (March 31, 2017)
This is a major release that introduces several new features, significant API changes, and a large number of bug fixes and minor improvements. Notable changes include:
* Support for Stan as the sampling back-end (in addition to PyMC3), via the PyStan package.
* Dropped support for the `add_term` API; all model specification is now done via formulas.
* Expanded support for arbitrary random effects specifications; any formula now supported by patsy can be passed in as the left-hand side of a random effects specification (e.g., previously, '(a*b)|c' would not have worked).
* Completely refactored `Results` classes that no longer depend on PyMC3, providing a completely generic representation of sampler results, independent of any back-end.
* Refactored plotting and summary methods implemented on the abstract MCMCResults classes rather than at the back-end level.
* *Much* better compilation and sampling performance for models that include random effects with many levels. In many cases, performance should now be comparable to the most efficient native implementations of the models in the respective back-ends.
* All random effects priors now use the "non-centered" parameterization by default, significantly reducing bias for some models.
* Improved naming conventions that are more consistent with other packages (e.g., random effects now include the '|' operator in term names).
* Refactored `Term` class, including a separate subclass for `RandomTerm`s, and a number of other associated changes to the internal object model.
* Updated documentation and notebooks, including two new notebooks featuring well-developed examples (datasets included).
* Improved handling of NA values in continuous columns.
* Support for flat priors everywhere (by setting `auto_scale=False`).
* Numerous bug fixes and minor improvements

## 0.0.5 (January 17, 2017)
* Weakly informative default priors now work the same for all response families & link functions
* Minor bug fixes/tweaks

## 0.0.4 (October 11, 2016)
* Fixes referencing of Theano ops after PyMC3 namespace clean-up
* Added example Jupyter notebooks
* Improved handling of priors
* Improved prior plots and result summaries
* Improved access to MCMC trace results
* Add handling for datasets with NaN values
* Added travis-ci and coveralls support
* Minor bug fixes/tweaks

## 0.0.3 (September 4, 2016)
First official release.