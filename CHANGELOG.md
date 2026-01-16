<a id="0.17.0"></a>
# [The sparse bambino (0.17.0)](https://github.com/bambinos/bambi/releases/tag/0.17.0) - 2026-01-16

## What's Changed

### New features

* Support sparse design matrix for group specific effects by [@tomicapretto](https://github.com/tomicapretto) in [#950](https://github.com/bambinos/bambi/pull/950)
* Add `SPARSE_DOT` configuration variable by [@tomicapretto](https://github.com/tomicapretto) in [#950](https://github.com/bambinos/bambi/pull/950)

### Maintenance and fixes

* Add interlinks by [@tomicapretto](https://github.com/tomicapretto) in [#946](https://github.com/bambinos/bambi/pull/946)
* Begin migration to "new arviz" by [@aloctavodia](https://github.com/aloctavodia) in [#945](https://github.com/bambinos/bambi/pull/945)
* Make quartodoc build verbose and copy objects.json to _site by [@tomicapretto](https://github.com/tomicapretto) in [#949](https://github.com/bambinos/bambi/pull/949)
* Fix non-deterministic predictions when `sample_new_groups=True` and `random_seed` is set by [@Delogon](https://github.com/Delogon) in [#954](https://github.com/bambinos/bambi/pull/954)
* Update how data is read from figshare by [@tomicapretto](https://github.com/tomicapretto) in [#956](https://github.com/bambinos/bambi/pull/956)

### Documentation

* Correct typos in Polynomial Regression notebook by [@star1327p](https://github.com/star1327p) in [#940](https://github.com/bambinos/bambi/pull/940)
* Updated Hierarchical Linear Regression (Pigs dataset) example ([#484](https://github.com/bambinos/bambi/issues/484)) by [@pranavduraisamy](https://github.com/pranavduraisamy) in [#939](https://github.com/bambinos/bambi/pull/939)
* Add entry in the example gallery linking to kulprit documentation by [@aloctavodia](https://github.com/aloctavodia) in [#944](https://github.com/bambinos/bambi/pull/944)
* Correct grammar issues regarding a/an usage by [@star1327p](https://github.com/star1327p) in [#952](https://github.com/bambinos/bambi/pull/952)

### Deprecation

* Remove `tan_2` link function for vonmises by [@aloctavodia](https://github.com/aloctavodia) in [#943](https://github.com/bambinos/bambi/pull/943)

## New Contributors

* [@pranavduraisamy](https://github.com/pranavduraisamy) made their first contribution in [#939](https://github.com/bambinos/bambi/pull/939)
* [@Delogon](https://github.com/Delogon) made their first contribution in [#954](https://github.com/bambinos/bambi/pull/954)

**Full Changelog**: https://github.com/bambinos/bambi/compare/0.16.0...0.17.0

[Changes][0.17.0]


<a id="0.16.0"></a>
# [The awakening bambino (0.16.0)](https://github.com/bambinos/bambi/releases/tag/0.16.0) - 2025-10-24

## What's Changed
* Fix broken links by [@B-Deforce](https://github.com/B-Deforce) in [#866](https://github.com/bambinos/bambi/pull/866)
* Fix random seed handling for "vi" by [@B-Deforce](https://github.com/B-Deforce) in [#869](https://github.com/bambinos/bambi/pull/869)
* typo fix -> `SETTINGS_FAMILIES` to `BUILTIN_FAMILIES` by [@Schefflera-Arboricola](https://github.com/Schefflera-Arboricola) in [#873](https://github.com/bambinos/bambi/pull/873)
* Add post-release.yml to update changelogs automatically by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#882](https://github.com/bambinos/bambi/pull/882)
* Adjustments to the negative binomial tutorial by [@tomicapretto](https://github.com/tomicapretto) in [#890](https://github.com/bambinos/bambi/pull/890)
* Changing to nutpie sampler on mr p example by [@NathanielF](https://github.com/NathanielF) in [#895](https://github.com/bambinos/bambi/pull/895)
* Correct a name accent in the LaTeX citation by [@star1327p](https://github.com/star1327p) in [#897](https://github.com/bambinos/bambi/pull/897)
* Add random_seed argument to predict by [@aloctavodia](https://github.com/aloctavodia) in [#892](https://github.com/bambinos/bambi/pull/892)
* Add link to the R package modelr by [@star1327p](https://github.com/star1327p) in [#899](https://github.com/bambinos/bambi/pull/899)
* Add R2 by [@aloctavodia](https://github.com/aloctavodia) in [#896](https://github.com/bambinos/bambi/pull/896)
* Update the link for "Keep it Maximal" paper by [@star1327p](https://github.com/star1327p) in [#901](https://github.com/bambinos/bambi/pull/901)
* Add link to the book "Causal Inference: What if" by [@star1327p](https://github.com/star1327p) in [#902](https://github.com/bambinos/bambi/pull/902)
* Updated links to PyMC and Bambi pages by [@star1327p](https://github.com/star1327p) in [#903](https://github.com/bambinos/bambi/pull/903)
* Fix data links to fish and marginaleffects by [@star1327p](https://github.com/star1327p) in [#904](https://github.com/bambinos/bambi/pull/904)
* Plot predictions categorical error by [@GStechschulte](https://github.com/GStechschulte) in [#905](https://github.com/bambinos/bambi/pull/905)
* DOC: Update three URLs in the Jupyter Notebooks by [@star1327p](https://github.com/star1327p) in [#909](https://github.com/bambinos/bambi/pull/909)
* Added the link to Strack RRR paper by [@star1327p](https://github.com/star1327p) in [#910](https://github.com/bambinos/bambi/pull/910)
* DOC Updated links in Robust Linear Regression by [@star1327p](https://github.com/star1327p) in [#913](https://github.com/bambinos/bambi/pull/913)
* DOC: Correct a few typos in the API Reference by [@star1327p](https://github.com/star1327p) in [#915](https://github.com/bambinos/bambi/pull/915)
* DOC: Clean up a few Bambi Example Notebooks by [@star1327p](https://github.com/star1327p) in [#916](https://github.com/bambinos/bambi/pull/916)
* DOC: Continue cleaning up the notebooks by [@star1327p](https://github.com/star1327p) in [#917](https://github.com/bambinos/bambi/pull/917)
* Remove bayeux for accessing alternative sampler backends by [@GStechschulte](https://github.com/GStechschulte) in [#919](https://github.com/bambinos/bambi/pull/919)
* Drop Python 3.10 add 3.13 by [@aloctavodia](https://github.com/aloctavodia) in [#922](https://github.com/bambinos/bambi/pull/922)
* DOC: Correct more typos in the Jupyter Notebooks by [@star1327p](https://github.com/star1327p) in [#924](https://github.com/bambinos/bambi/pull/924)
* Use mock sampling in tests and incorporate other improvements to the test suite. by [@tomicapretto](https://github.com/tomicapretto) in [#923](https://github.com/bambinos/bambi/pull/923)
* Modernize development infraestructure by [@tomicapretto](https://github.com/tomicapretto) in [#925](https://github.com/bambinos/bambi/pull/925)
* Update docs by [@tomicapretto](https://github.com/tomicapretto) in [#926](https://github.com/bambinos/bambi/pull/926)
* Fix build status badge by [@tomicapretto](https://github.com/tomicapretto) in [#929](https://github.com/bambinos/bambi/pull/929)
* DOC: Make a long equation into two lines by [@star1327p](https://github.com/star1327p) in [#931](https://github.com/bambinos/bambi/pull/931)
* DOC: Correct typos in the narrative of Examples by [@star1327p](https://github.com/star1327p) in [#933](https://github.com/bambinos/bambi/pull/933)
* Double-check existing docstrings + start preparing release by [@tomicapretto](https://github.com/tomicapretto) in [#935](https://github.com/bambinos/bambi/pull/935)
* DOC: Improve the formatting of Bambi examples by [@star1327p](https://github.com/star1327p) in [#936](https://github.com/bambinos/bambi/pull/936)
* Prepare release by [@tomicapretto](https://github.com/tomicapretto) in [#938](https://github.com/bambinos/bambi/pull/938)

## New Contributors
* [@B-Deforce](https://github.com/B-Deforce) made their first contribution in [#866](https://github.com/bambinos/bambi/pull/866)
* [@Schefflera-Arboricola](https://github.com/Schefflera-Arboricola) made their first contribution in [#873](https://github.com/bambinos/bambi/pull/873)
* [@rohanbabbar04](https://github.com/rohanbabbar04) made their first contribution in [#882](https://github.com/bambinos/bambi/pull/882)

**Full Changelog**: https://github.com/bambinos/bambi/compare/0.15.0...0.16.0

[Changes][0.16.0]


<a id="0.15.0"></a>
# [Release 0.15.0](https://github.com/bambinos/bambi/releases/tag/0.15.0) - 2024-12-21

### New features

* Add default priors for binomial and bernoulli families with logit link ([#830](https://github.com/bambinos/bambi/issues/830))
* Add horseshoe prior ([#836](https://github.com/bambinos/bambi/issues/836))
* Handle multivariate responses with HSGP ([#856](https://github.com/bambinos/bambi/issues/856))

### Maintenance and fixes

* Change the JAX random number generator key for 32 bit systems ([#833](https://github.com/bambinos/bambi/issues/833))
* Change `rename` to `replace` in `pre-render.py` ([#843](https://github.com/bambinos/bambi/issues/843))
* Fix out of sample prediction for multivariate families. It would not work for tables where the
number of rows was different from the one used to fit the model ([#847](https://github.com/bambinos/bambi/issues/847))
* Check variables before trying to access them in posterior predictive sampling ([#851](https://github.com/bambinos/bambi/issues/851))
* Pass kwargs to nutpie + create env.yml file ([#855](https://github.com/bambinos/bambi/issues/855))

### Documentation

* Fix typos and incomplete doc strings ([#765](https://github.com/bambinos/bambi/issues/765))
* Clarify elpd differences interepretation ([#825](https://github.com/bambinos/bambi/issues/825))
* Fix the contributing readme link ([#837](https://github.com/bambinos/bambi/issues/837))
* Add example using `offset` ([#842](https://github.com/bambinos/bambi/issues/842))
* Fix model formula in negative binomial notebook ([#859](https://github.com/bambinos/bambi/issues/859))
* Fix formatting in t-test examples ([#861](https://github.com/bambinos/bambi/issues/861))
* Fix issue 812 Broken link ([#862](https://github.com/bambinos/bambi/issues/862))
* Update repository documentation files ([#865](https://github.com/bambinos/bambi/issues/865))

[Changes][0.15.0]


<a id="0.14.0"></a>
# [Release 0.14.0](https://github.com/bambinos/bambi/releases/tag/0.14.0) - 2024-07-10

### New features

* Add configuration facilities to Bambi ([#745](https://github.com/bambinos/bambi/issues/745)) 
* Interpet submodule now outputs informative messages when computing default values ([#745](https://github.com/bambinos/bambi/issues/745)) 
* Bambi supports weighted responses ([#761](https://github.com/bambinos/bambi/issues/761))
* Bambi supports constrained responses ([#764](https://github.com/bambinos/bambi/issues/764))
* Implement `compute_log_likelihood()` method to compute the log likelihood on a model ([#769](https://github.com/bambinos/bambi/issues/769))
* Add a class `InferenceMethods` that allows users to access the available inference methods and kwargs ([#795](https://github.com/bambinos/bambi/issues/795))

### Maintenance and fixes

* Fix bug in predictions with models using HSGP ([#780](https://github.com/bambinos/bambi/issues/780))
* Fix `get_model_covariates()` utility function ([#801](https://github.com/bambinos/bambi/issues/801))
* Use `pm.compute_deterministics()` to compute deterministics when bayeux based samplers are used ([#803](https://github.com/bambinos/bambi/issues/803))
* Wrap all the parameters of the response distribution (the likelihood) with a `pm.Deterministic` ([#804](https://github.com/bambinos/bambi/issues/804))
* Keep `bayeux-ml` as the single direct JAX-related dependency ([#804](https://github.com/bambinos/bambi/issues/804))
* The response component only holds response information about the response, not about predictors of the parent parameter ([#804](https://github.com/bambinos/bambi/issues/804))
* Resolve import error associated with bayeux ([#822](https://github.com/bambinos/bambi/issues/822))

### Documentation

* Our Code of Conduct now includes how to send a report ([#783](https://github.com/bambinos/bambi/issues/783))
* Add polynomial regression example ([#809](https://github.com/bambinos/bambi/issues/809))
* Add Contact form to our webpage ([#816](https://github.com/bambinos/bambi/issues/816))

### Deprecation

* `f"{response_name}_obs"` has been replaced by `"__obs__"` as the dimension name for the observation index ([#804](https://github.com/bambinos/bambi/issues/804))
* `f"{response_name}_{parameter_name}"` is no longer the name for the name of parameters of the likelihood. Now Bambi uses `"{parameter_name}"` ([#804](https://github.com/bambinos/bambi/issues/804))
* `kind` in `Model.predict()` now use `"response_params"` and `"response"` instead of `"mean"` and `"pps"` ([#804](https://github.com/bambinos/bambi/issues/804))
* `include_mean` has been replaced by `include_response_params` in `Model.fit()` ([#804](https://github.com/bambinos/bambi/issues/804))

[Changes][0.14.0]


<a id="0.13.0"></a>
# [Bambi 0.13.0](https://github.com/bambinos/bambi/releases/tag/0.13.0) - 2023-10-25

This is the first version of Bambi that is released with a Governance structure. Added in [#709](https://github.com/bambinos/bambi/issues/709).
The highlights are the shiny `interpret` subpackage and the implementation of support for censored models.

### New features

* Bambi now supports censored responses ([#697](https://github.com/bambinos/bambi/issues/697))
* Implement `"exponential"` and `"weibull"` families ([#697](https://github.com/bambinos/bambi/issues/697))
* Add `"kidney"` dataset ([#697](https://github.com/bambinos/bambi/issues/697))
* Add `interpret` submodule ([#684](https://github.com/bambinos/bambi/issues/684), [#695](https://github.com/bambinos/bambi/issues/695), [#699](https://github.com/bambinos/bambi/issues/699), [#701](https://github.com/bambinos/bambi/issues/701), [#732](https://github.com/bambinos/bambi/issues/732), [#736](https://github.com/bambinos/bambi/issues/736))
    * Implements `comparisons`, `predictions`, `slopes`, `plot_comparisons`, `plot_predictions`, and `plot_slopes`
* Support censored families 

### Maintenance and fixes

* Replace `univariate_ordered` with `ordered` ([#724](https://github.com/bambinos/bambi/issues/724))
* Add missing docstring for `center_predictors` ([#726](https://github.com/bambinos/bambi/issues/726))
* Fix bugs in `plot_comparison` ([#731](https://github.com/bambinos/bambi/issues/731))

### Documentation

* Add docstrings to utility functions ([#696](https://github.com/bambinos/bambi/issues/696))
* Migrate documentation to Quarto ([#712](https://github.com/bambinos/bambi/issues/712))
* Add case study for MRP ([#716](https://github.com/bambinos/bambi/issues/716))
* Add example about ordinal regression ([#719](https://github.com/bambinos/bambi/issues/719))
* Add example about zero inflated models ([#725](https://github.com/bambinos/bambi/issues/725))
* Add example about predictions for new groups ([#734](https://github.com/bambinos/bambi/issues/734))

### Deprecation

* Drop official suport for Python 3.8 ([#720](https://github.com/bambinos/bambi/issues/720))
* Change `plots` submodule name to `interpret` ([#705](https://github.com/bambinos/bambi/issues/705))

[Changes][0.13.0]


<a id="0.12.0"></a>
# [Bambi 0.12.0: Ordinal models and predictions on new groups](https://github.com/bambinos/bambi/releases/tag/0.12.0) - 2023-07-02

## 0.12.0

### New features

* Implement new families `"ordinal"` and `"sratio"` for modeling of ordinal responses ([#678](https://github.com/bambinos/bambi/issues/678))
* Allow families to implement a custom `create_extra_pps_coord()` ([#688](https://github.com/bambinos/bambi/issues/688))
* Allow predictions on new groups ([#693](https://github.com/bambinos/bambi/issues/693))

### Maintenance and fixes

* Robustify how Bambi handles dims ([#682](https://github.com/bambinos/bambi/issues/682))
* Fix links in FAQ ([#686](https://github.com/bambinos/bambi/issues/686))
* Update additional dependencies install command ([#689](https://github.com/bambinos/bambi/issues/689))
* Update predict pps docstring ([#690](https://github.com/bambinos/bambi/issues/690))
* Add warning for aliases athat aren't used ([#691](https://github.com/bambinos/bambi/issues/691))

### Documentation

* Add families to the Getting Started guide ([#683](https://github.com/bambinos/bambi/issues/683))

[Changes][0.12.0]


<a id="0.11.0"></a>
# [Bambi 0.11.0: The family grows](https://github.com/bambinos/bambi/releases/tag/0.11.0) - 2023-05-25

## 0.11.0

### New features

* Add support for Gaussian Processes via the HSGP approximation ([#632](https://github.com/bambinos/bambi/issues/632)) 
* Add new families: `"zero_inflated_poisson"`, `"zero_inflated_binomial"`, and `"zero_inflated_negativebinomial"` ([#654](https://github.com/bambinos/bambi/issues/654))
* Add new families: `"beta_binomial"` and `"dirichlet_multinomial"` ([#659](https://github.com/bambinos/bambi/issues/659))
* Allow `plot_cap()` to show predictions at the observation level ([#668](https://github.com/bambinos/bambi/issues/668))
* Add new families: `"hurdle_gamma"`, `"hurdle_lognormal"`, `"hurdle_negativebinomial"`, and `"hurdle_poisson"` ([#676](https://github.com/bambinos/bambi/issues/676))

### Maintenance and fixes

* Modify how HSGP is built in PyMC when there are groups ([#661](https://github.com/bambinos/bambi/issues/661))
* Modify how Bambi is imported in the tests ([#662](https://github.com/bambinos/bambi/issues/662))
* Prevent underscores from being removed in dim names ([#664](https://github.com/bambinos/bambi/issues/664))
* Bump sphinx dependency to a version greater than 7 ([#672](https://github.com/bambinos/bambi/issues/672))

### Documentation

* Document how to use custom priors ([#656](https://github.com/bambinos/bambi/issues/656))
* Fix name of arviz traceplot function in the docs ([#666](https://github.com/bambinos/bambi/issues/666))
* Add example that shows how `plot_cap()` works ([#670](https://github.com/bambinos/bambi/issues/670))

[Changes][0.11.0]


<a id="0.10.0"></a>
# [Bambi 0.10.0](https://github.com/bambinos/bambi/releases/tag/0.10.0) - 2023-02-10

### New features

* Refactored the codebase to support distributional models ([#607](https://github.com/bambinos/bambi/issues/607))
* Added a default method to handle posterior predictive sampling for custom families ([#625](https://github.com/bambinos/bambi/issues/625))
* `plot_cap()` gains a new argument `target` that allows to plot different parameters of the response distribution ([#627](https://github.com/bambinos/bambi/issues/627))

### Maintenance and fixes

* Moved the `tests` directory to the root of the repository ([#607](https://github.com/bambinos/bambi/issues/607))
* Don't pass `dims` to the response of the likelihood distribution anymore ([#629](https://github.com/bambinos/bambi/issues/629))
* Remove requirements.txt and replace with `pyproject.toml` config file to distribute the package ([#631](https://github.com/bambinos/bambi/issues/631))

### Documentation

* Update examples to work with the new internals ([#607](https://github.com/bambinos/bambi/issues/607))
* Fixed figure in the Sleepstudy example ([#607](https://github.com/bambinos/bambi/issues/607))
* Add example using distributional models ([#641](https://github.com/bambinos/bambi/issues/641))

### Deprecation

* Removed versioned documentation webpage ([#616](https://github.com/bambinos/bambi/issues/616))
* Removed correlated priors for group-specific terms ([#607](https://github.com/bambinos/bambi/issues/607))
* Dictionary with tuple keys are not allowed for priors anymore ([#607](https://github.com/bambinos/bambi/issues/607))

[Changes][0.10.0]


<a id="0.9.3"></a>
# [Bambi 0.9.3](https://github.com/bambinos/bambi/releases/tag/0.9.3) - 2022-12-21

### Maintenance and fixes

* Update to PyMC >= 5, which means we use PyTensor instead of Aesara now ([#613](https://github.com/bambinos/bambi/issues/613), [#614](https://github.com/bambinos/bambi/issues/614))

[Changes][0.9.3]


<a id="0.9.2"></a>
# [Bambi 0.9.2](https://github.com/bambinos/bambi/releases/tag/0.9.2) - 2022-12-09

### New features

* Implement `censored()` ([#581](https://github.com/bambinos/bambi/issues/581))
* Add `Formula` class ([#585](https://github.com/bambinos/bambi/issues/585))
* Add common numpy transforms to extra_namespace ([#589](https://github.com/bambinos/bambi/issues/589))
* Add `AsymmetricLaplace` family for Quantile Regression ([#591](https://github.com/bambinos/bambi/issues/591))
* Add 'transforms' argument to `plot_cap()` ([#594](https://github.com/bambinos/bambi/issues/594))
* Add panel covariates to `plot_cap()` and make it more flexible ([#596](https://github.com/bambinos/bambi/issues/596))

### Maintenance and fixes

* Reimplemented predictions to make better usage of xarray data structures ([#573](https://github.com/bambinos/bambi/issues/573))
* Keep 0 dimensional parameters as 0 dimensional instead of 1 dimensional ([#575](https://github.com/bambinos/bambi/issues/575))
* Refactor terms for modularity and extensibility ([#582](https://github.com/bambinos/bambi/issues/582))
* Remove seed argument from `model.initial_point()` ([#592](https://github.com/bambinos/bambi/issues/592))
* Add build check function on prior predictive and plot prior ([#605](https://github.com/bambinos/bambi/issues/605))

### Documentation

* Add quantile regression example ([#608](https://github.com/bambinos/bambi/issues/608))

### Deprecation

* Remove `automatic_priors` argument from `Model` ([#603](https://github.com/bambinos/bambi/issues/603))
* Remove string option for data input in `Model` ([#604](https://github.com/bambinos/bambi/issues/604))

[Changes][0.9.2]


<a id="0.9.1"></a>
# [Bambi 0.9.1](https://github.com/bambinos/bambi/releases/tag/0.9.1) - 2022-08-27

## Bambi 0.9.1

### New features
-  Add support for jax sampling via numpyro and blackjax  samplers ([#526](https://github.com/bambinos/bambi/issues/526))
-  Add Laplace family ([#524](https://github.com/bambinos/bambi/issues/524)) 
-  Improve Laplace computation and integration ([#555](https://github.com/bambinos/bambi/issues/555) and [#563](https://github.com/bambinos/bambi/issues/563))


### Maintenance and fixes
-  Ensure order variable is preserved when ploting priors ([#529](https://github.com/bambinos/bambi/issues/529))
-  Treat offset accordingly ([#534](https://github.com/bambinos/bambi/issues/534))
-  Refactor tests to share data generation code ([#531](https://github.com/bambinos/bambi/issues/531))


### Documentation
-  Update documentation following good inferencedata practices ([#537](https://github.com/bambinos/bambi/issues/537))
-  Add logos to repo and docs ([#542](https://github.com/bambinos/bambi/issues/542))

### Deprecation
-  Deprecate method argument in favor of inference_method ([#554](https://github.com/bambinos/bambi/issues/554))


[Changes][0.9.1]


<a id="0.9.0"></a>
# [Bambi 0.9.0](https://github.com/bambinos/bambi/releases/tag/0.9.0) - 2022-06-06


### New features

- Bambi now uses [PyMC 4.0](https://www.pymc.io/blog/v4_announcement.html) as it's backend. Most if not all your previous model should run the same, without  the need of any change.
-  Add Plot Conditional Adjusted Predictions `plot_cap` ([#517](https://github.com/bambinos/bambi/issues/517))

### Maintenance and fixes
-  Group specific terms now work with numeric of multiple columns ([#516](https://github.com/bambinos/bambi/issues/516)) 

[Changes][0.9.0]


<a id="0.8.0"></a>
# [Bambi 0.8.0](https://github.com/bambinos/bambi/releases/tag/0.8.0) - 2022-05-18

## Bambi 0.8.0

### New features

- Add VonMises (`"vonmises"`) built-in family ([#453](https://github.com/bambinos/bambi/issues/453))
- `Model.predict()` gains a new argument `include_group_specific` to determine if group-specific effects are considered when making predictions ([#470](https://github.com/bambinos/bambi/issues/470))
- Add Multinomial (`"multinomial"`) built-in family ([#490](https://github.com/bambinos/bambi/issues/490))

### Maintenance and fixes

- Add posterior predictive sampling method to "categorical" family ([#458](https://github.com/bambinos/bambi/issues/458))
- Require Python >= 3.7.2 to fix NoReturn type bug in Python ([#463](https://github.com/bambinos/bambi/issues/463))
- Fixed the wrong builtin link given by `link="inverse"` was wrong. It returned the same result as `link="cloglog"` ([#472](https://github.com/bambinos/bambi/issues/472))
- Replaced plain dictionaries with `namedtuple`s when same dictionary structure was repeated many times ([#472](https://github.com/bambinos/bambi/issues/472))
- The function `check_full_rank()` in `utils.py` now checks the array is 2 dimensional ([#472](https://github.com/bambinos/bambi/issues/472))
- Removed `_extract_family_prior()` from `bambi/families` as it was unnecesary ([#472](https://github.com/bambinos/bambi/issues/472))
- Removed `bambi/families/utils.py` as it was unnecessary ([#472](https://github.com/bambinos/bambi/issues/472))
- Removed external links and unused datasets ([#483](https://github.com/bambinos/bambi/issues/483))
- Replaced `"_coord_group_factor"` with `"__factor_dim"` and `"_coord_group_expr"` with `"__expr_dim"` in dimension/coord names ([#499](https://github.com/bambinos/bambi/issues/499))
- Fixed a bug related to modifying the types of the columns in the original data frame ([#502](https://github.com/bambinos/bambi/issues/502))

### Documentation

- Add circular regression example ([#465](https://github.com/bambinos/bambi/issues/465))
- Add Categorical regression example ([#457](https://github.com/bambinos/bambi/issues/457))
- Add Beta regression example ([#442](https://github.com/bambinos/bambi/issues/442))
- Add Radon Example ([#440](https://github.com/bambinos/bambi/issues/440))
- Fix typos and clear up writing in some docs ([#462](https://github.com/bambinos/bambi/issues/462))
- Documented the module `bambi/defaults` ([#472](https://github.com/bambinos/bambi/issues/472))
- Improved documentation and made it more consistent ([#472](https://github.com/bambinos/bambi/issues/472))
- Cleaned Strack RRR example ([#479](https://github.com/bambinos/bambi/issues/479))

### Deprecation

- Removed old default priors ([#474](https://github.com/bambinos/bambi/issues/474))
- Removed `draws` parameter from `Model.predict()` method ([#504](https://github.com/bambinos/bambi/issues/504))

[Changes][0.8.0]


<a id="0.7.1"></a>
# [Bambi 0.7.1](https://github.com/bambinos/bambi/releases/tag/0.7.1) - 2022-01-15

This is a patch release where we fix a bug related to the shape of 2 level categorical group-specific effects ([#441](https://github.com/bambinos/bambi/issues/441))

[Changes][0.7.1]


<a id="0.7.0"></a>
# [Bambi 0.7.0](https://github.com/bambinos/bambi/releases/tag/0.7.0) - 2022-01-11

This release includes a mix of new features, fixes, and new examples on our webpage.

### New features

- Add "categorical" built-in family ([#426](https://github.com/bambinos/bambi/issues/426))
- Add `include_mean` argument to the method `Model.fit()` ([#434](https://github.com/bambinos/bambi/issues/434))
- Add `.set_alias()` method to `Model` ([#435](https://github.com/bambinos/bambi/issues/435))

### Maintenance and fixes

- Codebase for the PyMC backend has been refactored ([#408](https://github.com/bambinos/bambi/issues/408))
- Fix examples that averaged posterior values across chains ([#429](https://github.com/bambinos/bambi/issues/429))
- Fix issue [#427](https://github.com/bambinos/bambi/issues/427) with automatic priors for the intercept term ([#430](https://github.com/bambinos/bambi/issues/430))

### Documentation

- Add StudentT regression example, thanks to [@tjburch](https://github.com/tjburch) ([#414](https://github.com/bambinos/bambi/issues/414))
- Add B-Spline regression example with cherry blossoms dataset ([#416](https://github.com/bambinos/bambi/issues/416))
- Add hirarchical linear regression example with sleepstudy dataset ([#424](https://github.com/bambinos/bambi/issues/424))

[Changes][0.7.0]


<a id="0.6.3"></a>
# [Bambi 0.6.3](https://github.com/bambinos/bambi/releases/tag/0.6.3) - 2021-09-17

Use formulae 0.2.0

[Changes][0.6.3]


<a id="0.6.2"></a>
# [Bambi 0.6.2](https://github.com/bambinos/bambi/releases/tag/0.6.2) - 2021-09-17

Minor fixes to code and docs


[Changes][0.6.2]


<a id="0.6.1"></a>
# [Bambi 0.6.1](https://github.com/bambinos/bambi/releases/tag/0.6.1) - 2021-08-24

Mainly changes to the docs and minor fixes.

[Changes][0.6.1]


<a id="0.6.0"></a>
# [Bambi 0.6.0](https://github.com/bambinos/bambi/releases/tag/0.6.0) - 2021-08-09

Many changes are included in this release. Some of the most important changes are

* New model families (StudentT, Binomial, Beta).
* In-sample and out-of-sample predictions.
* Improved sampling performance due to predictor centering when the model contains an intercept.
* New default priors (similar to rstanarm default priors).
* It's possible to use potentials.
* There's a new function to load datasets used throughout examples

[Changes][0.6.0]


<a id="0.5.0"></a>
# [Bambi 0.5.0](https://github.com/bambinos/bambi/releases/tag/0.5.0) - 2021-05-16

The main changes in this release can be summarized as follows

* Modified the API. Now all information relative to the model is passed in `Model` instantiation instead of in `Model.fit()`.
* Fixed Gamma, Wald, and Negative Binomial families.
* Changed theme of the webpage and now the documentation is built automatically.

[Changes][0.5.0]


<a id="0.4.1"></a>
# [Release 0.4.1](https://github.com/bambinos/bambi/releases/tag/0.4.1) - 2021-04-06

The aim of this release is to update to formulae 0.0.9, which contains several bug fixes. There are also other minor fixes and improvements that can be found in the changelog.

[Changes][0.4.1]


<a id="0.4.0"></a>
# [The formulae bambino (0.4.0)](https://github.com/bambinos/bambi/releases/tag/0.4.0) - 2021-03-08

The main change in this release is the use of formulae, instead of patsy, to parse model formulas.

[Changes][0.4.0]


<a id="0.3.0"></a>
# [Release 0.3.0](https://github.com/bambinos/bambi/releases/tag/0.3.0) - 2020-12-17



[Changes][0.3.0]


<a id="0.2.0"></a>
# [The First Python 3 (and arviz) Bambino (0.2.0)](https://github.com/bambinos/bambi/releases/tag/0.2.0) - 2020-03-19

This release drops Python 2 support (Python >=3.6 is required) and relies on ArviZ for all the plotting and diagnostics/stats. Support for PyStan has been deprecated. If you like to contribute to maintaining PyStan support please contact us. We have done a lot of internal changes to clean the code and make it easier to maintain.

[Changes][0.2.0]


<a id="0.1.5"></a>
# [The last legacy Python Bambino (0.1.5)](https://github.com/bambinos/bambi/releases/tag/0.1.5) - 2019-05-13



[Changes][0.1.5]


<a id="0.1.0"></a>
# [great bambino (0.1.0)](https://github.com/bambinos/bambi/releases/tag/0.1.0) - 2017-04-01

This release features numerous new features and improvements, including support for Stan, a revamped API, expanded random effect support, considerably better compilation and sampling performance for large models, better parameterization of random effects, among other changes.

[Changes][0.1.0]


<a id="0.0.5"></a>
# [0.0.5](https://github.com/bambinos/bambi/releases/tag/0.0.5) - 2017-01-19

Release 0.0.5


[Changes][0.0.5]


[0.17.0]: https://github.com/bambinos/bambi/compare/0.16.0...0.17.0
[0.16.0]: https://github.com/bambinos/bambi/compare/0.15.0...0.16.0
[0.15.0]: https://github.com/bambinos/bambi/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/bambinos/bambi/compare/0.13.0...0.14.0
[0.13.0]: https://github.com/bambinos/bambi/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/bambinos/bambi/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/bambinos/bambi/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/bambinos/bambi/compare/0.9.3...0.10.0
[0.9.3]: https://github.com/bambinos/bambi/compare/0.9.2...0.9.3
[0.9.2]: https://github.com/bambinos/bambi/compare/0.9.1...0.9.2
[0.9.1]: https://github.com/bambinos/bambi/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/bambinos/bambi/compare/0.8.0...0.9.0
[0.8.0]: https://github.com/bambinos/bambi/compare/0.7.1...0.8.0
[0.7.1]: https://github.com/bambinos/bambi/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/bambinos/bambi/compare/0.6.3...0.7.0
[0.6.3]: https://github.com/bambinos/bambi/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/bambinos/bambi/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/bambinos/bambi/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/bambinos/bambi/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/bambinos/bambi/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/bambinos/bambi/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/bambinos/bambi/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/bambinos/bambi/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/bambinos/bambi/compare/0.1.5...0.2.0
[0.1.5]: https://github.com/bambinos/bambi/compare/0.1.0...0.1.5
[0.1.0]: https://github.com/bambinos/bambi/compare/0.0.5...0.1.0
[0.0.5]: https://github.com/bambinos/bambi/tree/0.0.5

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
