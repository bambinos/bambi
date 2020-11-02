# Guidelines for Contributing

As a scientific community-driven software project, Bambi welcomes contributions from interested individuals or groups. These guidelines are provided to give potential contributors information to make their contribution compliant with the conventions of the Bambi project, and maximize the probability of such contributions to be merged as quickly and efficiently as possible.

There are 4 main ways of contributing to the Bambi project (in descending order of difficulty or scope):

- Adding new or improved functionality to the existing codebase
- Fixing outstanding issues (bugs) with the existing codebase. They range from low-level software bugs to higher-level design problems
- Contributing or improving the documentation (`docs`) or examples (`bambi/examples`)
- Submitting issues related to bugs or desired enhancements

## Opening issues

We appreciate being notified of problems with the existing Bambi code. We prefer that issues be filed the on [Github Issue Tracker](https://github.com/bambinos/bambi/issues), rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for key words in the project issue tracker.

## Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are inclined to do so to submit patches for new or existing issues via pull
requests. This is particularly the case for simple fixes, such as typos or tweaks to documentation, which do not require a heavy investment
of time and attention.

Contributors are also encouraged to contribute new code to enhance Bambi's functionality, also via pull requests.

The preferred workflow for contributing to Bambi is to fork
the [GitHub repository](https://github.com/bambinos/bambi/), clone it to your local machine, and develop on a feature branch.

For more instructions see the
[Pull request checklist](#pull-request-checklist)

### Code Formatting

For code generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/versions/master/how_tos/style_guide.html)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
Both more or less follows PEP 8.

### Docstring formatting

Docstrings should follow the
[numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html)
Please reasonably document any additions or changes to the codebase, when in doubt, add a docstring.

## Steps

1. Fork the [project repository](https://github.com/bambinos/bambi/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the Bambi repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your GitHub handle>/bambi.git
   cd bambi
   git remote add upstream git@github.com:bambinos/bambi.git
   ```

3. Create a `feature` branch to hold your development changes:

   ```bash
   git checkout -b my-feature
   ```

   Always use a `feature` branch. It's good practice to never routinely work on the `master` branch of any repository.

4. Project requirements are in `requirements.txt`, and libraries used for development are in `requirements-dev.txt`. To set up a development environment, you may run:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. Develop the feature on your feature branch. Add changed files using `git add` and then `git commit` files:

   ```bash
   git add modified_files
   git commit
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:

   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

   Then push the changes to your GitHub account with:

   ```bash
   git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the Bambi repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

## Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

- If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

- All public methods must have informative docstrings with sample usage when appropriate.

- To indicate a work in progress please mark the PR as `draft`. Drafts may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

- All other tests pass when everything is rebuilt from scratch.

- When adding additional functionality, provide at least one example script or Jupyter Notebook in the `bambi/examples/` folder. Have a look at other examples for reference. Examples should demonstrate why the new functionality is useful in practice and, if possible, compare it to other methods available in Bambi.

- Added tests follow the [pytest fixture pattern](https://docs.pytest.org/en/latest/fixture.html#fixture)

- Documentation and high-coverage tests are necessary for enhancements to be accepted.

- Documentation follows Numpy style guide

- Run any of the pre-existing examples in `examples` that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit test, but also to show how your contribution improves Bambi for end users.

- Code coverage **cannot** decrease. Coverage can be checked with **pytest-cov** package:

  ```bash
  pip install pytest pytest-cov coverage
  pytest --cov=bambi --cov-report=html bambi/tests/
  ```

- Your code passes pylint

  ```bash
  pip install pylint
  pylint bambi/
  ```

**This guide was derived from the [arviz guide to contributing](https://github.com/arviz-devs/arviz/blob/master/CONTRIBUTING.md)**
