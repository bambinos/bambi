# Guidelines for Contributing

As a scientific community-driven software project, Bambi welcomes contributions from interested individuals or groups.
These guidelines are provided to give potential contributors information to make their contribution compliant with the conventions of the Bambi project,
and maximize the probability of such contributions to be merged as quickly and efficiently as possible.

There are 4 main ways of contributing to the Bambi project (in descending order of difficulty or scope):

- Adding new or improved functionality to the existing codebase
- Fixing outstanding issues (bugs) with the existing codebase.
They range from low-level software bugs to higher-level design problems
- Contributing or improving the documentation (`docs`) or examples (`bambi/examples`)
- Submitting issues related to bugs or desired enhancements

## Opening issues

We appreciate being notified of problems with the existing Bambi code.
We prefer that issues be filed the on [Github Issue Tracker](https://github.com/bambinos/bambi/issues),
rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for key words in the project issue tracker.

## Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are inclined to do so to submit patches for new or existing issues via pull requests.
This is particularly the case for simple fixes, such as typos or tweaks to documentation, which do not require a heavy investment of time and attention.

Contributors are also encouraged to contribute new code to enhance Bambi's functionality, also via pull requests.

The preferred workflow for contributing to Bambi is to fork
the [GitHub repository](https://github.com/bambinos/bambi/), clone it to your local machine,
and develop on a feature branch.

For more instructions see the [Pull request checklist](#pull-request-checklist)

### Code Formatting

For code generally follow the
[TensorFlow's style guide](https://www.tensorflow.org/versions/master/how_tos/style_guide.html)
or the [Google style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
Both more or less follows PEP 8.

### Docstring formatting

Docstrings should follow the
[NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html)
Please reasonably document any additions or changes to the codebase, when in doubt, add a docstring.

## Steps

1. Fork the [project repository](https://github.com/bambinos/bambi/) by clicking on the
'Fork' button near the top right of the main repository page.
This creates a copy of the code under your GitHub user account.

1. Clone your fork of the Bambi repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your GitHub handle>/bambi.git
   cd bambi
   git remote add upstream git@github.com:bambinos/bambi.git
   ```

1. Create a feature branch (e.g. `my-feature`) to hold your development changes:

   ```bash
   git checkout -b <your branch name>
   ```

   **Always use a feature branch**.
   It's good practice to never routinely work on the `main` branch of any repository.

1. Set up a [Pixi](https://pixi.sh/latest/) development environment (**only the first time**).

   1. If you don't have Pixi installed, please follow the [official installation instructions](https://pixi.sh/latest/installation/).

   1. This project defines multiple environments. For development, we need `dev`, which contains
   all the dependencies groups that are useful for development. To install it: 

      ```bash
      pixi install -e dev
      ```

   1. Set up `pre-commit`:

      ```bash
      pixi run -e dev pre-commit-setup
      ```

1. Activate the `dev` environment:

   ```bash
   pixi shell -e dev
   ```

   And use `exit` to deactivate it, when needed.

1. Develop the feature on your feature branch. Add changed files using `git add` and then `git commit` files:

   ```bash
   git add <modified files>
   git commit -m "Message summarizing commit changes"
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Then, push the changes to your branch on your fork on GitHub with:

   ```bash
   git push -u origin <your branch name>
   ```

6. Go to the GitHub web page of your fork of the Bambi repo.
   Click the **'Pull Request'** button to send your changes to the project's maintainers for review.
   This will send an email to the committers.

## Building the documentation locally

The documentation is built using [Quarto](https://quarto.org/) with [quartodoc](https://github.com/machow/quartodoc) for the API reference. The `dev` environment already includes all the necessary dependencies.

### Prerequisites

- The `dev` Pixi environment must be installed (see [Steps](#steps) above).
- [Quarto](https://quarto.org/docs/get-started/) must be installed separately. On macOS you can install it via Homebrew:

  ```bash
  brew install quarto
  ```

  For other platforms, follow the [official Quarto installation instructions](https://quarto.org/docs/get-started/).

### Building the docs

1. Activate the `dev` environment:

   ```bash
   pixi shell -e dev
   ```

2. Generate the API reference pages with quartodoc:

   ```bash
   quartodoc build --config docs/_quarto.yml
   ```

3. Preview the docs locally (with live reload):

   ```bash
   quarto preview docs/
   ```

   Or render the full site (output goes to `docs/_site/`):

   ```bash
   quarto render docs/
   ```

## Pull request checklist

We recommend that your contribution complies with the following guidelines before you submit a pull request:

- If your pull request addresses an issue, please use the pull request title to describe the issue
and mention the issue number in the pull request description.
This will make sure a link back to the original issue is created.

- Please use a Draft Pull Request to indicate an incomplete contribution or work in progress.
Draft PRs may be useful to (1) signal you are working on something and avoid duplicated work,
(2) request early feedback on functionality or API, or (3) seek collaborators.

- Run any of the pre-existing examples notebooks that contain analyses that would be affected by
your changes to ensure that nothing breaks. This is a useful opportunity to not only check your
work for bugs that might not be revealed by unit test, but also to show how your contribution 
improves ArviZ for end users.

- All public functions and methods must have informative NumPy-style docstrings,
including Examples and, when appropriate, See Also, References, and Notes.

- New functionality must be covered by tests, and tests context must follow the
[pytest fixture pattern](https://docs.pytest.org/en/latest/fixture.html#fixture).

- When adding additional functionality, provide at least one example script or Jupyter Notebook in the `bambi/examples/` folder.
Have a look at other examples for reference. Examples should demonstrate why the new functionality is useful in practice and,
if possible, compare it to other methods available in Bambi.

- Your code follows the style guidelines of the project:

  ```shell
  pixi run -e dev pre-commit run --all
  ```

- Your code passes pylint

  ```shell
  pixi run -e dev pylint bambi
  ```

* All **tests must pass**.

  ```shell
  pixi run -e dev pytest tests
  ```

**This guide was derived from the [ArviZ guide to contributing](https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md)**
