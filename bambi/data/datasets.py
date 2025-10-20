"""Base IO code for datasets. Heavily influenced by Arviz's (and scikit-learn's) implementation."""

import hashlib
import itertools
import os
import shutil

from collections import namedtuple
from urllib.request import urlretrieve

import pandas as pd


FileMetadata = namedtuple("FileMetadata", ["filename", "url", "checksum", "description"])
DATASETS = {
    "my_data": FileMetadata(
        filename="my_data.csv",
        url="https://ndownloader.figshare.com/files/28850355",
        checksum="1bfcdd10d0848c1811e33e467c92734fb488406ef3f9b9aae16a57b258a30fac",
        description="""
Toy dataset with one response variable "y" and two covariates "x" and "z".
""",
    ),
    "adults": FileMetadata(
        filename="adults.csv",
        url="https://ndownloader.figshare.com/files/28870743",
        checksum="27a5270ba720125dfb24a7708cbee0218b2ead36248ae244813655d03320e43e",
        description="""
A sample with census data from 1994 in United States.
""",
    ),
    "ANES": FileMetadata(
        filename="ANES_2016_pilot.csv",
        url="https://ndownloader.figshare.com/files/28870740",
        checksum="3106beb6ded5a592ea0405d23b868bd8e74c259d7a7f5242c907555692905772",
        description="""
The ANES is a nationally representative, cross-sectional survey used extensively in political
science. This is a dataset from the 2016 pilot study, consisting of responses from 1200 voting-age
 U.S. citizens.
""",
    ),
    "ESCS": FileMetadata(
        filename="ESCS.csv",
        url="https://ndownloader.figshare.com/files/28870722",
        checksum="0195545797a4258de138a205a013a84022bbe23e7ff47782b179055c706300b6",
        description="""
A longitudinal study of hundreds of adults who completed dozens of different self-report and
behavioral measures over the course of 15 years. Among the behavioral measures is an index of
illegal drug use.
""",
    ),
    "carclaims": FileMetadata(
        filename="carclaims.csv",
        url="https://ndownloader.figshare.com/files/28870713",
        checksum="74924bf5f0a6e5aa5453d87845cea05e6b41bb2052cf6f096d7f102235ae5cdf",
        description="""
67856 insurance policies and 4624 (6.8%) claims in Australia between 2004 and 2005
""",
    ),
    "batting": FileMetadata(
        filename="Batting.csv",
        url="https://ndownloader.figshare.com/files/29749140",
        checksum="bbbc9459632c738a07bbe0877970a7bbd1f4c2448193979337fe5bc3a4ab0228",
        description="""
Baseball Databank is a compilation of historical baseball data in a convenient, tidy format,
distributed under Open Data terms by the Baseball Data Bank.
""",
    ),
    "cherry_blossoms": FileMetadata(
        filename="cherry_blossoms.csv",
        url="https://figshare.com/ndownloader/files/31072807",
        checksum="b859dd4f64c231c76ecb80b78f26da71e2f92698c50e0ceb93be0399dee24f51",
        description="""
Historical Series of Phenological data for Cherry Tree Flowering at Kyoto City. Extracted from
the `rethinking` library in R.
""",
    ),
    "sleepstudy": FileMetadata(
        filename="sleepstudy.csv",
        url="https://figshare.com/ndownloader/files/31181002",
        checksum="0a002bec8be2fa9d40dbbf3d5038e614d113a4fd5bf8813f6f4271c3d6294675",
        description="""
The average reaction time per day (in milliseconds) for subjects in a sleep deprivation study.
Days 0-1 were adaptation and training (T1/T2), day 2 was baseline (B); sleep deprivation started
after day 2.

Reaction
    Average reaction time (ms)

Days
    Number of days of sleep deprivation

Subject
    Subject number on which the observation was made
""",
    ),
    "periwinkles": FileMetadata(
        filename="periwinkles.csv",
        url="https://ndownloader.figshare.com/files/34446077",
        checksum="50da9791b7a66fbcc9ea4dd828dc7a3a66d5e067faf10f3bfd143af6c590923a",
        description="""Data for 31 periwinkles transplanted downshore as a function of the distance
        travelled by them after release.""",
    ),
    "admissions": FileMetadata(
        filename="admissions.csv",
        url="https://figshare.com/ndownloader/files/34757857",
        checksum="41e2312ca09d50e99c2db67fbabc78d215df6ce71eefe880df5e9310a9fa8397",
        description="""Admission into graduate school data. This dataset has a binary response
        variable called 'admit'. There are three predictor variables: 'gre', 'gpa' and 'rank'.""",
    ),
    "bikes": FileMetadata(
        filename="bike_sharing.csv",
        url="https://figshare.com/ndownloader/files/38737026",
        checksum="3e1844b6da435f910b10899e18188568f7d789c715a286c9c6c2ca23833ee7ac",
        description="""
This dataset contains the hourly and daily count of rental bikes between
years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal
information.

- instant: record index
- dteday : date
- season : season (1:winter, 2:spring, 3:summer, 4:fall)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not (extracted from [Web Link])
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit :
    * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temperature : Normalized temperature in Celsius.
    The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius.
    The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
- humidty: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- count: count of total rental bikes including both casual and registered

Original source: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
""",
    ),
    "mtcars": FileMetadata(
        filename="mtcars.csv",
        url="https://figshare.com/ndownloader/files/40208785",
        checksum="c802190c43e02246da9c6c9c3f13a58f076cc6b77922f4d9766a3c6bdb1b52bd",
        description="""
The data was extracted from the 1974 Motor Trend US magazine, and comprises fuel consumption and 10
aspects of automobile design and performance for 32 automobiles (1973--74 models).
The following is a description of the variables:

* mpg: Miles/(US) gallon
* cyl: Number of cylinders
* disp: Displacement (cu.in.)
* hp: Gross horsepower
* drat: Rear axle ratio
* wt: Weight (1000 lbs)
* qsec: 1/4 mile time
* vs: Engine (0 = V-shaped, 1 = straight)
* am: Transmission (0 = automatic, 1 = manual)
* gear: Number of forward gears
""",
    ),
    "kidney": FileMetadata(
        filename="kidney.csv",
        url="https://figshare.com/ndownloader/files/41645361",
        checksum="46e49372b4e8c3044dca0ffbb4eb2244f56d7398746802e351baac6c12625564",
        description="""
It describes the first and second recurrence times of infection in kidney patients together with
information on risk variables such as age, sex, and disease type.
This dataset is taken from McGilchrist and Aisbett (1991).

* time: Days to first or second recurrence of the infection, or the time of censoring
* censored: Indicates censoring status. 0 indicates no censoring and 1 indicates right censoring
* patient: Patient ID
* recur: Indicates if the infection occurs for first or second time.
* age: Age of the patient
* sex: Sex of the patient
* disease: The type of disease. Can be "AN", "GN", "PKG", or "other"

McGilchrist, C. A., & Aisbett, C. W. (1991). Regression with frailty in survival analysis.
Biometrics, 47(2), 461-466
""",
    ),
}


def get_data_home(data_home: str | None = None):
    """Return the path of the Bambi data dir.

    This folder is used to avoid downloading the data several times.

    By default the data dir is set to a folder named 'bambi_data' in the user home folder.
    Alternatively, it can be set by the `"BAMBI_DATA"` environment variable or programmatically by
    giving an explicit folder path. The `"~"` symbol is expanded to the user home folder. If the
    folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home :  str or None, optional
        The path to Bambi data dir.
    """
    if data_home is None:
        data_home = os.environ.get("BAMBI_DATA", os.path.join("~", "bambi_data"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home: str | None = None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home: str or None, optional
        The path to Bambi data dir.
        By default a folder named `"bambi_data"` in the user home folder.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as buff:
        while True:
            buffer = buff.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)

    return sha256hash.hexdigest()


def load_data(dataset: str | None = None, data_home: str | None = None):
    """Load a dataset.

    Run with no parameters to get a list of all available data sets.

    The directory to save can also be set with the environment variable `BAMBI_HOME`.
    The checksum of the dataset is checked against a hardcoded value to watch for data corruption.
    Run `bmb.clear_data_home()` to clear the data directory.

    Parameters
    ----------
    dataset : str or None, optional
        Name of dataset to load.
    data_home : str or None, optional
        Where to save remote datasets.

    Returns
    -------
    pd.DataFrame or str
    """
    home_dir = get_data_home(data_home=data_home)

    if dataset in DATASETS:
        datafile = DATASETS[dataset]
        file_path = os.path.join(home_dir, datafile.filename)

        if not os.path.exists(file_path):
            urlretrieve(datafile.url, file_path)
            checksum = _sha256(file_path)
            if datafile.checksum != checksum:
                raise IOError(
                    f"{file_path} has an SHA256 checksum ({checksum}) differing from expected "
                    f"({datafile.checksum}), file may be corrupted. Run `bambi.clear_data_home()` "
                    "and try again, or please open an issue."
                )

        return pd.read_csv(file_path)

    if dataset is None:
        return _list_datasets(home_dir)

    raise ValueError(
        f"Dataset {dataset} not found! " f"The following are available:\n{_list_datasets(home_dir)}"
    )


def _list_datasets(home_dir):
    """Get a string representation of all available datasets with descriptions.

    Parameters
    ----------
    home_dir : str
        Name of the home directoy.

    Returns
    -------
    str
    """
    lines = []
    for filename, resource in itertools.chain(DATASETS.items()):
        file_path = os.path.join(home_dir, filename)
        if not os.path.exists(file_path):
            location = f"location: {resource.url}"
        else:
            location = f"location: {file_path}"
        lines.append(f"{filename}\n{'=' * len(filename)}\n{resource.description}\n{location}")

    return f"\n\n{10 * '-'}\n\n".join(lines)
