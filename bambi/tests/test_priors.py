import pytest
from bambi.models import Model
from bambi.priors import Prior, Family, PriorFactory
from os.path import dirname, join
import json
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'diabetes.txt'), sep='\t')
    data['age_grp'] = 0
    data['age_grp'][data['AGE'] > 40] = 1
    data['age_grp'][data['AGE'] > 60] = 2
    return data


def test_prior_class():
    prior = Prior('CheeseWhiz', holes=0, taste=-10)
    assert prior.name == 'CheeseWhiz'
    assert isinstance(prior.args, dict)
    assert prior.args['taste'] == -10
    prior.update(taste=-100, return_to_store=1)
    assert prior.args['return_to_store'] == 1


def test_family_class():
    prior = Prior('CheeseWhiz', holes=0, taste=-10)
    family = Family('cheese', prior, link='ferment', parent='holes')
    for name in ['name', 'prior', 'link', 'parent']:
        assert hasattr(family, name)


def test_prior_factory_init_from_default_config():
    pf = PriorFactory()
    for d in ['dists', 'terms', 'families']:
        assert hasattr(pf, d)
        assert isinstance(getattr(pf, d), dict)
    assert 'normal' in pf.dists
    assert 'fixed' in pf.terms
    assert 'gaussian' in pf.families


def test_prior_factory_init_from_config():
    config_file = join(dirname(__file__), 'data', 'sample_priors.json')
    pf = PriorFactory(config_file)
    for d in ['dists', 'terms', 'families']:
        assert hasattr(pf, d)
        assert isinstance(getattr(pf, d), dict)
    config_dict = json.load(open(config_file, 'r'))
    pf = PriorFactory(config_dict)
    for d in ['dists', 'terms', 'families']:
        assert hasattr(pf, d)
        assert isinstance(getattr(pf, d), dict)
    assert 'feta' in pf.dists
    assert 'hard' in pf.families
    assert 'yellow' in pf.terms
    pf = PriorFactory(dists=config_dict['dists'])
    assert 'feta' in pf.dists
    pf = PriorFactory(terms=config_dict['terms'])
    assert 'yellow' in pf.terms
    pf = PriorFactory(families=config_dict['families'])
    assert 'hard' in pf.families


def test_prior_retrieval():
    config_file = join(dirname(__file__), 'data', 'sample_priors.json')
    pf = PriorFactory(config_file)
    prior = pf.get(dist='asiago')
    assert prior.name == 'Asiago'
    assert isinstance(prior, Prior)
    assert prior.args['hardness'] == 10
    with pytest.raises(KeyError):
        assert prior.args['holes'] == 4
    family = pf.get(family='hard')
    assert isinstance(family, Family)
    assert family.link == 'grate'
    backup = family.prior.args['backup']
    assert isinstance(backup, Prior)
    assert backup.args['flavor'] == 10000
    prior = pf.get(term='yellow')
    assert prior.name == 'Swiss'

    # Test exception raising
    with pytest.raises(ValueError):
        pf.get(dist='apple')
    with pytest.raises(ValueError):
        pf.get(term='banana')
    with pytest.raises(ValueError):
        pf.get(family='cantaloupe')


def test_update_term_priors_after_init(diabetes_data):
    model = Model(diabetes_data)
    model.add('Y ~ BMI')
    model.add('S1')
    model.add(random='age_grp|BP')

    p1 = Prior('Normal', mu=-10, sd=10)
    p2 = Prior('Beta', alpha=2, beta=2)

    model.set_priors({'BMI': 0.3, 'S1': p2})
    model.build(backend='pymc')
    assert model.terms['S1'].prior.args['beta'] == 2
    assert model.terms['BMI'].prior.scale == 0.3
    assert np.isclose(model.terms['BMI'].prior.args['sd'], 4.7, rtol=.1)[0]

    model.set_priors({('S1', 'BMI'): p1})
    model.build(backend='pymc')
    assert model.terms['S1'].prior.args['sd'] == 10
    assert model.terms['BMI'].prior.args['mu'] == -10

    p3 = Prior('Normal', mu=0, sd=Prior('Normal', mu=0, sd=7))
    model.set_priors(fixed=0.3, random=p3)
    model.build(backend='pymc')
    assert model.terms['BMI'].prior.scale == 0.3
    assert np.isclose(model.terms['BMI'].prior.args['sd'], 4.7, rtol=.1)[0]
    assert model.terms['age_grp|BP'].prior.args['sd'].args['sd'] == 7

    # Invalid names should raise error
    with pytest.raises(ValueError):
        model.set_priors({'nonexistent_term': 0.3})
        model.build(backend='pymc')

    # Test for partial names, e.g., 'threecats' should match 'threecats[0]'.
    model = Model(diabetes_data)
    model.add('Y ~ 1', random='age_grp|BP', categorical='age_grp')
    model.set_priors({'age_grp|BP': 0.5})
    model.build(backend='pymc')
    assert model.terms['age_grp[T.1]|BP'].prior.scale == 0.5
    assert np.isclose(model.terms['age_grp[T.1]|BP'].prior.args['sd'].args['sd'],
                      111, rtol=.1)[0]
    assert model.terms['1|BP'].prior.scale == 0.5


def test_auto_scale(diabetes_data):

    # By default, should scale everything except custom Prior() objects
    model = Model(diabetes_data)
    priors = {
        'S1': 0.3,
        'BP': Prior('Cauchy', alpha=1, beta=17.5)
    }
    model.fit('BMI ~ S1 + S2 + BP', run=False, priors=priors)
    model.build(backend='pymc3')
    p1 = model.terms['S1'].prior
    p2 = model.terms['S2'].prior
    p3 = model.terms['BP'].prior
    assert p1.name == p2.name == 'Normal'
    assert 0 < p1.args['sd'] < 1
    assert p2.args['sd'] > p1.args['sd']
    assert p3.name == 'Cauchy'
    assert p3.args['beta'] == 17.5

    # With auto_scale off, everything should be flat unless explicitly named
    # in priors
    model = Model(diabetes_data, auto_scale=False)
    model.fit('BMI ~ S1 + S2 + BP', run=False, priors=priors)
    model.build(backend='pymc3')
    p1_off = model.terms['S1'].prior
    p2_off = model.terms['S2'].prior
    p3_off = model.terms['BP'].prior
    assert p1_off.name == 'Normal'
    assert p2_off.name == 'Flat'
    assert 0 < p1_off.args['sd'] < 1
    assert 'sd' not in p2_off.args
    assert p3_off.name == 'Cauchy'
    assert p3_off.args['beta'] == 17.5
