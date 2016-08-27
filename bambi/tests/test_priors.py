import pytest
from bambi.priors import Prior, Family, PriorFactory
from os.path import dirname, join
import json


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
