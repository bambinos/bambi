import pytest
from bambi.models import Term, Model
from bambi.priors import Prior
import pandas as pd
import numpy as np
import matplotlib
import re
matplotlib.use('Agg')


@pytest.fixture(scope="module")
def crossed_data():
    '''
    Random effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    Fixed effects:
    A continuous predictor, a numeric dummy, and a three-level category
    (levels a,b,c)

    Structure:
    Subjects nested in dummy (e.g., gender), crossed with threecats
    Items crossed with dummy, nested in threecats
    Sites partially crossed with dummy (4/5 see a single dummy, 1/5 sees both
    dummies)
    Sites crossed with threecats
    '''
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'crossed_random.csv'))
    return data


def test_empty_model(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.add_y('Y')
    model0.build()
    model0.fit(samples=1)

    # using add_term
    model1 = Model(crossed_data)
    model1.fit('Y ~ 0', run=False)
    model1.build()
    model1.fit(samples=1)

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_nan_handling(crossed_data):
    data = crossed_data.copy()

    # Should fail because predictor has NaN
    model_fail_na = Model(crossed_data)
    model_fail_na.fit('Y ~ continuous', run=False)
    model_fail_na.terms['continuous'].data[[4, 6, 8], :] = np.nan
    with pytest.raises(ValueError):
        model_fail_na.build()

    # Should drop 3 rows with warning
    model_drop_na = Model(crossed_data, dropna=True)
    model_drop_na.fit('Y ~ continuous', run=False)
    model_drop_na.terms['continuous'].data[[4, 6, 8], :] = np.nan
    with pytest.warns(UserWarning) as w:
        model_drop_na.build()
    assert '3 rows' in w[0].message.args[0]


def test_intercept_only_model(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 1', run=False)
    model0.build()
    model0.fit(samples=1)

    # using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_intercept()
    model1.build()
    model1.fit(samples=1)

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_slope_only_model(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + continuous', run=False)
    model0.build()
    model0.fit(samples=1)

    # using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_term('continuous')
    model1.build()
    model1.fit(samples=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_simple_regression(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ continuous', run=False)
    model0.build()
    model0.fit(samples=1)

    # using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_intercept()
    model1.add_term('continuous')
    model1.build()
    model1.fit(samples=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_many_fixed_effects(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ continuous + dummy + threecats', run=False)
    model0.build()
    model0.fit(samples=1)

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_intercept()
    model1.add_term('continuous')
    model1.add_term('dummy')
    model1.add_term('threecats')
    model1.build()
    model1.fit(samples=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_cell_means_parameterization(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats', run=False)
    model0.build()
    model0.fit(samples=1)

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_term('threecats', drop_first=False)
    model1.build()
    model1.fit(samples=1)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_3x4_fixed_anova(crossed_data):
    # add a four-level category that's perfectly crossed with threecats
    crossed_data['fourcats'] = sum(
        [[x]*10 for x in ['a', 'b', 'c', 'd']], list())*3

    # using formula, with intercept
    model0 = Model(crossed_data)
    model0.fit('Y ~ threecats*fourcats', run=False)
    model0.build()
    fitted0 = model0.fit(samples=1)
    # make sure X has 11 columns (not including the intercept)
    assert len(fitted0.diagnostics['VIF']) == 11

    # using formula, without intercept (i.e., 2-factor cell means model)
    model1 = Model(crossed_data)
    model1.fit('Y ~ 0 + threecats*fourcats', run=False)
    model1.build()
    fitted1 = model1.fit(samples=1)
    # make sure X has 12 columns
    assert len(fitted1.diagnostics['VIF']) == 12


def test_cell_means_with_covariate(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats + continuous', run=False)
    model0.build()
    model0.fit(samples=1)

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_term('threecats', drop_first=False)
    model1.add_term('continuous')
    model1.build()
    model1.fit(samples=1)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that threecats priors have finite variance
    assert not any(np.isinf(model0.terms['threecats'].prior.args['sd']))

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_cell_means_with_random_intercepts(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats', random=['subj'], run=False)
    model0.build()
    fitted = model0.fit(samples=100)

    # using add_term
    model1 = Model(crossed_data, intercept=False)
    model1.add_y('Y')
    model1.add_term('threecats', categorical=True, drop_first=False)
    model1.add_term('subj', categorical=True, random=True, drop_first=False)
    model1.build()
    model1.fit(samples=1)

    # check that they have the same random terms
    assert set(model0.random_terms) == set(model1.random_terms)

    # check that fixed effect design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)

    # check that add_formula and add_term models have same priors for random
    # effects
    priors0 = {x.name: x.prior.args[
        'sd'].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args[
        'sd'].args for x in model1.terms.values() if x.random}
    assert set(priors0) == set(priors1)

    # test summary
    # it looks like some versions of pymc3 add a trailing '_' to transformed
    # vars and some dont. so here for consistency we strip out any trailing '_'
    # that we find
    full = fitted.summary(exclude_ranefs=False, hide_transformed=False).index
    full = set([re.sub(r'_$', r'', x) for x in full])
    test_set = fitted.summary(exclude_ranefs=False).index
    test_set = set([re.sub(r'_$', r'', x) for x in test_set])
    assert test_set == full.difference(set(['Y_sd_interval', 'u_subj_sd_log']))
    test_set = fitted.summary(hide_transformed=False).index
    test_set = set([re.sub(r'_$', r'', x) for x in test_set])
    assert test_set == full.difference(
        set(['subj[{}]'.format(i) for i in range(10)]))

    # test get_trace
    test_set = fitted.get_trace().columns
    test_set = set([re.sub(r'_$', r'', x) for x in test_set])
    assert test_set == full.difference(set(['Y_sd_interval', 'u_subj_sd_log'])) \
        .difference(set(['subj[{}]'.format(i) for i in range(10)]))

    # test plots
    fitted.plot(kind='priors')
    fitted.plot()


def test_random_intercepts(crossed_data):
    # using formula and '1|' syntax
    model0 = Model(crossed_data)
    model0.fit(
        'Y ~ continuous', random=['1|subj', '1|item', '1|site'], run=False)
    model0.build()
    # model0.fit(samples=1)

    # # using formula and 'subj' syntax
    # model1 = Model(crossed_data)
    # model1.fit('Y ~ continuous', random=['subj','item','site'], run=False)
    # model1.build()
    # # model1.fit(samples=1)

    # # check that they have the same random terms
    # assert set(model1.random_terms) == set(model0.random_terms)

    # using add_term
    model2 = Model(crossed_data)
    model2.add_y('Y')
    model2.add_intercept()
    model2.add_term('continuous')
    model2.add_term('subj', random=True)
    model2.add_term('item', random=True)
    model2.add_term('site', random=True)
    model2.build()
    # model2.fit(samples=1)

    # check that this has the same random terms as above
    assert set(model0.random_terms) == set(model2.random_terms)

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors2 = {
        x.name: x.prior.args for x in model2.terms.values() if not x.random}
    assert set(priors0) == set(priors2)

    # check that add_formula and add_term models have same priors for random
    # effects
    priors0 = {x.name: x.prior.args[
        'sd'].args for x in model0.terms.values() if x.random}
    priors2 = {x.name: x.prior.args[
        'sd'].args for x in model2.terms.values() if x.random}
    assert set(priors0) == set(priors2)


def test_many_random_effects(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ continuous',
               random=['0+threecats|subj', 'continuous|item', 'dummy|item',
                       'threecats|site'], run=False)
    model0.build()
    # model0.fit(samples=1)

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    # fixed effects
    model1.add_intercept()
    model1.add_term('continuous')
    # random effects
    model1.add_term('threecats', over='subj', drop_first=False, random=True,
                    categorical=True)
    model1.add_term('item', random=True, categorical=True)
    model1.add_term('continuous', over='item', random=True)
    model1.add_term('dummy', over='item', random=True)
    model1.add_term('site', random=True, categorical=True)
    model1.add_term('threecats', over='site', random=True, categorical=True)
    model1.build()
    # model1.fit(samples=1)

    # check that the random effects design matrices have the same shape
    X0 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model0.random_terms.values()], axis=1)
    X1 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model0.random_terms.values()], axis=1)
    assert X0.shape == X1.shape

    # check that the random effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that fixed effect design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)

    # check that add_formula and add_term models have same priors for random
    # effects
    priors0 = {x.name: x.prior.args[
        'sd'].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args[
        'sd'].args for x in model1.terms.values() if x.random}
    assert set(priors0) == set(priors1)


def test_cell_means_with_many_random_effects(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats',
               random=['0+threecats|subj', 'continuous|item', 'dummy|item',
                       'threecats|site'], run=False)
    model0.build()
    # model0.fit(samples=1)

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    # fixed effects
    model1.add_term('threecats', drop_first=False)
    # random effects
    model1.add_term('threecats', over='subj', drop_first=False, random=True,
                    categorical=True)
    model1.add_term('item', random=True, categorical=True)
    model1.add_term('continuous', over='item', random=True)
    model1.add_term('dummy', over='item', random=True)
    model1.add_term('site', random=True, categorical=True)
    model1.add_term('threecats', over='site', random=True, categorical=True)
    model1.build()
    # model1.fit(samples=1)

    # check that the random effects design matrices have the same shape
    X0 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model0.random_terms.values()], axis=1)
    X1 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model0.random_terms.values()], axis=1)
    assert X0.shape == X1.shape

    # check that the random effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that fixed effect design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)

    # check that add_formula and add_term models have same priors for random
    # effects
    priors0 = {x.name: x.prior.args[
        'sd'].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args[
        'sd'].args for x in model1.terms.values() if x.random}
    assert set(priors0) == set(priors1)


def test_logistic_regression(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('threecats[b] ~ continuous + dummy',
               family='binomial', link='logit', run=False)
    model0.build()
    fitted = model0.fit(samples=100)

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('threecats',
                 data=pd.DataFrame(1*(crossed_data['threecats'] == 'b')),
                 family='binomial', link='logit')
    model1.add_intercept()
    model1.add_term('continuous')
    model1.add_term('dummy')
    model1.build()
    model1.fit(samples=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that add_formula and add_term models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)

    # test that summary reminds user which event is being modeled
    fitted.summary()

    # test that traceplot reminds user which event is being modeled
    fitted.plot()
