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
    model0 = Model(crossed_data)
    model0.add('Y ~ 0')
    model0.build(backend='pymc3')
    model0.fit(samples=1)

    model1 = Model(crossed_data)
    model1.fit('Y ~ 0', run=False)
    model1.build(backend='pymc3')
    model1.fit(samples=1)

    # check that both models have same priors for fixed effects
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
        model_fail_na.build(backend='pymc3')

    # Should drop 3 rows with warning
    model_drop_na = Model(crossed_data, dropna=True)
    model_drop_na.fit('Y ~ continuous', run=False)
    model_drop_na.terms['continuous'].data[[4, 6, 8], :] = np.nan
    with pytest.warns(UserWarning) as w:
        model_drop_na.build(backend='pymc3')
    assert '3 rows' in w[0].message.args[0]


def test_intercept_only_model(crossed_data):
    # using fit
    model0 = Model(crossed_data)
    model0.fit('Y ~ 1', run=False)
    model0.build(backend='pymc3')
    model0.fit(samples=1)

    # using add
    model1 = Model(crossed_data)
    model1.add('Y ~ 0')
    model1.add('1')
    model1.build(backend='pymc3')
    model1.fit(samples=1)

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_slope_only_model(crossed_data):
    # using fit
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + continuous', run=False)
    model0.build(backend='pymc3')
    model0.fit(samples=1)

    # using add
    model1 = Model(crossed_data)
    model1.add('Y ~ 0')
    model1.add('0 + continuous')
    model1.build(backend='pymc3')
    model1.fit(samples=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_cell_means_parameterization(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats', run=False)
    model0.build(backend='pymc3')
    model0.fit(samples=1)

    # build model using add
    model1 = Model(crossed_data)
    model1.add('Y ~ 0')
    model1.add('0 + threecats')
    model1.build(backend='pymc3')
    model1.fit(samples=1)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that fit and add models have same priors for fixed
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

    # using fit, with intercept
    model0 = Model(crossed_data)
    model0.fit('Y ~ threecats*fourcats', run=False)
    model0.build(backend='pymc3')
    fitted0 = model0.fit(samples=1)
    # make sure X has 11 columns (not including the intercept)
    assert len(fitted0.diagnostics['VIF']) == 11

    # using fit, without intercept (i.e., 2-factor cell means model)
    model1 = Model(crossed_data)
    model1.fit('Y ~ 0 + threecats*fourcats', run=False)
    model1.build(backend='pymc3')
    fitted1 = model1.fit(samples=1)
    # make sure X has 12 columns
    assert len(fitted1.diagnostics['VIF']) == 12


def test_cell_means_with_covariate(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats + continuous', run=False)
    model0.build(backend='pymc3')
    # model0.fit(samples=1)

    # build model using add
    model1 = Model(crossed_data)
    model1.add('Y ~ 0')
    model1.add('0 + threecats')
    model1.add('0 + continuous')
    model1.build(backend='pymc3')
    # model1.fit(samples=1)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1

    # check that threecats priors have finite variance
    assert not any(np.isinf(model0.terms['threecats'].prior.args['sd']))

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)


def test_many_fixed_many_random(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    fitted = model0.fit('Y ~ continuous + dummy + threecats',
        random=['0+threecats|subj', '1|item', '0+continuous|item', 
            'dummy|item', 'threecats|site'],
        backend='pymc3', samples=10)
    # model0.build(backend='pymc3')
    # model0.fit(samples=1)

    # build model using add(append=True)
    model1 = Model(crossed_data)
    model1.add('Y ~ 1')
    model1.add('continuous')
    model1.add('dummy')
    model1.add('threecats')
    model1.add(random='0+threecats|subj')
    model1.add(random='1|item')
    model1.add(random='0+continuous|item')
    model1.add(random='dummy|item')
    model1.add(random='threecats|site')
    model1.build(backend='pymc3')
    # model1.fit(samples=1)

    # build model using stan backend
    model2 = Model(crossed_data)
    fitted2 = model2.fit('Y ~ continuous + dummy + threecats',
        random=['0+threecats|subj', '1|item', '0+continuous|item',
            'dummy|item', 'threecats|site'],
        backend='stan', samples=10)

    # check that the random effects design matrices have the same shape
    X0 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model0.random_terms.values()], axis=1)
    X1 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model1.random_terms.values()], axis=1)
    X2 = pd.concat([pd.DataFrame(t.data) if not isinstance(t.data, dict) else
                    pd.concat([pd.DataFrame(t.data[x])
                               for x in t.data.keys()], axis=1)
                    for t in model2.random_terms.values()], axis=1)
    assert X0.shape == X1.shape
    assert X0.shape == X2.shape
    assert X1.shape == X2.shape

    # check that the random effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    X2_set = set(tuple(X1.iloc[:, i]) for i in range(len(X2.columns)))
    assert X0_set == X1_set
    assert X0_set == X2_set
    assert X1_set == X2_set

    # check that fixed effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:, lev]) for t in model0.fixed_terms.values()
              for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:, lev]) for t in model1.fixed_terms.values()
              for lev in range(len(t.levels))])
    X2 = set([tuple(t.data[:, lev]) for t in model2.fixed_terms.values()
              for lev in range(len(t.levels))])
    assert X0 == X1
    assert X0 == X2
    assert X1 == X2

    # check that fit and add models have same priors for fixed effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    priors2 = {
        x.name: x.prior.args for x in model2.terms.values() if not x.random}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    assert set(priors0) == set(priors2)
    assert set(priors1) == set(priors2)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=.01) for x in a.keys()]
    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])
    assert all([dicts_close(priors0[x], priors2[x]) for x in priors0.keys()])
    assert all([dicts_close(priors1[x], priors2[x]) for x in priors0.keys()])

    # check that fit and add models have same priors for random effects
    priors0 = {x.name: x.prior.args[
        'sd'].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args[
        'sd'].args for x in model1.terms.values() if x.random}
    priors2 = {x.name: x.prior.args[
        'sd'].args for x in model2.terms.values() if x.random}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    assert set(priors0) == set(priors2)
    assert set(priors1) == set(priors2)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=.01) for x in a.keys()]
    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])
    assert all([dicts_close(priors0[x], priors2[x]) for x in priors0.keys()])
    assert all([dicts_close(priors1[x], priors2[x]) for x in priors0.keys()])

    # test consistency between summary and to_df for pymc3
    assert len(set(fitted.to_df().columns)) == 15
    assert set(fitted.to_df().columns)==set(fitted.summary().index)

    # test consistency between summary and to_df for stan
    assert len(set(fitted2.to_df().columns)) == 15
    assert set(fitted2.to_df().columns)==set(fitted2.summary().index)

    # test consistenct between pymc3 and stan
    assert set(fitted.to_df().columns)==set(fitted2.to_df().columns)
    assert set(fitted.summary().index)==set(fitted2.summary().index)

    # check hide_transformed for pymc3
    # it looks like some versions of pymc3 add a trailing '_' to transformed
    # vars and some dont. so here for consistency we strip out any trailing '_'
    # that we find
    full = fitted.summary(exclude_ranefs=False, hide_transformed=False).index
    full = set([re.sub(r'_$', r'', x) for x in full])
    test_set = fitted.summary(exclude_ranefs=False).index
    test_set = set([re.sub(r'_$', r'', x) for x in test_set])
    answer = {'1|item_offset[0]','1|item_offset[10]','1|item_offset[11]',
        '1|item_offset[1]','1|item_offset[2]','1|item_offset[3]',
        '1|item_offset[4]','1|item_offset[5]','1|item_offset[6]',
        '1|item_offset[7]','1|item_offset[8]','1|item_offset[9]',
        '1|item_sd_log','1|site_offset[0]','1|site_offset[1]',
        '1|site_offset[2]','1|site_offset[3]','1|site_offset[4]',
        '1|site_sd_log','Y_sd_interval','continuous|item_offset[0]',
        'continuous|item_offset[10]','continuous|item_offset[11]',
        'continuous|item_offset[1]','continuous|item_offset[2]',
        'continuous|item_offset[3]','continuous|item_offset[4]',
        'continuous|item_offset[5]','continuous|item_offset[6]',
        'continuous|item_offset[7]','continuous|item_offset[8]',
        'continuous|item_offset[9]','continuous|item_sd_log',
        'dummy|item_offset[0]','dummy|item_offset[10]','dummy|item_offset[11]',
        'dummy|item_offset[1]','dummy|item_offset[2]','dummy|item_offset[3]',
        'dummy|item_offset[4]','dummy|item_offset[5]','dummy|item_offset[6]',
        'dummy|item_offset[7]','dummy|item_offset[8]','dummy|item_offset[9]',
        'dummy|item_sd_log','threecats[T.b]|site_offset[0]',
        'threecats[T.b]|site_offset[1]','threecats[T.b]|site_offset[2]',
        'threecats[T.b]|site_offset[3]','threecats[T.b]|site_offset[4]',
        'threecats[T.b]|site_sd_log','threecats[T.c]|site_offset[0]',
        'threecats[T.c]|site_offset[1]','threecats[T.c]|site_offset[2]',
        'threecats[T.c]|site_offset[3]','threecats[T.c]|site_offset[4]',
        'threecats[T.c]|site_sd_log','threecats[a]|subj_offset[0]',
        'threecats[a]|subj_offset[1]','threecats[a]|subj_offset[2]',
        'threecats[a]|subj_offset[3]','threecats[a]|subj_offset[4]',
        'threecats[a]|subj_offset[5]','threecats[a]|subj_offset[6]',
        'threecats[a]|subj_offset[7]','threecats[a]|subj_offset[8]',
        'threecats[a]|subj_offset[9]','threecats[a]|subj_sd_log',
        'threecats[b]|subj_offset[0]','threecats[b]|subj_offset[1]',
        'threecats[b]|subj_offset[2]','threecats[b]|subj_offset[3]',
        'threecats[b]|subj_offset[4]','threecats[b]|subj_offset[5]',
        'threecats[b]|subj_offset[6]','threecats[b]|subj_offset[7]',
        'threecats[b]|subj_offset[8]','threecats[b]|subj_offset[9]',
        'threecats[b]|subj_sd_log','threecats[c]|subj_offset[0]',
        'threecats[c]|subj_offset[1]','threecats[c]|subj_offset[2]',
        'threecats[c]|subj_offset[3]','threecats[c]|subj_offset[4]',
        'threecats[c]|subj_offset[5]','threecats[c]|subj_offset[6]',
        'threecats[c]|subj_offset[7]','threecats[c]|subj_offset[8]',
        'threecats[c]|subj_offset[9]','threecats[c]|subj_sd_log'}
    assert full.difference(test_set) == answer

    # check hide_transformed for stan
    # the major transformed parameter here should just be "lp__",
    # which is technically not transformed, but equally uninteresting
    full2 = fitted2.summary(exclude_ranefs=False, hide_transformed=False)
    full2 = set(full2.index)
    test_set2 = fitted2.summary(exclude_ranefs=False)
    test_set2 = set(test_set2.index)
    answer = set(['lp__'] \
        + ['yhat[{}]'.format(i) for i in range(len(crossed_data.index))])
    assert full2.difference(test_set2) == answer

    # check for consistency in parameter names between pymc3 and stan
    assert test_set == test_set2

    # check exclude_ranefs for pymc3
    test_set = fitted.summary(hide_transformed=False).index
    test_set = set([re.sub(r'_$', r'', x) for x in test_set])
    answer = {'1|item[0]','1|item[10]','1|item[11]','1|item[1]','1|item[2]',
        '1|item[3]','1|item[4]','1|item[5]','1|item[6]','1|item[7]','1|item[8]',
        '1|item[9]','1|item_offset[0]','1|item_offset[10]','1|item_offset[11]',
        '1|item_offset[1]','1|item_offset[2]','1|item_offset[3]',
        '1|item_offset[4]','1|item_offset[5]','1|item_offset[6]',
        '1|item_offset[7]','1|item_offset[8]','1|item_offset[9]','1|site[0]',
        '1|site[1]','1|site[2]','1|site[3]','1|site[4]','1|site_offset[0]',
        '1|site_offset[1]','1|site_offset[2]','1|site_offset[3]',
        '1|site_offset[4]','continuous|item[0]','continuous|item[10]',
        'continuous|item[11]','continuous|item[1]','continuous|item[2]',
        'continuous|item[3]','continuous|item[4]','continuous|item[5]',
        'continuous|item[6]','continuous|item[7]','continuous|item[8]',
        'continuous|item[9]','continuous|item_offset[0]',
        'continuous|item_offset[10]','continuous|item_offset[11]',
        'continuous|item_offset[1]','continuous|item_offset[2]',
        'continuous|item_offset[3]','continuous|item_offset[4]',
        'continuous|item_offset[5]','continuous|item_offset[6]',
        'continuous|item_offset[7]','continuous|item_offset[8]',
        'continuous|item_offset[9]','dummy|item[0]','dummy|item[10]',
        'dummy|item[11]','dummy|item[1]','dummy|item[2]','dummy|item[3]',
        'dummy|item[4]','dummy|item[5]','dummy|item[6]','dummy|item[7]',
        'dummy|item[8]','dummy|item[9]','dummy|item_offset[0]',
        'dummy|item_offset[10]','dummy|item_offset[11]','dummy|item_offset[1]',
        'dummy|item_offset[2]','dummy|item_offset[3]','dummy|item_offset[4]',
        'dummy|item_offset[5]','dummy|item_offset[6]','dummy|item_offset[7]',
        'dummy|item_offset[8]','dummy|item_offset[9]','threecats[T.b]|site[0]',
        'threecats[T.b]|site[1]','threecats[T.b]|site[2]',
        'threecats[T.b]|site[3]','threecats[T.b]|site[4]',
        'threecats[T.b]|site_offset[0]','threecats[T.b]|site_offset[1]',
        'threecats[T.b]|site_offset[2]','threecats[T.b]|site_offset[3]',
        'threecats[T.b]|site_offset[4]','threecats[T.c]|site[0]',
        'threecats[T.c]|site[1]','threecats[T.c]|site[2]',
        'threecats[T.c]|site[3]','threecats[T.c]|site[4]',
        'threecats[T.c]|site_offset[0]','threecats[T.c]|site_offset[1]',
        'threecats[T.c]|site_offset[2]','threecats[T.c]|site_offset[3]',
        'threecats[T.c]|site_offset[4]','threecats[a]|subj[0]',
        'threecats[a]|subj[1]','threecats[a]|subj[2]','threecats[a]|subj[3]',
        'threecats[a]|subj[4]','threecats[a]|subj[5]','threecats[a]|subj[6]',
        'threecats[a]|subj[7]','threecats[a]|subj[8]','threecats[a]|subj[9]',
        'threecats[a]|subj_offset[0]','threecats[a]|subj_offset[1]',
        'threecats[a]|subj_offset[2]','threecats[a]|subj_offset[3]',
        'threecats[a]|subj_offset[4]','threecats[a]|subj_offset[5]',
        'threecats[a]|subj_offset[6]','threecats[a]|subj_offset[7]',
        'threecats[a]|subj_offset[8]','threecats[a]|subj_offset[9]',
        'threecats[b]|subj[0]','threecats[b]|subj[1]','threecats[b]|subj[2]',
        'threecats[b]|subj[3]','threecats[b]|subj[4]','threecats[b]|subj[5]',
        'threecats[b]|subj[6]','threecats[b]|subj[7]','threecats[b]|subj[8]',
        'threecats[b]|subj[9]','threecats[b]|subj_offset[0]',
        'threecats[b]|subj_offset[1]','threecats[b]|subj_offset[2]',
        'threecats[b]|subj_offset[3]','threecats[b]|subj_offset[4]',
        'threecats[b]|subj_offset[5]','threecats[b]|subj_offset[6]',
        'threecats[b]|subj_offset[7]','threecats[b]|subj_offset[8]',
        'threecats[b]|subj_offset[9]','threecats[c]|subj[0]',
        'threecats[c]|subj[1]','threecats[c]|subj[2]','threecats[c]|subj[3]',
        'threecats[c]|subj[4]','threecats[c]|subj[5]','threecats[c]|subj[6]',
        'threecats[c]|subj[7]','threecats[c]|subj[8]','threecats[c]|subj[9]',
        'threecats[c]|subj_offset[0]','threecats[c]|subj_offset[1]',
        'threecats[c]|subj_offset[2]','threecats[c]|subj_offset[3]',
        'threecats[c]|subj_offset[4]','threecats[c]|subj_offset[5]',
        'threecats[c]|subj_offset[6]','threecats[c]|subj_offset[7]',
        'threecats[c]|subj_offset[8]','threecats[c]|subj_offset[9]'}
    assert full.difference(test_set) == answer

    # check exclude_ranefs for stan
    test_set2 = set(fitted2.summary(hide_transformed=False).index)
    answer = {'1|item[0]','1|item[10]','1|item[11]','1|item[1]','1|item[2]',
        '1|item[3]','1|item[4]','1|item[5]','1|item[6]','1|item[7]','1|item[8]',
        '1|item[9]','1|site[0]','1|site[1]','1|site[2]','1|site[3]','1|site[4]',
        'continuous|item[0]','continuous|item[10]','continuous|item[11]',
        'continuous|item[1]','continuous|item[2]','continuous|item[3]',
        'continuous|item[4]','continuous|item[5]','continuous|item[6]',
        'continuous|item[7]','continuous|item[8]','continuous|item[9]',
        'dummy|item[0]','dummy|item[10]','dummy|item[11]','dummy|item[1]',
        'dummy|item[2]','dummy|item[3]','dummy|item[4]','dummy|item[5]',
        'dummy|item[6]','dummy|item[7]','dummy|item[8]','dummy|item[9]',
        'threecats[T.b]|site[0]','threecats[T.b]|site[1]',
        'threecats[T.b]|site[2]','threecats[T.b]|site[3]',
        'threecats[T.b]|site[4]','threecats[T.c]|site[0]',
        'threecats[T.c]|site[1]','threecats[T.c]|site[2]',
        'threecats[T.c]|site[3]','threecats[T.c]|site[4]',
        'threecats[a]|subj[0]','threecats[a]|subj[1]','threecats[a]|subj[2]',
        'threecats[a]|subj[3]','threecats[a]|subj[4]','threecats[a]|subj[5]',
        'threecats[a]|subj[6]','threecats[a]|subj[7]','threecats[a]|subj[8]',
        'threecats[a]|subj[9]','threecats[b]|subj[0]','threecats[b]|subj[1]',
        'threecats[b]|subj[2]','threecats[b]|subj[3]','threecats[b]|subj[4]',
        'threecats[b]|subj[5]','threecats[b]|subj[6]','threecats[b]|subj[7]',
        'threecats[b]|subj[8]','threecats[b]|subj[9]','threecats[c]|subj[0]',
        'threecats[c]|subj[1]','threecats[c]|subj[2]','threecats[c]|subj[3]',
        'threecats[c]|subj[4]','threecats[c]|subj[5]','threecats[c]|subj[6]',
        'threecats[c]|subj[7]','threecats[c]|subj[8]','threecats[c]|subj[9]'}
    assert full2.difference(test_set2) == answer

    # test plots for pymc3
    fitted.plot(kind='priors')
    fitted.plot()

    # test plots for stan
    fitted2.plot(kind='priors')
    fitted2.plot()


def test_cell_means_with_many_random_effects(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats',
               random=['0+threecats|subj', 'continuous|item', 'dummy|item',
                       'threecats|site'], run=False)
    model0.build(backend='pymc3')
    # model0.fit(samples=1)

    # build model using add(append=True)
    model1 = Model(crossed_data)
    model1.add('Y ~ 0')
    model1.add('0 + threecats')
    model1.add(random='0+threecats|subj')
    model1.add(random='1|item')
    model1.add(random='0+continuous|item')
    model1.add(random='dummy|item')
    model1.add(random='threecats|site')
    model1.build(backend='pymc3')
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

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)

    # check that fit and add models have same priors for random
    # effects
    priors0 = {x.name: x.prior.args[
        'sd'].args for x in model0.terms.values() if x.random}
    priors1 = {x.name: x.prior.args[
        'sd'].args for x in model1.terms.values() if x.random}
    assert set(priors0) == set(priors1)


def test_logistic_regression(crossed_data):
    # build model using fit
    model0 = Model(crossed_data)
    model0.fit('threecats[b] ~ continuous + dummy',
               family='binomial', link='logit', run=False)
    model0.build(backend='pymc3')
    fitted = model0.fit(samples=100)

    # build model using add
    model1 = Model(crossed_data)
    model1.add('threecats[b] ~ 1', family='binomial', link='logit')
    model1.add('continuous')
    model1.add('dummy')
    model1.build(backend='pymc3')
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

    # check that fit and add models have same priors for fixed
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

def test_poisson_regression(crossed_data):
    # build model using fit
    crossed_data['count'] = (crossed_data['Y'] - crossed_data['Y'].min()).round()
    model0 = Model(crossed_data)
    model0.fit('count ~ threecats + continuous + dummy',
        family='poisson', run=False)
    model0.build(backend='pymc3')
    model0.fit(samples=1)

    # build model using add
    model1 = Model(crossed_data)
    model1.add('count ~ 1', family='poisson')
    model1.add('threecats')
    model1.add('continuous')
    model1.add('dummy')
    model1.build(backend='pymc3')
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

    # check that fit and add models have same priors for fixed
    # effects
    priors0 = {
        x.name: x.prior.args for x in model0.terms.values() if not x.random}
    priors1 = {
        x.name: x.prior.args for x in model1.terms.values() if not x.random}
    assert set(priors0) == set(priors1)
