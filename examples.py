# piecewise_regression_example
import pandas as pd
import statsmodels.formula.api as smf

# Example data
data = pd.DataFrame({
    'x': range(30),
    'response': [i + (i >= 10) * 5 + (i >= 20) * 10 for i in range(30)]
})

# Piecewise function
def truncate(x, l):
    x = (x - l) * (x >= l)
    return x

# Piecewise regression formula
formula = "response ~ x + truncate(x, 10) + truncate(x, 20)"

# Fit the model
model = smf.ols(formula=formula, data=data).fit()
print(model.summary())
