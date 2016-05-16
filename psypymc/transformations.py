def scale(data):
    ''' Standardize the data (i.e., subtract mean and divide by SD). '''
    d = data
    d = (d - d.mean())/d.std()
    return d