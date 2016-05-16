def scale(data):
    d = data
    d = (d - d.mean())/d.std()
    return d