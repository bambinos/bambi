def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple
    way to accept flexible arguments. '''
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]
