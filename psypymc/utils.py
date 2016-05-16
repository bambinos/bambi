def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple
    way to accept flexible arguments. '''
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def transformer(func):
    ''' Decorator that indicates that a method acts as a data transformer. '''
    def wrapper(self, *args, **kwargs):
       self.transform(func.__name__, *args, **kwargs)
    return wrapper