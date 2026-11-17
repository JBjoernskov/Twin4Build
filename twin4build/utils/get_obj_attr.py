def get_obj_attr(obj, inverse=False):
    if inverse:
        attributes = {v: k for k, v in obj.__dict__.items() if k[:2] != "__"}
    else:
        attributes = {k: v for k, v in obj.__dict__.items() if k[:2] != "__"}
    return attributes
