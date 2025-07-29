def get_object_properties(object_):
    return {key: value for (key, value) in vars(object_).items()}
