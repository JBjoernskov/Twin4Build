def get_object_attributes(obj):
    attributes = dir(obj)
    attributes = [attr for attr in attributes if attr[:2]!="__"]#Remove callables
    return attributes