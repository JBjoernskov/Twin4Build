def istype(instance, type_tuple):
    instance_type = type(instance)
    return any([instance_type==type_ for type_ in type_tuple])

