from models.CNN import CNN
from models.DAE_CNN import DAE_CNN


def build_model(model_name, **kwargs):
    if model_name == 'CNN':
        return CNN(**kwargs)
    elif model_name == 'DAE_CNN':
        return DAE_CNN(**kwargs)
    else:
        raise Exception('Model unknown %s' % model_name)