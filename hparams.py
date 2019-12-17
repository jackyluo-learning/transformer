import six


class params(object):
    def __init__(self, **kwargs):
        self.params_dict = {}
        for name, value in six.iteritems(kwargs):
            self.add_param(name, value)
            self.params_dict[name] = value

    def add_param(self, name, value):
        if getattr(self, name, None) is not None:
            raise ValueError('Parameter name has taken: %s' % name)
        else:
            self.params_dict[name] = value
            setattr(self, name, value)

    def set_param(self, name, value):
        if hasattr(name):
            setattr(self, name, value)
            self.params_dict[name] = value
        else:
            raise NameError('Do not exist this parameter: %s' % name)

    def del_param(self, name):
        if hasattr(self, name):
            delattr(self, name)
            del self.params_dict[name]


def basic_param():
    return params(
        max_length=40,
        batch_size=128,
        buffer_size=15000,
        epoches=30,
        num_layers=4,
        d_model=128,
        dff=512,
        num_heads=8,
        dropout_rate=0.1,
        train_perc=20
    )


if __name__ == '__main__':
    print("test:\n")
    print("Hyper-Parameters:")
    params = basic_param()
    print(params.max_length)
    params.add_param('input_vocab_size', 8137)
    print(params.input_vocab_size)
