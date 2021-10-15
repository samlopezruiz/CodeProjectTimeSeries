from numpy import mean
from numpy import median


class SimpleForecast:
    def __init__(self, train, config):
        self.config = config
        self.history = list(train)

    def forecast(self):
        n, offset, avg_type = self.config
        # persist value, ignore other architectures
        if avg_type == 'persist':
            return self.history[-n]
        # collect values to average
        values = list()
        if offset == 1:
            values = self.history[-n:]
        else:
            # skip bad configs
            if n * offset > len(self.history):
                raise Exception('Config beyond end of data: %d %d' % (n, offset))
            # try and collect n values using offset
            for i in range(1, n + 1):
                ix = i * offset
                values.append(self.history[-ix])
        # check if we can average
        if len(values) < 2:
            raise Exception('Cannot calculate average')
        # mean of last n values
        if avg_type == 'mean':
            return mean(values)
        # median of last n values
        return median(values)

    def append(self, new_history):
        if isinstance(new_history, list):
            [self.history.append(d) for d in new_history]
        else:
            self.history.append(new_history)
        return self


def simple_fit(train, config):
    return SimpleForecast(train, config)


# create a set of simple configs to try
def simple_configs(max_avg_len, offsets=[1]):
    configs = list()
    for i in range(1, max_avg_len + 1):
        for o in offsets:
            for t in ['persist', 'mean', 'median']:
                cfg = [i, o, t]
                configs.append(cfg)
    return configs


def simple_forecast(model, steps=1, history=None, cfg=None):
    return model.forecast()
