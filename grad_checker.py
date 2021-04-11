import numpy as np

def calc_grad(func, h):
    shape = h.shape

    df_dh = np.zeros(shape)

    eps = 1e-7

    for x in range(shape[0]):
        for y in range(shape[1]):
            h_plus_eps = h.copy()
            h_minus_eps = h.copy()

            h_plus_eps[x, y] += np.array(eps)
            h_minus_eps[x, y] += np.array(-eps)
            h_plus_loss = func(h_plus_eps)
            h_minus_loss = func(h_minus_eps)

            df_dh[x, y] = np.array((h_plus_loss - h_minus_loss) / (2*eps))
    
    return df_dh


def calc_param_grads(model_runner, model):
    results = {}

    def run_model(name, idx, new_param, prev_param):
            model.params[name] = model.params[name][:idx] + (new_param, ) + model.params[name][idx+1:]
            res = model_runner(model)
            model.params[name] = model.params[name][:idx] + (prev_param, ) + model.params[name][idx+1:]
            return res
    for param_name, params in model.params.items():
        results[param_name] = [ calc_grad(lambda x: run_model(param_name, i, x, param), param) for i, param in enumerate(params)]
            
    return results

def collect_param_results(grads, results):
    result = 0
    total = 0

    for param_name, params in grads:
        for i in range(len(params)):
            delta = params[i] - results[param_name][i]
            delta[abs(delta) < 1e-7] = 0
            result += np.linalg.norm(delta)
            total += 1

    return result / total

def run_checks(checker, times, eps, name):
    results = np.array([np.linalg.norm(checker()) for _ in range(times)])
    result = (results < eps).all()
    if not result:
        print(results)
    else:
        print(f"{name} checks are passed")
