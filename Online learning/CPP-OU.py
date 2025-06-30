''' sequential probabilistic learning algorithm: Empirical Bayes Adaptive Filtering
state_drift: how much to let the belief's uncertainty grow over time
deflate_covariance: same, but for the belief's covariance
deflate_mean: how much to shrink the belief's mean over time



scan

'''


class EmpiricalBayesAdaptive(ABC):
    def __init__(
        self, n_inner, ebayes_lr, state_drift, deflate_mean, deflate_covariance
    ):
        self.n_inner = n_inner
        self.ebayes_lr = ebayes_lr # empirical bayes learning rate
        self.state_drift = state_drift
        self.deflate_mean = deflate_mean * 1.0
        self.deflate_covariance = deflate_covariance * 1.0

    @abstractmethod
    def init_bel(self):
        """
        Initialize belief state
        """
        ...

    @abstractmethod
    def log_predictive_density(self, y, X, bel):
        ...


    @abstractmethod
    def update_bel(self, y, X, bel):
        ...

    def predict_bel(self, eta, bel):
        gamma = jnp.exp(-eta / 2)
        dim = bel.mean.shape[0]

        deflation_mean = gamma ** self.deflate_mean
        deflation_covariance = (gamma ** 2) ** self.deflate_covariance

        mean = deflation_mean * bel.mean
        cov = deflation_covariance * bel.cov + (1 - gamma ** 2) * jnp.eye(dim) * self.state_drift
        bel = bel.replace(mean=mean, cov=cov)
        return bel


    def log_reg_predictive_density(self, eta, y, X, bel):
        bel = self.predict_bel(eta, bel)
        log_p_pred = self.log_predictive_density(y, X, bel)
        return log_p_pred


    def step(self, y, X, bel):
        grad_log_predict_density = jax.grad(self.log_reg_predictive_density, argnums=0)

        def _inner_pred(i, eta, bel):
            grad = grad_log_predict_density(eta, y, X, bel)
            eta = eta + self.ebayes_lr * grad
            eta = eta * (eta > 0) # hard threshold
            return eta

        _inner = partial(_inner_pred, bel=bel)
        eta = jax.lax.fori_loop(0, self.n_inner, _inner, bel.eta)
        bel = bel.replace(eta=eta)
        bel = self.predict_bel(bel.eta, bel)
        bel = self.update_bel(y, X, bel)
        return bel


    def scan(self, y, X, bel, callback_fn=None):
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        def _step(bel, yX):
            y, X = yX
            bel_posterior = self.step(y, X, bel)
            out = callback_fn(bel_posterior, bel, y, X)

            return bel_posterior, out

        bel, hist = jax.lax.scan(_step, bel, (y, X))
        return bel, hist
    



''' 
Init -- initializes, lr is probably the learning rate of the ebayes.
init_bel (bel of belief) -- it returns a jax class with the initial values as astributes
'''


class ExpfamEBA(EmpiricalBayesAdaptive):
    def __init__(
        self, n_inner, ebayes_lr, state_drift,  deflate_mean, deflate_covariance, filter
    ):
        super().__init__(n_inner, ebayes_lr, state_drift, deflate_mean, deflate_covariance)
        self.filter = filter

    def init_bel(self, mean, cov, eta=0.0):
        """
        Initialize belief state
        """
        state_filter = self.filter.init_bel(mean, cov)
        mean = state_filter.mean
        cov = state_filter.cov

        bel = states.GammaFilterState(
            mean=mean,
            cov=cov,
            eta=eta
        )
        return bel

    def log_predictive_density(self, y, X, bel):
        return self.filter.log_predictive_density(y, X, bel)

    def update_bel(self, y, X, bel):
        bel_pred = self.filter._predict(bel)
        bel = self.filter._update(bel_pred, y, X)
        return bel




''' Probabilistic Filter

filter init -- initializes the filter.
filter scan -- makes a sequential scan with y, x, and the initial state. Collects the get_predicted_mean and updates the beliefs of the initial state. 
Hist are the predicted means (estimated internal state)

'''


def run_cppd(drift, lr, y, x):
    filter = ExpfamEBA(
        n_inner=1, ebayes_lr=lr, state_drift=drift,
        deflate_mean=True, deflate_covariance=True, filter=base_filter
    )
    mean_init = jnp.zeros(2)
    bel_init = filter.init_bel(mean_init, cov=1.0)
    bel_final, hist = filter.scan(y, x, bel_init, get_predicted_mean)
    return hist


'''  run_cppd is the prediction model.
yhat seems to be the prediction output after doing the dot product. My interpretation is that those are the weights of x.
sigmoid makes the prediction and then it is compared to the real results. And the accuracy is calculated.
'''
@partial(jax.jit)
def eval_cppd(drift, lr, y, x):
    mean = run_cppd(drift, lr, y, x)

    yhat = jnp.einsum("sm,sm->s", mean.squeeze(), x)
    return (jax.nn.sigmoid(yhat).round() == y.ravel()).mean()


''' This is a bayesian optimization to find the parameters drift and lr. 
partial(...) is from functools library: it fixes x and y and looks for drift and lr.
pbounds stablishes the bounds of the variables values to find.'''
bo = BayesianOptimization(
    partial(eval_cppd, y=y_samples[:, None], x=x_samples),
    pbounds = {"drift": (0.0, 2.0), "lr": (0.0, 1.0)},
    random_state=314,
    verbose=1
)
bo.maximize()