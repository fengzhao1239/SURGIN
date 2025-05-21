from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            # ! the core part of the conditioning
            # ! we need to define the < operator.forward > and the < orresponding measurement >
            # ! need to be fully differentiable

            difference = measurement - self.operator.forward(x_0_hat, **kwargs)    # y - A * \hat{x_0}
            norm = torch.linalg.norm(difference)                                   # || y - A * \hat{x_0} ||2
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]        # the gardient \gard_{x_prev} || y - A * \hat{x_0} ||
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t,**kwargs):
        return x_t, None
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm

@register_conditioning_method(name='ps_linear_decay')
class PosteriorSamplingLinearDecay(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.total_diff_steps = kwargs.get('total_diff_steps', 1000)
        self.curr_diff_step = kwargs.get('curr_diff_step', 1000)
        self.start_scale = kwargs.get('start_scale', 20)
        self.end_scale = kwargs.get('end_scale', 1)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        scale = (self.start_scale - self.end_scale)*(self.curr_diff_step/self.total_diff_steps) +  self.end_scale
        x_t -= norm_grad * scale
        if self.curr_diff_step == 0:
            self.curr_diff_step = self.total_diff_steps
            return x_t, norm
        else:
            self.curr_diff_step -= 1
            return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

@register_conditioning_method(name='ps_adam')
class PosteriorSampling_Adam(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)    #1e-2 ~ 1e-3
        self.b1 = kwargs.get('b1', 0.9)
        self.b2 = kwargs.get('b2', 0.999)
        self.eps = kwargs.get('eps', 1e-8)
        self.total_diff_steps = kwargs.get('total_diff_steps', 1000)
        self.curr_diff_step = kwargs.get('curr_diff_step', 1000)
        self.mt_1 = 0.
        self.vt_1 = 0.

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        self.mt_1 = self.b1*self.mt_1 + (1 - self.b1)*norm_grad
        self.vt_1 = self.b2*self.vt_1 + (1 - self.b2)*(norm_grad)**2

        m_hat = self.mt_1/(1 - (self.b1)**(self.total_diff_steps - self.curr_diff_step + 1))
        v_hat = self.vt_1/(1 - (self.b2)**(self.total_diff_steps - self.curr_diff_step + 1))
        
        x_t -= m_hat/(v_hat.sqrt() + self.eps) * self.scale
        
        
        if self.curr_diff_step == 0:
            self.curr_diff_step = self.total_diff_steps
            self.mt_1 = 0.
            self.vt_1 = 0.
            return x_t, norm
        else:
            self.curr_diff_step -= 1
            return x_t, norm
