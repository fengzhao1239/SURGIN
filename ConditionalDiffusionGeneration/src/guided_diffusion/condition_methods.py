from abc import ABC, abstractmethod
import torch
import numpy as np

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
            # print( self.operator.forward(x_0_hat, **kwargs).shape, measurement.shape)
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





@register_conditioning_method(name='ps_adam_decay')
class PosteriorSampling_Adam(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        # Adam 参数
        self.b1  = kwargs.get('b1', 0.9)
        self.b2  = kwargs.get('b2', 0.999)
        self.eps = kwargs.get('eps', 1e-8)
        self.total_diff_steps = kwargs.get('total_diff_steps', 1000)
        self.curr_diff_step   = kwargs.get('curr_diff_step', 1000)
        self.mt_1 = 0.
        self.vt_1 = 0.

        # 引导强度调度
        self.scale_min = kwargs.get('scale_min', 1e-4)
        self.scale_max = kwargs.get('scale_max', 1e-2)
        self.scale_schedule = kwargs.get('scale_schedule', 'cosine')  # linear|cosine|exp|sigmoid|sigma
        self.exp_k   = kwargs.get('exp_k', 4.0)
        self.sig_k   = kwargs.get('sig_k', 10.0)
        self.sig_mid = kwargs.get('sig_mid', 0.6)

        # ---- 记录用 ----
        self.scale_log = []   # list of dict，记录每步scale

    def _progress(self):
        return 1.0 - (self.curr_diff_step / max(1, self.total_diff_steps))

    def _sigma_progress(self):
        if hasattr(self.noiser, "sigma"):
            sigma_t = float(self.noiser.sigma(self.curr_diff_step))
            sigma_max = float(self.noiser.sigma(self.total_diff_steps))
            sigma_min = float(self.noiser.sigma(0))
            if sigma_max == sigma_min:
                return self._progress()
            return (sigma_max - sigma_t) / (sigma_max - sigma_min)
        return self._progress()

    def guidance_scale(self):
        u = self._sigma_progress() if self.scale_schedule == 'sigma' else self._progress()
        if self.scale_schedule == 'linear':
            w = u
        elif self.scale_schedule == 'cosine':
            w = 0.5 * (1 - np.cos(np.pi * u))
        elif self.scale_schedule == 'exp':
            w = (np.exp(self.exp_k * u) - 1) / (np.exp(self.exp_k) - 1)
        elif self.scale_schedule == 'sigmoid':
            w = 1.0 / (1.0 + np.exp(-self.sig_k * (u - self.sig_mid)))
        else:
            w = 0.5 * (1 - np.cos(np.pi * u))
        scale = self.scale_min + (self.scale_max - self.scale_min) * float(np.clip(w, 0.0, 1.0))

        # ---- 记录当前 step 的 scale ----
        self.scale_log.append({
            "step": self.curr_diff_step,
            "progress": float(u),
            "scale": float(scale)
        })
        return scale

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        self.mt_1 = self.b1 * self.mt_1 + (1 - self.b1) * norm_grad
        self.vt_1 = self.b2 * self.vt_1 + (1 - self.b2) * (norm_grad ** 2)

        k = self.total_diff_steps - self.curr_diff_step + 1
        m_hat = self.mt_1 / (1 - (self.b1 ** k))
        v_hat = self.vt_1 / (1 - (self.b2 ** k))

        step_scale = self.guidance_scale()
        x_t = x_t - step_scale * (m_hat / (v_hat.sqrt() + self.eps))

        if self.curr_diff_step == 0:
            self.curr_diff_step = self.total_diff_steps
            self.mt_1 = 0.
            self.vt_1 = 0.
        else:
            self.curr_diff_step -= 1

        return x_t, norm

    def get_scale_log(self):
        """返回记录的scale变化情况, 方便后续分析/绘图"""
        return self.scale_log
