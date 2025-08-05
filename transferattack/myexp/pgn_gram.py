import torch
from ..utils import *
from ..attack import Attack
from ..input_transformation.admix import Admix
from ..input_transformation.dim import DIM
from ..input_transformation.sia import SIA
from ..input_transformation.bsr import BSR
from ..gradient.mifgsm import MIFGSM
import copy


class PGNGRAM(MIFGSM):

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='PGNGRAM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch)
        self.alpha = alpha if targeted else epsilon / epoch
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch if not targeted else 200
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.targeted=targeted
        self.cc=cc
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print(f"Selected layer:{layer} is not in Model. Select in {model_layers_name}")
                exit()
            else:
                m_layers=dict(m.named_modules()).get(layer)
        return m_layers
    
    def random_orthogonal_noise(self,grads):
        result=torch.zeros_like(grads)
        radom_noise=torch.empty_like(grads).uniform_(-self.zeta,self.zeta)
        project_const_ = (radom_noise * grads).sum([1, 2, 3]) / ((grads * grads).sum([1, 2, 3]))
        for idx in range(grads.shape[0]):
            if project_const_[idx].item()<0:
                parallel_g = project_const_[idx].item() * grads[idx]
                orthogonal_g = radom_noise[idx] - parallel_g
                result[idx]=(orthogonal_g/torch.max(torch.abs(orthogonal_g)))*self.zeta
            else:
                result[idx]=radom_noise[idx]
        return result
    
    def __forward_hook(self,m,i,o):
        global mid_output
        # print(mid_outputs)
        mid_output=o
        # self.mid_output.append(o)

    def get_grad(self, loss, delta, retain_graph=False,**kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]

    def __backward_hook(self,m,i,o):
        global mid_grad
        mid_grad = o

    def sgam_noise(self, maplist, cc):
        assert len(maplist) == 2
        b, a = maplist[0], maplist[1]
        mean_abs_a = torch.mean(torch.abs(a), dim=[1, 2, 3], keepdim=True)
        mean_abs_b = torch.mean(torch.abs(b), dim=[1, 2, 3], keepdim=True)
        a, b = a / mean_abs_a, b / mean_abs_b
        # return a*cc+b

        project_const_ = (a * b).sum(dim=[1, 2, 3]) / (b * b).sum(dim=[1, 2, 3])
        noise = torch.zeros_like(b)

        negative_proj = project_const_ <= 0
        positive_proj = ~negative_proj

        parallel_g = project_const_.unsqueeze(1).unsqueeze(2).unsqueeze(3) * b
        orthogonal_g = a - parallel_g

        noise[negative_proj] = b[negative_proj]* cc + orthogonal_g[negative_proj]
        # noise[negative_proj] = b[negative_proj]  + a[negative_proj] * cc
        noise[positive_proj] = b[positive_proj]* cc + a[positive_proj]

        return noise

    def batch_channel_gaussian_kernel(self, x, gamma=None, sigma=None):
        """
        Compute the Gaussian (RBF) kernel matrix between channels for each sample in the batch.
        Args:
            x (Tensor): Input tensor of shape (batch_size, C, W, H)
            gamma (float, optional): Gaussian kernel parameter gamma. If None, will be computed from sigma.
            sigma (float, optional): Gaussian kernel bandwidth. If None, will be estimated from data.
        Returns:
            Tensor: Gaussian kernel matrix of shape (batch_size, C, C)
        """
        B, C, W, H = x.shape
        x_flat = x.view(B, C, -1)  # (B, C, D)
        l2_distance = torch.cdist(x_flat, x_flat, p=2)  # (B, C, C)

        # auto gamma
        if gamma is None:
            if sigma is None:
                with torch.no_grad():
                    median_distance = torch.median(l2_distance.view(B, -1), dim=1).values.clamp(min=1e-8)  # (B,)
                    gamma = 1.0 / median_distance  # (B,)
                    gamma = gamma.view(B, 1, 1)  
            else:
                gamma = 1.0 / sigma 

        kernel_matrix = torch.exp(-gamma * l2_distance)
        return kernel_matrix
    
    def batch_channel_laplacian_kernel(self, x, gamma=None, sigma=None):
        """
        Compute the Laplacian kernel matrix between channels for each sample in the batch, optimized for memory and speed.

        Args:
            x (Tensor): Input tensor of shape (batch_size, C, W, H)
            gamma (float, optional): Laplacian kernel parameter γ. If None, will be estimated from data.
            sigma (float, optional): Laplacian kernel parameter σ. If None, will be estimated from data.

        Returns:
            Tensor: Laplacian kernel matrix of shape (batch_size, C, C)
        """
        B, C, W, H = x.shape
        x_flat = x.view(B, C, -1)  # (B, C, D)


        l1_distance = torch.cdist(x_flat, x_flat, p=1)  # (B, C, C)
        # smooth_l1_distance=F.smooth_l1_loss(x_flat, x_flat, reduction='none', beta=0.1).sum(dim=-1)


        if gamma is None:
            if sigma is None:

                with torch.no_grad():
                    median_distance = torch.median(l1_distance.view(B, -1), dim=1).values.clamp(min=1e-8)  # (B,)
                    gamma = 1.0 / median_distance  # (B,)
                    gamma = gamma.view(B, 1, 1) 
            else:
                gamma = 1.0 / sigma  

        kernel_matrix = torch.exp(-gamma * l1_distance)

        return kernel_matrix
    
    def get_gram(self, feature,method='gram'):
        assert len(feature.shape) == 4
        if method=='gram':
            b, c, w, h = feature.shape
            f = feature.view(b, c, w * h)
            f_t = torch.transpose(f, 1, 2)
            result = torch.bmm(f, f_t)
        elif method=='gram_up':
            b, c, w, h = feature.shape
            f = feature.view(b, c, w * h)
            f_t = torch.transpose(f, 1, 2)
            result = torch.bmm(f, f_t)
            result=result**2
        elif method=='gaussian':
            result = self.batch_channel_gaussian_kernel(feature)
        elif method=='laplacian':
            result = self.batch_channel_laplacian_kernel(feature)
        elif method=='anova':
            result = self.batch_channel_anova_kernel(feature)
        else:
            print('method error')
            exit()
        return result

    def gram_loss(self,f_ori,f_adv):
        # f_ori_repeat=f_ori.repeat(self.num_scale*self.num_admix,1,1,1)
        f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat,method=self.k_m)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        loss=torch.nn.MSELoss()(ori_gram,adv_gram)
        # loss=0
        # for i in range(adv_gram.shape[0]):
        #     loss+=torch.nn.MSELoss()(ori_gram[i], adv_gram[i])
        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = 0
        averaged_style_gradient = 0
        for n in range(self.num_neighbor):



            # Random sample an example
            x_near = self.transform(
                data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # Calculate the output of the x_near
            logits = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss, delta,True)


            # Calculate the loss of the x_near
            loss = self.gram_loss(kwargs['f_ori'],mid_output)

            # Calculate the gradient of the x_near
            g_style = self.get_grad(loss, delta,True)

            # Compute the predicted point x_next
            # x_next = self.transform(x_near + self.alpha * (-admix_g1 / (torch.abs(admix_g1).mean(dim=(1, 2, 3), keepdim=True))))
            x_next = x_near + self.alpha * (-g_1 / (torch.abs(g_1).mean(dim=(1, 2, 3), keepdim=True)))

            # Calculate the output of the x_next
            logits = self.get_logits(x_next)

            # Calculate the loss of the x_next
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_next
            g_2 = self.get_grad(loss, delta)

            # Calculate the gradients
            averaged_gradient += (1 - self.gamma) * g_1 + self.gamma * g_2
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor,averaged_style_gradient/self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
 

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient = 0, 0
        with torch.no_grad():
            _ = self.model(data)
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map,delta_start=delta)
            if self.targeted:
                averaged_gradient=-averaged_gradient
            grad=self.sgam_noise([averaged_gradient, averaged_style_gradient], self.cc)

            momentum = self.get_momentum(grad, momentum)

            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    
    

class SSEPPGN_DIM(PGNGRAM,DIM):

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPPGN_DIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch,feature_layer=feature_layer, device=device)
        self.alpha = epsilon / epoch
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.cc=cc
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print("Selected layer is not in Model")
                exit()
            else:
                m_layers=dict(m.named_modules()).get(layer)
        return m_layers
    
    def random_orthogonal_noise(self,grads):
        result=torch.zeros_like(grads)
        radom_noise=torch.empty_like(grads).uniform_(-self.zeta,self.zeta)
        project_const_ = (radom_noise * grads).sum([1, 2, 3]) / ((grads * grads).sum([1, 2, 3]))
        for idx in range(grads.shape[0]):
            if project_const_[idx].item()<0:
                parallel_g = project_const_[idx].item() * grads[idx]
                orthogonal_g = radom_noise[idx] - parallel_g
                result[idx]=(orthogonal_g/torch.max(torch.abs(orthogonal_g)))*self.zeta
            else:
                result[idx]=radom_noise[idx]
        return result
    
    def __forward_hook(self,m,i,o):
        global mid_output
        # print(mid_outputs)
        mid_output=o
        # self.mid_output.append(o)

    def get_grad(self, loss, delta, retain_graph=False,**kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]


    def gram_loss(self,f_ori,f_adv):
        # f_ori_repeat=f_ori.repeat(self.num_scale*self.num_admix,1,1,1)
        f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat,method=self.k_m)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        loss=0
        for i in range(adv_gram.shape[0]):
            loss+=torch.nn.MSELoss()(ori_gram[i], adv_gram[i])
        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = 0
        averaged_style_gradient = 0
        for n in range(self.num_neighbor):


            # Random sample an example
            x_near = self.transform(
                data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # Calculate the output of the x_near
            logits = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss, delta,True)
            # admix_g1=g_1.repeat(self.num_scale,1,1,1)
            # Calculate the output of the x_near
            _ = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.gram_loss(kwargs['f_ori'],mid_output)

            # Calculate the gradient of the x_near
            g_style = self.get_grad(loss, delta,True)

            # Compute the predicted point x_next
            x_next = x_near + self.alpha * (-g_1 / (torch.abs(g_1).mean(dim=(1, 2, 3), keepdim=True)))

            # Calculate the output of the x_next
            logits = self.get_logits(x_next)

            # Calculate the loss of the x_next
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_next
            g_2 = self.get_grad(loss, delta)

            # Calculate the gradients
            averaged_gradient += (1 - self.gamma) * g_1 + self.gamma * g_2
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor,averaged_style_gradient/self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)


        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient = 0, 0
        with torch.no_grad():
            _ = self.model(data)
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map,delta_start=delta)
            grad=self.sgam_noise([averaged_gradient, averaged_style_gradient], self.cc)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    

class SSEPPGN_Admix(PGNGRAM,Admix):


    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPPGN_Admix', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch,feature_layer=feature_layer, device=device)
        self.alpha = epsilon / epoch
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.cc=cc
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print("Selected layer is not in Model")
                exit()
            else:
                m_layers=dict(m.named_modules()).get(layer)
        return m_layers
    
    def random_orthogonal_noise(self,grads):
        result=torch.zeros_like(grads)
        radom_noise=torch.empty_like(grads).uniform_(-self.zeta,self.zeta)
        project_const_ = (radom_noise * grads).sum([1, 2, 3]) / ((grads * grads).sum([1, 2, 3]))
        for idx in range(grads.shape[0]):
            if project_const_[idx].item()<0:
                parallel_g = project_const_[idx].item() * grads[idx]
                orthogonal_g = radom_noise[idx] - parallel_g
                result[idx]=(orthogonal_g/torch.max(torch.abs(orthogonal_g)))*self.zeta
            else:
                result[idx]=radom_noise[idx]
        return result
    
    def __forward_hook(self,m,i,o):
        global mid_output
        # print(mid_outputs)
        mid_output=o
        # self.mid_output.append(o)

    def get_grad(self, loss, delta, retain_graph=False,**kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]



    def gram_loss(self,f_ori,f_adv):
        f_ori_repeat=f_ori.repeat(self.num_scale*self.num_admix,1,1,1)
        # f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat,method=self.k_m)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        loss=0
        for i in range(adv_gram.shape[0]):
            loss+=torch.nn.MSELoss()(ori_gram[i], adv_gram[i])
        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = 0
        averaged_style_gradient = 0
        for n in range(self.num_neighbor):

            # Random sample an example
            x_near = self.transform(
                data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # Calculate the output of the x_near
            logits = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss, delta,True)
            admix_g1=g_1.repeat(self.num_scale*self.num_admix,1,1,1)
            # Calculate the output of the x_near
            _ = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.gram_loss(kwargs['f_ori'],mid_output)

            # Calculate the gradient of the x_near
            g_style = self.get_grad(loss, delta,True)

            # Compute the predicted point x_next
            x_next = x_near + self.alpha * (-admix_g1 / (torch.abs(admix_g1).mean(dim=(1, 2, 3), keepdim=True)))

            # Calculate the output of the x_next
            logits = self.get_logits(x_next)

            # Calculate the loss of the x_next
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_next
            g_2 = self.get_grad(loss, delta)

            # Calculate the gradients
            averaged_gradient += (1 - self.gamma) * g_1 + self.gamma * g_2
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor,averaged_style_gradient/self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)


        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient = 0, 0
        with torch.no_grad():
            _ = self.model(data)
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map,delta_start=delta)
            grad=self.sgam_noise([averaged_gradient, averaged_style_gradient], self.cc)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    

class SSEPPGN_SIA(PGNGRAM,SIA):


    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPPGN_SIA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch,feature_layer=feature_layer, device=device)
        self.alpha = epsilon / epoch
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.cc=cc
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print("Selected layer is not in Model")
                exit()
            else:
                m_layers=dict(m.named_modules()).get(layer)
        return m_layers
    
    def random_orthogonal_noise(self,grads):
        result=torch.zeros_like(grads)
        radom_noise=torch.empty_like(grads).uniform_(-self.zeta,self.zeta)
        project_const_ = (radom_noise * grads).sum([1, 2, 3]) / ((grads * grads).sum([1, 2, 3]))
        for idx in range(grads.shape[0]):
            if project_const_[idx].item()<0:
                parallel_g = project_const_[idx].item() * grads[idx]
                orthogonal_g = radom_noise[idx] - parallel_g
                result[idx]=(orthogonal_g/torch.max(torch.abs(orthogonal_g)))*self.zeta
            else:
                result[idx]=radom_noise[idx]
        return result
    
    def __forward_hook(self,m,i,o):
        global mid_output
        mid_output=o

    def get_grad(self, loss, delta, retain_graph=False,**kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]



    def gram_loss(self,f_ori,f_adv):
        f_ori_repeat=f_ori.repeat(self.num_scale,1,1,1)
        # f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat,method=self.k_m)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        loss=0
        for i in range(adv_gram.shape[0]):
            loss+=torch.nn.MSELoss()(ori_gram[i], adv_gram[i])
        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = 0
        averaged_style_gradient = 0
        for n in range(self.num_neighbor):

            # Random sample an example
            x_near = self.transform(
                data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # Calculate the output of the x_near
            logits = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss, delta,True)
            admix_g1=g_1.repeat(self.num_scale,1,1,1)
            # Calculate the output of the x_near
            _ = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.gram_loss(kwargs['f_ori'],mid_output)

            # Calculate the gradient of the x_near
            g_style = self.get_grad(loss, delta,True)

            # Compute the predicted point x_next
            x_next = x_near + self.alpha * (-admix_g1 / (torch.abs(admix_g1).mean(dim=(1, 2, 3), keepdim=True)))

            # Calculate the output of the x_next
            logits = self.get_logits(x_next)

            # Calculate the loss of the x_next
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_next
            g_2 = self.get_grad(loss, delta)

            # Calculate the gradients
            averaged_gradient += (1 - self.gamma) * g_1 + self.gamma * g_2
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor,averaged_style_gradient/self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)


        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient = 0, 0
        with torch.no_grad():
            _ = self.model(data)
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map,delta_start=delta)
            grad=self.sgam_noise([averaged_gradient, averaged_style_gradient], self.cc)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    

class SSEPPGN_BSR(PGNGRAM,BSR):

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPPGN_BSR', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch,feature_layer=feature_layer, device=device)
        self.alpha = epsilon / epoch
        self.zeta = beta * epsilon
        self.gamma = gamma
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.cc=cc
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print("Selected layer is not in Model")
                exit()
            else:
                m_layers=dict(m.named_modules()).get(layer)
        return m_layers
    
    def random_orthogonal_noise(self,grads):
        result=torch.zeros_like(grads)
        radom_noise=torch.empty_like(grads).uniform_(-self.zeta,self.zeta)
        project_const_ = (radom_noise * grads).sum([1, 2, 3]) / ((grads * grads).sum([1, 2, 3]))
        for idx in range(grads.shape[0]):
            if project_const_[idx].item()<0:
                parallel_g = project_const_[idx].item() * grads[idx]
                orthogonal_g = radom_noise[idx] - parallel_g
                result[idx]=(orthogonal_g/torch.max(torch.abs(orthogonal_g)))*self.zeta
            else:
                result[idx]=radom_noise[idx]
        return result
    
    def __forward_hook(self,m,i,o):
        global mid_output
        # print(mid_outputs)
        mid_output=o
        # self.mid_output.append(o)

    def get_grad(self, loss, delta, retain_graph=False,**kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=retain_graph, create_graph=False)[0]


    def gram_loss(self,f_ori,f_adv):
        f_ori_repeat=f_ori.repeat(self.num_scale,1,1,1)
        # f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat,self.k_m)
        adv_gram=self.get_gram(f_adv,self.k_m)
        loss=0
        for i in range(adv_gram.shape[0]):
            loss+=torch.nn.MSELoss()(ori_gram[i], adv_gram[i])
        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = 0
        averaged_style_gradient = 0
        for n in range(self.num_neighbor):

            # Random sample an example
            x_near = self.transform(
                data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))

            # Calculate the output of the x_near
            logits = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss, delta,True)
            admix_g1=g_1.repeat(self.num_scale,1,1,1)
            # Calculate the output of the x_near
            _ = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.gram_loss(kwargs['f_ori'],mid_output)

            # Calculate the gradient of the x_near
            g_style = self.get_grad(loss, delta,True)

            # Compute the predicted point x_next
            x_next = x_near + self.alpha * (-admix_g1 / (torch.abs(admix_g1).mean(dim=(1, 2, 3), keepdim=True)))
            # x_next = x_near + self.alpha * (-g_1 / (torch.abs(g_1).mean(dim=(1, 2, 3), keepdim=True)))

            # Calculate the output of the x_next
            logits = self.get_logits(x_next)

            # Calculate the loss of the x_next
            loss = self.get_loss(logits, label)

            # Calculate the gradient of the x_next
            g_2 = self.get_grad(loss, delta)

            # Calculate the gradients
            averaged_gradient += (1 - self.gamma) * g_1 + self.gamma * g_2
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor,averaged_style_gradient/self.num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for PGN

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient = 0, 0
        with torch.no_grad():
            _ = self.model(data)
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map,delta_start=delta)
            grad=self.sgam_noise([averaged_gradient, averaged_style_gradient], self.cc)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
