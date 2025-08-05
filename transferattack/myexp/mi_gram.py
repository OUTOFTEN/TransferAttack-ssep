import math

import torch
import copy

from ..utils import *
from ..gradient.mifgsm import MIFGSM
from ..input_transformation.admix import Admix
from ..input_transformation.dim import DIM
from ..input_transformation.sia import SIA
from ..input_transformation.bsr import BSR

import torch.nn.functional as F

mid_outputs = []
mid_grads = []
class MIGRAM(MIFGSM):


    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0, num_ens=1,
                 targeted=False, random_start=False, feature_layer='layer3',beta=4.0,num=20,cc=1.0,
                 norm='linfty', loss='crossentropy', device=None, attack='MIGRAM',drop_rate=0.0, **kwargs):
        super().__init__(model_name, epsilon, alpha)
        self.num_ens = num_ens
        self.drop_rate = drop_rate
        self.zeta = beta * epsilon
        self.targeted=targeted
        self.random_start=random_start
        self.mid_output=[]
        self.feature_layer_names=feature_layer
        self.feature_layer = self.find_layer(feature_layer)
        self.op=[self.blur,self.random_sharpen_image,self.adjust_colors]
        self.num=num
        self.flag=True
        self.cc=cc

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
    
    def __forward_hook(self,m,i,o):
        global mid_output


        mid_output=o
      

    def get_gram(self,feature):
        assert len(feature.shape)==4
        b,c,w,h=feature.shape
        f=feature.view(b,c,w*h)
        f_t=torch.transpose(f,1,2)
        result=torch.zeros((b,c,c)).cuda()
        for i in range(b):
            result[i]=torch.mm(f[i],f_t[i])
        return result

    def gram_loss(self,f_ori,f_adv,test=False):
        # f_ori_repeat=f_ori.repeat(self.num_scale,1,1,1)
        f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat)
        adv_gram=self.get_gram(f_adv)
        loss=0
        print_loss=[]

        for i in range(adv_gram.shape[0]):
            print_loss.append((torch.nn.MSELoss()(ori_gram[i], adv_gram[i])).item())
            loss+=torch.nn.MSELoss()(ori_gram[i], adv_gram[i])

        return loss
    

    
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

    def sgam_noise(self, maplist, cc):
        assert len(maplist) == 2
        a, b = maplist[0]/torch.mean(torch.abs(maplist[0]),[1,2,3],True), maplist[1]/torch.mean(torch.abs(maplist[1]),[1,2,3],True)
        # a, b = maplist[0], maplist[1]
        project_const_ = (a * b).sum([1, 2, 3]) / ((b * b).sum([1, 2, 3]))
        noise=torch.zeros_like(b)
        for c in range(project_const_.shape[0]):
            if project_const_[c]<=0:
                parallel_g = project_const_[c].item() * b[c]
                orthogonal_g = a[c] - parallel_g
                noise[c] = b[c]*cc + orthogonal_g
            else:
                noise[c]=b[c]*cc+a[c]
        return noise
    
    def cauculate_avg_style_g(self,data,delta,ori_feature_map,momentum,num,delta_start,i):
        grad_style=0
        if num==0:
            x_near = self.transform(data + delta)
            self.mid_output.clear()
            # Obtain the output
            logits = self.get_logits(x_near, momentum=momentum)
            # Calculate the loss
            loss_style = self.gram_loss(ori_feature_map, mid_output)
            # print(loss_out/loss_style)
            self.model.zero_grad()
            grad_style=torch.autograd.grad(loss_style, delta, retain_graph=False, create_graph=False)[0]



            return grad_style
        else:
            for n in range(num):

    

                x_near = self.transform(data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))
                # x_near = self.transform(data + delta)
                self.mid_output.clear()
                # Obtain the output
                logits = self.get_logits(x_near, momentum=momentum)
                # Calculate the loss
                loss_style= self.gram_loss(ori_feature_map, mid_output)
                # print(loss_style)
                self.model.zero_grad()
                grad_style+=torch.autograd.grad(loss_style, delta, retain_graph=False, create_graph=False)[0]
            return grad_style/num
        
    def cauculate_avg_cs_g(self,data,delta,label,momentum,num,delta_start,i):
        grad_out=0
        logits=0
        if num==0:
            x_near = self.transform(data + delta)
            # Obtain the output
            logits = self.get_logits(x_near, momentum=momentum)
            # Calculate the loss
            loss_out=self.get_loss(logits,label)
  
            self.model.zero_grad()
            grad_out = torch.autograd.grad(loss_out, delta, retain_graph=False, create_graph=False)[0]


            return grad_out
        else:
            for n in range(num):

          

                x_near = self.transform(data + delta + torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device))
                # x_near = self.transform(data + delta)
                # Obtain the output
                logits = self.get_logits(x_near, momentum=momentum)
                # Calculate the loss
                loss_out=self.get_loss(logits,label)
                self.model.zero_grad()
                grad_out += torch.autograd.grad(loss_out, delta, retain_graph=False, create_graph=False)[0]
            return grad_out/num

    def forward(self, data, label, **kwargs):
 
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
 
        delta = self.init_delta(data)


        if isinstance(self.feature_layer,list):
            h=[f.register_forward_hook(self.__forward_hook) for f in self.feature_layer]
        else:
            h = self.feature_layer.register_forward_hook(self.__forward_hook)
        # h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)
        with torch.no_grad():
            output_random = self.model(data)
            ori_feature_map = copy.deepcopy(mid_output)
        momentum=0
        for i in range(self.epoch):
            # Obtain the output

            grad_out=self.cauculate_avg_cs_g(data,delta,label,momentum,self.num,delta,i)

            if i>0:
                grad_style=self.cauculate_avg_style_g(data,delta,ori_feature_map,momentum,self.num,delta,i)
                grad=self.sgam_noise([grad_style,grad_out],1-(i/self.epoch))
                # grad=self.sgam_noise([grad_out,grad_style],1)
                # grad=grad_style / torch.mean(torch.abs(grad_style), [1, 2, 3], True)
            else:
                grad = grad_out / torch.mean(torch.abs(grad_out), [1, 2, 3], True)
            # grad=self.sgam_noise([grad_out,grad_style],1-i/self.epoch)
            momentum=self.get_momentum(grad,momentum)
            # momentum = self.decay * momentum + grad_out
            # Update adversarial perturbation
            delta = self.update_delta(delta, data,momentum, self.alpha)
            # print('*'*30)
        if isinstance(h,list):
            for hf in h:
                hf.remove()
        else:
            h.remove()
        return delta.detach()


class SSEPMI(MIFGSM):


    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,gep=-1,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPMI', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch)
        self.alpha = alpha if targeted else epsilon/epoch
        self.zeta = beta * epsilon
        # self.gamma = gamma
        self.epoch = epoch if not targeted else 200
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.targeted=targeted
        self.cc=cc
        self.norm=norm
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print(f"SSEPMI--Selected layer:{layer} is not in Model. Select in {model_layers_name}")
                exit()
            else:
                m_layers=dict(m.named_modules()).get(layer)
        return m_layers
    
    
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

        B, C, W, H = x.shape
        x_flat = x.view(B, C, -1)  # (B, C, D)

        l2_distance = torch.cdist(x_flat, x_flat, p=2)  # (B, C, C)

        if gamma is None:
            if sigma is None:
          
                with torch.no_grad():
                    median_distance = torch.median(l2_distance.view(B, -1), dim=1).values.clamp(min=1e-8)  # (B,)
                    gamma = 1.0 / median_distance  # (B,)
                    gamma = gamma.view(B, 1, 1)  
            else:
                gamma = 1.0 / sigma  

        kernel_matrix = torch.exp(-gamma * (l2_distance**2))
        return kernel_matrix
    
    def batch_channel_laplacian_kernel(self, x, gamma=None, sigma=None):

        B, C, W, H = x.shape
   
        x_flat = x.view(B, C, -1)  # (B, C, D)

   
        l1_distance = torch.cdist(x_flat, x_flat, p=1)  # (B, C, C)

 
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
        elif method=='laplacian':
            result = self.batch_channel_laplacian_kernel(feature)
        elif method=='gaussian':
            result = self.batch_channel_gaussian_kernel(feature)
        else:
            print('method error')
            exit()
        return result

    def gram_loss(self,f_ori,f_adv):

        ori_gram=self.get_gram(f_ori,method=self.k_m)
        # ori_gram=self.get_gram(f)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        # print(loss_type)
        loss=torch.nn.MSELoss()(ori_gram,adv_gram)

        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = torch.zeros_like(delta)
        averaged_style_gradient = torch.zeros_like(delta)
        # delta_start = kwargs['delta_start']

        for n in range(self.num_neighbor):
            # Random sample an example
            noise = torch.zeros_like(delta).uniform_(-self.zeta, self.zeta)
            x_near = self.transform(data + delta + noise)

            # Calculate the output and loss of the x_near
            logits = self.get_logits(x_near)
            loss1 = self.get_loss(logits, label)
            # loss1=torch.nn.MSELoss(reduction='sum')(nn.Softmax()(logits),nn.Softmax()(ori_logits))

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss1, delta, True)
            averaged_gradient += g_1


            loss_style = self.gram_loss(kwargs['f_ori'], mid_output)
            g_style = self.get_grad(loss_style, delta, False)
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor, averaged_style_gradient / self.num_neighbor

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
        # print(self.cc)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient, ori_feature_map = 0,0,0
        features=[]
        # print(self.zeta)
        with torch.no_grad():
            ori_logits = self.model(data)
            # target_feature=torch.max(mid_output, dim=0, keepdim=True).values-mid_output
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map)
            if self.targeted:
                averaged_gradient=-averaged_gradient
 
            grad=self.sgam_noise([averaged_gradient,averaged_style_gradient], self.cc)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    

class SSEPMI_DIM(SSEPMI,DIM):


    def __init__(self, model_name,feature_layer, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,cc=0.7,gep=-1,k_m='laplacian',
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='SSEPMI_DIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch,feature_layer=feature_layer, device=device)
        self.alpha = alpha if targeted else epsilon/epoch
        self.zeta = beta * epsilon
        # self.gamma = gamma
        self.epoch = epoch if not targeted else 200
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.feature_layer=self.find_layer(feature_layer)
        self.mid_output=[]
        self.targeted=targeted
        self.cc=cc
        self.norm=norm
        self.k_m=k_m

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        model_layers_name=[mn for mn,_ in m.named_modules()]
        for layer in parser:
            if layer not in model_layers_name:
                print(f"SSEPMIDIM--Selected layer:{layer} is not in Model. Select in {model_layers_name}")
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


    def gram_loss(self,f_ori,f_adv):

        ori_gram=self.get_gram(f_ori,method=self.k_m)
        # ori_gram=self.get_gram(f)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        # print(loss_type)
        loss=torch.nn.MSELoss()(ori_gram,adv_gram)

        return loss

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        """
        Calculate the averaged updated gradient
        """
        averaged_gradient = torch.zeros_like(delta)
        averaged_style_gradient = torch.zeros_like(delta)
        # delta_start = kwargs['delta_start']

        for n in range(self.num_neighbor):
            # Random sample an example
            noise = torch.zeros_like(delta).uniform_(-self.zeta, self.zeta)
            x_near = self.transform(data + delta + noise)

            # Calculate the output and loss of the x_near
            logits = self.get_logits(x_near)
            loss1 = self.get_loss(logits, label)
            # loss1=torch.nn.MSELoss(reduction='sum')(nn.Softmax()(logits),nn.Softmax()(ori_logits))

            # Calculate the gradient of the x_near
            g_1 = self.get_grad(loss1, delta, True)
            averaged_gradient += g_1

   

            loss_style = self.gram_loss(kwargs['f_ori'], mid_output)
            g_style = self.get_grad(loss_style, delta, False)
            averaged_style_gradient += g_style

        return averaged_gradient / self.num_neighbor, averaged_style_gradient / self.num_neighbor

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
        # print(self.cc)

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        momentum, averaged_gradient, ori_feature_map = 0,0,0
        features=[]
        # print(self.zeta)
        with torch.no_grad():
            ori_logits = self.model(data)
            # target_feature=torch.max(mid_output, dim=0, keepdim=True).values-mid_output
            ori_feature_map = copy.deepcopy(mid_output)
        for i in range(self.epoch):
            # Calculate the averaged updated gradient
            averaged_gradient ,averaged_style_gradient = self.get_averaged_gradient(data, delta, label,epcho=i,f_ori=ori_feature_map)
            if self.targeted:
                averaged_gradient=-averaged_gradient
            grad=self.sgam_noise([averaged_gradient,averaged_style_gradient], self.cc)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    

class SSEPMI_Admix(SSEPMI,Admix):

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,k_m='laplacian',cc=0.7,
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPMI_Admix', **kwargs):
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

    # def __backward_hook(self,m,i,o):
    #     global mid_grad
    #     mid_grad = o

    def gram_loss(self,f_ori,f_adv):
        f_ori_repeat=f_ori.repeat(self.num_scale*self.num_admix,1,1,1)
        # f_ori_repeat=f_ori
        ori_gram=self.get_gram(f_ori_repeat,method=self.k_m)
        adv_gram=self.get_gram(f_adv,method=self.k_m)
        loss=torch.nn.MSELoss()(ori_gram,adv_gram)
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
            _ = self.get_logits(x_near)

            # Calculate the loss of the x_near
            loss = self.gram_loss(kwargs['f_ori'],mid_output)

            # Calculate the gradient of the x_near
            g_style = self.get_grad(loss, delta,False)


            # Calculate the gradients
            averaged_gradient += g_1
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
    

class SSEPMI_SIA(SSEPMI,SIA):

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,k_m='laplacian',cc=0.7,
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPMI_SIA', **kwargs):
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
        epcho=kwargs['epcho']
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
            g_style = self.get_grad(loss, delta,False)


            # Calculate the gradients
            averaged_gradient += g_1
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
    

class SSEPMI_BSR(SSEPMI,BSR):


    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, beta=3.5, gamma=0.5, num_neighbor=20, epoch=10,
                 decay=1., targeted=False,k_m='laplacian',cc=0.7,
                 random_start=False, norm='linfty', loss='crossentropy',feature_layer='layer3', device=None, attack='SSEPMI_BSR', **kwargs):
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
            g_style = self.get_grad(loss, delta,False)

            # Calculate the gradients
            averaged_gradient += g_1
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
            if i >0:
                grad=self.sgam_noise([averaged_gradient, averaged_style_gradient], self.cc)
            else:
                grad=averaged_gradient
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)
 
            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()