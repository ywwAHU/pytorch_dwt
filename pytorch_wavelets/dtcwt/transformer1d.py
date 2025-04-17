import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets.dtcwt.coeffs import qshift as _qshift
from pytorch_wavelets.dtcwt.coeffs import biort as _biort
from pytorch_wavelets.dtcwt.lowlevel import prep_filt


class DTCWTForward1D(nn.Module):
    """ 
    实现一维双树复小波变换的前向传播
    
    参数:
        J (int): 分解级别数量
        biort (str): 用于第一级分解的双正交滤波器
        qshift (str): 用于后续级别的Q-shift滤波器
        o_dim (int, optional): 输出维度，0表示最后一个维度，1表示第一个维度
        ri_dim (int, optional): 实部/虚部维度，0表示最后一个维度，1表示第一个维度
    """
    def __init__(self, J=3, biort='near_sym_a', qshift='qshift_a',
                 o_dim=0, ri_dim=1):
        super().__init__()

        # 检查参数
        self.J = J
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        
        # 获取滤波器
        h0o, h1o, h0a, h1a = _biort(biort)
        h0b, h0a, h1b, h1a = _qshift(qshift)
        
        # 准备滤波器
        self.h0o = prep_filt(h0o, 1)
        self.h1o = prep_filt(h1o, 1)
        self.h0a = prep_filt(h0a, 1)
        self.h1a = prep_filt(h1a, 1)
        self.h0b = prep_filt(h0b, 1)
        self.h1b = prep_filt(h1b, 1)
        
        # 注册滤波器为缓冲区
        self.register_buffer('h0o_', self.h0o)
        self.register_buffer('h1o_', self.h1o)
        self.register_buffer('h0a_', self.h0a)
        self.register_buffer('h1a_', self.h1a)
        self.register_buffer('h0b_', self.h0b)
        self.register_buffer('h1b_', self.h1b)

    def forward(self, x):
        """ 
        前向传播 - 双树复小波变换
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch, channel, length]
            
        返回:
            yl (torch.Tensor): 低频系数，形状为 [batch, channel, length/(2^J)]
            yh (list): 高频系数列表，每个元素是复数张量
        """
        # 检查输入维度
        dims = x.shape
        if len(dims) < 3:
            raise ValueError("Input tensor must have at least 3 dimensions")
        
        # 存储高频系数
        yh = []
        
        # 第一级分解 - 使用双正交滤波器
        lo = F.conv1d(x, self.h0o_, padding='same', stride=1)
        hi = F.conv1d(x, self.h1o_, padding='same', stride=1)
        
        # 下采样
        lo = lo[:, :, ::2]
        hi = hi[:, :, ::2]
        
        # 存储第一级高频系数
        yh.append(hi.unsqueeze(self.ri_dim))
        
        # 后续级别分解 - 使用Q-shift滤波器
        for j in range(1, self.J):
            # 树a分解
            loa = F.conv1d(lo, self.h0a_, padding='same', stride=1)
            hia = F.conv1d(lo, self.h1a_, padding='same', stride=1)
            
            # 树b分解
            lob = F.conv1d(lo, self.h0b_, padding='same', stride=1)
            hib = F.conv1d(lo, self.h1b_, padding='same', stride=1)
            
            # 下采样
            loa = loa[:, :, ::2]
            hia = hia[:, :, ::2]
            lob = lob[:, :, ::2]
            hib = hib[:, :, ::2]
            
            # 更新低频系数 - 使用树a的低频系数
            lo = loa
            
            # 构建复数高频系数
            if self.ri_dim == 0:
                hi = torch.stack((hia, hib), dim=-1)
            else:
                hi = torch.stack((hia, hib), dim=1)
            
            # 存储高频系数
            yh.append(hi)
        
        # 最终的低频系数
        yl = lo
        
        return yl, yh


class DTCWTInverse1D(nn.Module):
    """ 
    实现一维双树复小波变换的逆变换
    
    参数:
        J (int): 分解级别数量
        biort (str): 用于第一级分解的双正交滤波器
        qshift (str): 用于后续级别的Q-shift滤波器
        o_dim (int, optional): 输出维度，0表示最后一个维度，1表示第一个维度
        ri_dim (int, optional): 实部/虚部维度，0表示最后一个维度，1表示第一个维度
    """
    def __init__(self, J=3, biort='near_sym_a', qshift='qshift_a',
                 o_dim=0, ri_dim=1):
        super().__init__()

        # 检查参数
        self.J = J
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        
        # 获取滤波器
        g0o, g1o, g0a, g1a = _biort(biort, True)
        g0b, g0a, g1b, g1a = _qshift(qshift, True)
        
        # 准备滤波器
        self.g0o = prep_filt(g0o, 1)
        self.g1o = prep_filt(g1o, 1)
        self.g0a = prep_filt(g0a, 1)
        self.g1a = prep_filt(g1a, 1)
        self.g0b = prep_filt(g0b, 1)
        self.g1b = prep_filt(g1b, 1)
        
        # 注册滤波器为缓冲区
        self.register_buffer('g0o_', self.g0o)
        self.register_buffer('g1o_', self.g1o)
        self.register_buffer('g0a_', self.g0a)
        self.register_buffer('g1a_', self.g1a)
        self.register_buffer('g0b_', self.g0b)
        self.register_buffer('g1b_', self.g1b)

    def forward(self, yl, yh):
        """ 
        逆变换 - 双树复小波逆变换
        
        参数:
            yl (torch.Tensor): 低频系数，形状为 [batch, channel, length/(2^J)]
            yh (list): 高频系数列表，每个元素是复数张量
            
        返回:
            y (torch.Tensor): 重构信号，形状为 [batch, channel, length]
        """
        # 检查输入
        if len(yh) != self.J:
            raise ValueError(f"Expected {self.J} levels of detail coefficients, got {len(yh)}")
        
        # 从最粗糙的尺度开始重构
        lo = yl
        
        # 逐级重构
        for j in range(self.J-1, 0, -1):
            # 获取当前级别的高频系数
            hi = yh[j]
            
            # 分离实部和虚部
            if self.ri_dim == 0:
                hia = hi[..., 0]
                hib = hi[..., 1]
            else:
                hia = hi[:, 0, ...]
                hib = hi[:, 1, ...]
            
            # 上采样
            batch, channel, length = lo.shape
            loa_up = torch.zeros(batch, channel, length*2, device=lo.device)
            loa_up[:, :, ::2] = lo
            hia_up = torch.zeros(batch, channel, length*2, device=hia.device)
            hia_up[:, :, ::2] = hia
            
            lob_up = torch.zeros(batch, channel, length*2, device=lo.device)
            lob_up[:, :, ::2] = lo
            hib_up = torch.zeros(batch, channel, length*2, device=hib.device)
            hib_up[:, :, ::2] = hib
            
            # 树a重构
            ya = F.conv1d(loa_up, self.g0a_, padding='same', stride=1) + \
                 F.conv1d(hia_up, self.g1a_, padding='same', stride=1)
            
            # 树b重构
            yb = F.conv1d(lob_up, self.g0b_, padding='same', stride=1) + \
                 F.conv1d(hib_up, self.g1b_, padding='same', stride=1)
            
            # 合并两棵树的结果
            lo = (ya + yb) / 2.0
        
        # 第一级重构 - 使用双正交滤波器
        hi = yh[0]
        if self.ri_dim == 0:
            hi = hi[..., 0]  # 只使用实部
        else:
            hi = hi[:, 0, ...]  # 只使用实部
        
        # 上采样
        batch, channel, length = lo.shape
        lo_up = torch.zeros(batch, channel, length*2, device=lo.device)
        lo_up[:, :, ::2] = lo
        hi_up = torch.zeros(batch, channel, length*2, device=hi.device)
        hi_up[:, :, ::2] = hi
        
        # 最终重构
        y = F.conv1d(lo_up, self.g0o_, padding='same', stride=1) + \
            F.conv1d(hi_up, self.g1o_, padding='same', stride=1)
        
        return y


class DTCWT1D(nn.Module):
    """
    一维双树复小波变换完整实现
    
    参数:
        J (int): 分解级别
        biort (str): 用于第一级分解的双正交滤波器
        qshift (str): 用于后续级别的Q-shift滤波器
        o_dim (int, optional): 输出维度，0表示最后一个维度，1表示第一个维度
        ri_dim (int, optional): 实部/虚部维度，0表示最后一个维度，1表示第一个维度
    """
    def __init__(self, J=3, biort='near_sym_a', qshift='qshift_a',
                 o_dim=0, ri_dim=1):
        super().__init__()
        
        self.J = J
        self.biort = biort
        self.qshift = qshift
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        
        # 前向和逆变换模块
        self.forward_transform = DTCWTForward1D(
            J=J, biort=biort, qshift=qshift, o_dim=o_dim, ri_dim=ri_dim)
        self.inverse_transform = DTCWTInverse1D(
            J=J, biort=biort, qshift=qshift, o_dim=o_dim, ri_dim=ri_dim)
    
    def forward(self, x):
        """前向变换"""
        return self.forward_transform(x)
    
    def inverse(self, yl, yh):
        """逆变换"""
        return self.inverse_transform(yl, yh)