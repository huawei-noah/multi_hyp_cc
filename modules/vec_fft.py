import torch
import torch.nn as nn
import numpy as np
import math
from modules.fft_vec import fft2vec
from torch.autograd import Variable
import sys

def vec2fft(x_vec, apply_scaling):
    x_vec = x_vec.clone()
    n = int(math.sqrt(x_vec.shape[1]))
    hn = n // 2
    bs = x_vec.shape[0]

    real = torch.zeros(bs, n, n).type(x_vec.type())
    imag = torch.zeros(bs, n, n).type(x_vec.type())

    if apply_scaling:
        x_vec = x_vec / math.sqrt(2)

    # fill real values
    real[:, 0:hn+1, 0] = x_vec[:, 0:hn+1]
    real[:, 0:hn+1, hn] = x_vec[:, hn+1 + torch.arange(0, hn+1)]
    real[:, :, 1:hn] = x_vec[:, n+2 + torch.arange(0, n*(hn-1))].view(bs, hn-1, n).transpose(1,2)

    # apply scaling
    if apply_scaling:
        real[:, 0, 0] = real[:, 0, 0] * math.sqrt(2)
        real[:, 0, hn] = real[:, 0, hn] * math.sqrt(2)
        real[:, hn, 0] = real[:, hn, 0] * math.sqrt(2)
        real[:, hn, hn] = real[:, hn, hn] * math.sqrt(2)

    # fill imaginary values
    s = n*n//2 + 2
    imag[:, 1:hn, 0] = x_vec[:, s + torch.arange(0, hn-1)]
    imag[:, 1:hn, hn] = x_vec[:, s + hn-1 + torch.arange(0, hn-1)]
    imag[:, :, 1:hn] = x_vec[:, s + n-2 + torch.arange(0, n*(hn-1))].view(bs, hn-1, n).transpose(1,2)

    # fill the rest
    real[:, 0, hn+1:] = torch.flip(real[:, 0, 1:hn], [1])
    imag[:, 0, hn+1:] = -torch.flip(imag[:, 0, 1:hn], [1])

    real[:, 1:, hn+1:] = torch.flip(real[:, 1:, 1:hn], [1,2]) # rotate 180 degrees
    imag[:, 1:, hn+1:] = -torch.flip(imag[:, 1:, 1:hn], [1,2]) # rotate 180 degrees

    real[:, hn+1:, [0, hn]] = torch.flip(real[:, 1:hn, [0, hn]], [1])
    imag[:, hn+1:, [0, hn]] = -torch.flip(imag[:, 1:hn, [0, hn]], [1])

    out = torch.cat((real.unsqueeze(-1), imag.unsqueeze(-1)), 3)

    return out

class Vec2Fft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_vec, apply_scaling = True):
        if ctx is not None:
            ctx.apply_scaling = apply_scaling

        return vec2fft(x_vec, apply_scaling)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        I = torch.ifft(grad_output, 2)
        I[:,:,:,1] = 0
        F = torch.fft(I, 2)
        #F = grad_output

        out = fft2vec(F, ctx.apply_scaling)

        return out

if __name__ == '__main__':
    from torch.autograd import Variable

    if True:
        vec_fft = Vec2Fft.apply
        B = 1
        N = 64
        tensor = 1000*torch.randn(B, N, N)
        #tensor[:,:,:,1] = 0 # no imaginary
        #tensor = torch.zeros(B, N, N, 2)
        #for i in range(N):
        #    for j in range(N):
        #        tensor[:,i,j,0] = i*N+j

        print('tensor', tensor[0,:,:])

        tensor_fft = torch.rfft(tensor, 2, onesided=False)

        print('tensor_fft', tensor_fft[0,:,:,0])
        print('tensor_fft', tensor_fft[0,:,:,1])

        vec = fft2vec(tensor_fft, apply_scaling=True)

        #print(vec.squeeze())

        fft_recovered = vec_fft(vec)

        print('rec_tensor_fft', fft_recovered[0,:,:,0])
        print('rec_tensor_fft', fft_recovered[0,:,:,1])

        print(torch.abs(tensor_fft-fft_recovered).sum())
        print(torch.abs(tensor_fft-fft_recovered).max())

        tensor_recovered = torch.irfft(fft_recovered, 2, onesided=False)

        print('rec_tensor', tensor_recovered[0,:,:])

        print(torch.abs(tensor-tensor_recovered).sum())
        print(torch.abs(tensor-tensor_recovered).max())

        #sys.exit(0)

    if False:

        B = 1
        N = 8
        rand_tensor = 100*torch.randn(B, N, N)
        rand_tensor2 = 100*torch.randn(B, N, N)
        for i in range(1):
            vec_fft = Vec2Fft.apply

            tensor = rand_tensor.clone()
            tensor[:,:,:,1] = 0 # no imaginary

            tensor2 = rand_tensor2.clone()
            #tensor2[:,:,:,1] = 0 # no imaginary
            tensor2_fft = torch.rfft(tensor2, 2, onesided=False)

            tensor_vec = Variable(fft2vec(torch.rfft(tensor, 2, onesided=False),True), requires_grad=True)
            tensor_fft = vec_fft(tensor_vec)
            tensor_fft.retain_grad()

            loss = 0.5*((tensor_fft-tensor2_fft)*(tensor_fft-tensor2_fft)).sum()

            loss.backward()
            #print(tensor_fft.grad)
            print(tensor_vec.grad)

    if False:
        B = 1
        N = 4
        for i in range(N):
            for j in range(N):
                for z in range(2):
                    grad = torch.zeros(B, N, N, 2)
                    grad[0,i,j,z] = 1
                    print(i,j,z,fft2vec(grad, True))

    #####
    if True:
        print('grad check')
        vec_fft = Vec2Fft.apply
        B = 2
        N = 64
        tensor = torch.zeros(B, N, N, dtype=torch.float64)
        zeros = torch.zeros(B, N, N, dtype=torch.float64)

        for i in range(N):
            for j in range(N):
                tensor[:,i,j] = i*N+j

        tensor_vec = Variable(fft2vec(torch.rfft(torch.cat((tensor, zeros), -1), 2, onesided=False),True), requires_grad=True)

        print(torch.autograd.gradcheck(lambda t: vec_fft(t), tensor_vec))
