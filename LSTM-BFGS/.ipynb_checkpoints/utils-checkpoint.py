import torch
def H_BFGS(H,x,x_past,size):

    x_del = x - x_past
    s = x_del[:size//2]
    y = x_del[size//2:]

    sty=torch.matmul(s,y.t())
    syt=torch.matmul(s.t(),y)
    yst=torch.matmul(y.t(),s)
    sst=torch.matmul(s.t(),)

    H_bfgs =  H-(torch.matmul(syt,H)+torch.matmul(H,yst)+(torch.matmul(y,torch.matmul(H,y.t()))/yts+1)*sst)/yts

    return H_bfgs

def H_SR1(H,x,x_past,size):

    x_del = x - x_past
    s = x_del[:size//2]
    y = x_del[size//2:]
    Hy_s = torch.matmul(H,y.t())-s.t()

    H_sr1 =  H-torch.matmul(Hy_s,Hy_s.t())/torch.matmul(Hy_s.t(),y.t())

    return H_sr1