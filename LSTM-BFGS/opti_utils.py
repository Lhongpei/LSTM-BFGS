import torch
class BFGS:
    def __init__(self,batchsize,DIM,device,x):
        self.I = torch.eye(DIM, device=x.device, dtype=x.dtype,requires_grad=True).unsqueeze(0).expand(batchsize, -1, -1)
        self.H = self.I.clone()
    def H_BFDS_update(self,s,y):
        st=s.unsqueeze(1)
        s=s.unsqueeze(2)
        yt=y.unsqueeze(1)
        y=y.unsqueeze(2)
        sy=torch.matmul(s,yt)
        ss=torch.matmul(s,st)
        yst=torch.matmul(y,st)
        yts=torch.matmul(yt,s)
        #print(yts.min())
        yHy=torch.matmul(torch.matmul(yt,self.H),y)
        self.H=self.H-(torch.matmul(sy,self.H)+torch.matmul(self.H,yst))/yts+(yHy/yts+1)*ss/yts
        #print(self.H)

# define a logistic regression function
def f(W,Y,x):
    """quadratic function : f(\theta) = \|W\theta - y\|_2^2"""
    return ((torch.matmul(W,x.unsqueeze(-1)).squeeze()-Y)**2).sum(dim=1)

def logistic(A,b,x,lamb,n):
    sig=torch.sigmoid(torch.matmul(A,x.unsqueeze(-1)).squeeze())
    return -(b*torch.log(sig)+(1-b)*torch.log(1-sig)).mean(dim=1)+lamb/n*torch.norm(x,dim=-1)


def strong_wolfe(W,Y,batchsize,f, x, step_dir, loss, grad, device, c1=1e-4, c2=0.9, max_iter=10):
    """
    Strong Wolfe Line Search
    Args:
    - f: Function to be optimized
    - x: Current point
    - step_dir: Search direction
    - loss: Loss at current point
    - grad: Gradient at current point
    - c1: Armijo condition parameter
    - c2: Curvature condition parameter
    - max_iter: Maximum number of iterations

    Returns:
    - step_size: Calculated step size
    """

    # Initialize step size
    step_size = torch.ones((batchsize,1),device=device)
    grad=grad.unsqueeze(1)
    #print(step_dir.shape)
    #step_dir=step_dir.unsqueeze(2)
    for i in range(max_iter):
        new_loss = f(W,Y,x + step_size * step_dir.squeeze())
        
        armijo_condition = new_loss > loss + c1 * step_size.squeeze() * torch.matmul(grad, step_dir).squeeze()
        #print(armijo_condition.shape)
        new_grad = torch.autograd.grad(new_loss.mean(), x)[0]
        curvature_condition = torch.matmul(new_grad.unsqueeze(1), step_dir).squeeze() < c2 * torch.matmul(grad, step_dir).squeeze()

        if torch.sum(armijo_condition)==0 and torch.sum(curvature_condition)==0:
            return step_size
        elif torch.sum(armijo_condition)>0:
            step_size[armijo_condition] *= 0.5  # Reduce step size by half
        elif torch.sum(curvature_condition)>0:
            step_size[curvature_condition] *= 2.0  # Double step size

    return step_size  # Return step size after maximum iterations reached

# Usage Example:
# Define your function f, starting point x, loss, gradient, and search direction
# Then call strong_wolfe function to get the step size
# step_size = strong_wolfe(f, x, step_dir, loss, grad)
