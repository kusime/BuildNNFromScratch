class SGD:
    def __init__(self,lr) -> None:
        self.lr = lr
    
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
