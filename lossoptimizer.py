import torch
import torch.optim as optim
class lossoptimizer():
    def __init__(self):
        self.LR=0.001
        pass
    def set_NET(self,actorNet):
        self.actorNet=actorNet

        self.optimizer = optim.AdamW(self.actorNet.parameters(), lr=self.LR, amsgrad=True)
        self.optimizer.zero_grad()
    def get_sample(self):
        state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=self.replaybuff.get_Batch_data(self.BATCHSIZE)
        return state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states
    def calcloss(self,predicted_point,nextpoints):
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_point, nextpoints)
        return loss
    def updateNetwork(self,update_method=0):
        if(update_method==0):#此方法激活使用优化器
            torch.nn.utils.clip_grad_value_(self.actorNet.parameters(), 100) #梯度裁剪，一种防止梯度爆炸的优化策略，非必要
            self.optimizer.step()
            self.optimizer.zero_grad()
            return
        if(update_method==1):#此方法激活手动优化，极不推荐
            for param in self.actorNet.parameters():
                param.data = param.data-param.grad*self.LR
            self.actorNet.zero_grad()