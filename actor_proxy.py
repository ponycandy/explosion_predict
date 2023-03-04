from DNN import MyDNN
import torch
class actor_proxy():
    def __init__(self):
        self.actor = MyDNN()
        pass
    def predict(self,points):
        real_input=self.TransformData(points)
        points_predicted=self.actor(real_input)
        # DATA_list=self.TransformData(points_predicted)
        return points_predicted
    def TransformData(self,points):
        cutting_time=0
        # for iter in range(1,cutting_time):
        #     horizon=torch.zeros((1,1,100+2*iter))
        #     horizon_1 =torch.zeros((1, 1, 100 + 2 * iter))
        #     vertical=torch.zeros((1,200+2*(iter-1),1))
        #     vertical_1 = torch.zeros((1, 200 + 2 * (iter - 1), 1))
        #     mid1=torch.cat((vertical,points),dim=2)
        #     mid2=torch.cat((mid1,vertical_1),dim=2)
        #     mid3 = torch.cat((horizon, mid2),dim=1)
        #     mid4 = torch.cat((mid3, horizon_1),dim=1)
        #     points=mid4
        return points
        pass