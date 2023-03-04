import numpy as np
import torch
class message_collect():
    def __init__(self):
        self.DATA_list=[]
        self.DATA_NOW=[]
        pass


#添加点对，暂时不用
    def add_point_pair(self):
        pass

#一次全读完10000个点
    def read_all_once(self,pointnum=10000):
        for pointorder in range(1,pointnum+1):
            filenamex = "./dataperpoint/coordinate_x_number" + str(pointorder) + ".npy"
            filenamey = "./dataperpoint/coordinate_y_number" + str(pointorder) + ".npy"
            x=np.load(filenamex)
            y=np.load(filenamey)
            pointi = [x, y]
            self.DATA_list.append(pointi)
        pass
    def get_Data_at_Moment(self,time):
        Matrix=torch.zeros((1,200,100))
        x_dim=0
        y_dim=0
        for pointorder in range(1, 10000):
            point=self.DATA_list[pointorder]
            pointx=point[0]
            pointy = point[1]
            current_moment_x=pointx[time]
            current_moment_y = pointy[time]
            if x_dim%2==0:
                x_now = current_moment_x
            else:
                x_now = current_moment_y
            Matrix[0,x_dim,y_dim]=x_now
            x_dim+=1
            if x_dim==100:
                x_dim=0
                y_dim+=1
            # self.DATA_NOW.append(pointi)
        return Matrix
        # point=self.DATA_list[time]
        pass





