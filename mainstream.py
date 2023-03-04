from DNN import MyDNN
from messagecollector import message_collect
from actor_proxy import actor_proxy
from lossoptimizer import lossoptimizer
actor_s=actor_proxy()           
optimizer=lossoptimizer()
optimizer.set_NET(actor_s.actor)
Data_collection=message_collect()
Data_collection.read_all_once()
time_stamp=0
while True:
    lastpoints=Data_collection.get_Data_at_Moment(time_stamp)
    nextpoints=Data_collection.get_Data_at_Moment(time_stamp+1)
    predicted_point=actor_s.predict(lastpoints)
    loss=optimizer.calcloss(predicted_point,nextpoints)
    print(loss)
    loss.backward()
    optimizer.updateNetwork()
    #some verify fu
    # nction
    time_stamp+=1
    if time_stamp>=50:
        time_stamp=0



    
