from json import load
from unicodedata import decimal
from argon2 import Parameters
from lib import *
import torch
from config import *
# hàm số tạo 1 list chứa các đường link dẫn ảnh
# liệt kê các path dẫn đến ảnh vào 1 list rồi truyền list đó vào data_class
def make_datapath_list(phase='train'):
    rootpath ="./data/hymenoptera_data/"
    target_path = osp.join(rootpath+ phase+ "/**/*.jpg")
    # print(target_path)
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

# Hàm số để training model
# training
def train_model(net, datalaoder_dict, criterior, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        
        # move network to device( GPU/CPU)
        net.to(device)
        
        torch.backends.cudnn.benchmark = True
        
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(datalaoder_dict[phase]):
                #move inputs, labels to GPU /CPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #set grandiant of optimizer to be zero
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs,1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item() *inputs.size(0) # lay patch_size
                    epoch_corrects += torch.sum(preds==labels.data)
                    
            epoch_loss = epoch_loss / len(datalaoder_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(datalaoder_dict[phase].dataset)
            
            print("{} Loss:{:.4f} Accuracy:{:.4f}".format(phase, epoch_loss, epoch_accuracy))
    
    torch.save(net.state_dict(), save_path)

def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []
    
    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weigth","classifier.0.bias","classifier.3.weigth","classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weigth","classifier.6.bias"]
    
    for name, param in net.named_parameters():
        if name  in update_param_name_1[0] :
            param.requires_grad = True
            params_to_update_1.append(param)
        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update_3.append(param)
        
        else:
            param.requires_grad = False
        
    return params_to_update_1, params_to_update_2, params_to_update_3
            
            
def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)
    print(net)
    
    for name, param in net.named_parameters():
        print(name, param)
    # load_weights = torch.load(model_path,map_location("cuda:0","cpu"))
    # net.load_state_dict(load_weights)