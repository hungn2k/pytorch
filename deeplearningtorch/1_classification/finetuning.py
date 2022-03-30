from lib import *
from image_transform import  ImageTransformer
from config import *
from utils import make_datapath_list, train_model, params_to_update, load_module
from Dataset import MyDataset

def main():
    
# gọi list đó ra
    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')
        
    train_dataset = MyDataset(train_list,transform=ImageTransformer(size,mean,std),phase="train")
    val_dataset = MyDataset(val_list,transform=ImageTransformer(size,mean,std),phase="val")

# Tạo dataloader từ dataset
    batch_size = 4   # nhom anh

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = False)

    datalaoder_dict = {"train": train_dataloader , "val": val_dataloader}

# Tạo network
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)
    # print(net)
    # setting mode 
    # net = net.train()

# Tạo hàm loss
    criterior = nn.CrossEntropyLoss()
    
# Tạo optimizer
    #finetuning
    params1, params2, params3 = params_to_update(net)
    optimizer = optim.SGD([
        {'param': params1, 'lr': 1e-4},
        {'param': params2, 'lr': 5e-4},
        {'param': params3, 'lr': 1e-3}
        ], momentum = 0.9)

# training
    train_model(net , datalaoder_dict, criterior, optimizer, num_epochs)
    
    if __name__ == "__main__":
        main()
        
        # Tạo network
        # use_pretrained = True
        # net = models.vgg16(pretrained = use_pretrained)
        # net.classifier[6] = nn.Linear(in_features = 4096, out_features = 2)
        
        # load_module(net, save_path)